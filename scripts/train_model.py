import argparse
import csv
import logging
import random
import sys
from itertools import groupby
from pathlib import Path

import torch

from document_segmentation.model.dataset import DocumentDataset
from document_segmentation.model.page_sequence_tagger import PageSequenceTagger
from document_segmentation.pagexml.annotations.generale_missiven import GeneraleMissiven
from document_segmentation.pagexml.annotations.sheet import download
from document_segmentation.pagexml.datamodel.label import Label
from document_segmentation.settings import (
    GENERALE_MISSIVEN_DOCUMENT_DIR,
    GENERALE_MISSIVEN_SHEET,
)

if __name__ == "__main__":
    ########################################################################################
    # PARSE ARGUMENTS
    ########################################################################################
    arg_parser = argparse.ArgumentParser(description="Train a model")

    arg_parser.add_argument(
        "--gm-sheet",
        type=Path,
        default=GENERALE_MISSIVEN_SHEET,
        help="The sheet with input annotations (Generale Missiven).",
    )
    arg_parser.add_argument("--split", type=float, default=0.8, help="Train/val split.")
    # TODO handle different sheet types

    arg_parser.add_argument(
        "-n", type=int, default=None, help="Maximum number of documents to use"
    )
    arg_parser.add_argument(
        "--model-file",
        type=Path,
        default=Path("model.pt"),
        help="Output file for the model",
    )

    arg_parser.add_argument(
        "--eval-output",
        type=argparse.FileType("xt"),
        default=sys.stdout,
        help="Output file for the evaluation.",
    )
    arg_parser.add_argument(
        "--test-output",
        type=argparse.FileType("xt"),
        default=sys.stdout,
        help="Output file for the evaluation.",
    )
    arg_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    arg_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")

    arg_parser.add_argument(
        "--device",
        "-d",
        type=str,
        choices=["cuda", "mps", "cpu"],
        required=False,
        help="The device to use for training. Auto-detects if not given.",
    )

    arg_parser.add_argument("--seed", type=int, required=False, help="Random Seed")

    args = arg_parser.parse_args()

    ########################################################################################
    # LOAD ANNOTATION SHEET AND DATA
    ########################################################################################
    random.seed(args.seed)

    # TODO: allow for more sheets
    sheet = GeneraleMissiven(sheet_file=args.gm_sheet)
    download(sheet, GENERALE_MISSIVEN_DOCUMENT_DIR, n=args.n)

    dataset = DocumentDataset.from_dir(GENERALE_MISSIVEN_DOCUMENT_DIR, n=args.n)
    dataset.shuffle()
    training_data, test_data = dataset.split(args.split)

    ########################################################################################
    # LOAD OR TRAIN MODEL
    ########################################################################################
    if args.model_file.exists():
        logging.info(f"Loading model from {args.model_file}")

        model = torch.load(args.model_file)
    else:
        logging.info("Training model from scratch")

        model = PageSequenceTagger(device=args.device)
        if args.device is not None:
            assert model.to_device(args.device)
        model.train_(
            training_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            weights=dataset.class_weights(),
        )
        torch.save(model, args.model_file)

    logging.debug(str(model))

    ########################################################################################
    # EVALUATE MODEL
    ########################################################################################
    metrics = model.eval_(test_data, args.batch_size, args.test_output)

    for average, metrics_group in groupby(
        sorted(metrics, key=lambda m: (m.average is None, m.average)),
        key=lambda m: m.average,
    ):
        if average is None:
            writer = csv.DictWriter(
                args.eval_output,
                fieldnames=["Metric"] + [label.name for label in Label],
                delimiter="\t",
            )
            writer.writeheader()

            for metric in metrics_group:
                scores: list[float] = metric.compute().tolist()

                writer.writerow(
                    {"Metric": metric.__class__.__name__}
                    | {
                        label.name: f"{score:.4f}"
                        for label, score in zip(Label, scores)
                    }
                )
            args.eval_output.flush()
        else:
            for metric in metrics_group:
                score: float = metric.compute().item()
                print(
                    f"{metric.__class__.__name__} ({average} average):\t{score:.4f}",
                    file=args.eval_output,
                )
