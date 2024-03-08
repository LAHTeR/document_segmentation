import argparse
import csv
import logging
import random
import sys
from itertools import groupby
from pathlib import Path
from typing import Iterable

import torch

from document_segmentation.model.dataset import AbstractDataset, DocumentDataset
from document_segmentation.model.page_sequence_tagger import PageSequenceTagger
from document_segmentation.pagexml.annotations.generale_missiven import GeneraleMissiven
from document_segmentation.pagexml.annotations.renate_analysis import (
    RenateAnalysis,
    RenateAnalysisInv,
)
from document_segmentation.pagexml.annotations.sheet import Sheet
from document_segmentation.pagexml.datamodel.document import Document
from document_segmentation.pagexml.datamodel.label import Label
from document_segmentation.settings import (
    GENERALE_MISSIVEN_DOCUMENT_DIR,
    GENERALE_MISSIVEN_SHEET,
    RENATE_ANALYSIS_DIR,
    RENATE_ANALYSIS_SHEETS,
    RENATE_TANAP_CATEGORISATION_SHEET,
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
    arg_parser.add_argument(
        "--renate-categorisation-sheet",
        type=Path,
        default=RENATE_TANAP_CATEGORISATION_SHEET,
        help="The sheet with input annotations (Generale Missiven).",
    )
    arg_parser.add_argument(
        "--renate-analysis-sheet",
        nargs="*",
        type=Path,
        default=RENATE_ANALYSIS_SHEETS,
        help="The sheet with input annotations (Generale Missiven).",
    )

    arg_parser.add_argument("--split", type=float, default=0.8, help="Train/val split.")

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

    random.seed(args.seed)

    ########################################################################################
    # LOAD ANNOTATION SHEETS AND DATA
    ########################################################################################

    sheets: list[Sheet] = []

    if args.gm_sheet:
        sheets.append(GeneraleMissiven(args.gm_sheet))
    if args.renate_categorisation_sheet:
        sheets.append(RenateAnalysis(args.renate_categorisation_sheet))
    sheets.extend([RenateAnalysisInv(sheet) for sheet in args.renate_analysis_sheet])

    training_sets: list[AbstractDataset] = []
    test_sets: list[AbstractDataset] = []

    for sheet in sheets:
        if isinstance(sheet, GeneraleMissiven):
            target_dir = GENERALE_MISSIVEN_DOCUMENT_DIR
        else:
            target_dir = RENATE_ANALYSIS_DIR
        sheet.download(target_dir, args.n)

        # TODO add name to dataset for evaluation output
        documents: Iterable[Document] = sheet.to_documents(n=args.n)
        dataset: AbstractDataset = DocumentDataset.from_documents(documents)
        dataset.shuffle()
        train, test = dataset.split(args.split)

        training_sets.append(train)
        test_sets.append(test)

    training_data: DocumentDataset = sum(training_sets, DocumentDataset([]))
    training_data.shuffle()

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
            weights=training_data.class_weights(),
        )
        torch.save(model, args.model_file)

    logging.debug(str(model))

    ########################################################################################
    # EVALUATE MODEL
    ########################################################################################
    for test_set in test_sets:
        metrics = model.eval_(test_set, args.batch_size, args.test_output)

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
