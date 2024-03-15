import argparse
import csv
import logging
import random
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
        default="eval.csv",
        help="Output file for the evaluation.",
    )
    arg_parser.add_argument(
        "--test-output",
        type=argparse.FileType("xt"),
        default="test.out.txt",
        help="Output file for the evaluation.",
    )

    training_args = arg_parser.add_argument_group("Training Arguments")
    training_args.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    training_args.add_argument("--batch-size", type=int, default=64, help="Batch size")

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

    if args.model_file.exists():
        arg_parser.error(f"Model file {args.model_file} already exists.")

    random.seed(args.seed)

    ########################################################################################
    # LOAD ANNOTATION SHEETS AND DATA
    ########################################################################################

    document_sets: list[Iterable[Document]] = []
    sheet_paths: list[Path] = []

    training_sets: list[AbstractDataset] = []
    test_sets: list[AbstractDataset] = []

    if args.gm_sheet:
        sheet = GeneraleMissiven(args.gm_sheet)
        document_sets.append(sheet.download(GENERALE_MISSIVEN_DOCUMENT_DIR, args.n))
        sheet_paths.append(args.gm_sheet)
    if args.renate_categorisation_sheet:
        sheet = RenateAnalysis(args.renate_categorisation_sheet)
        document_sets.append(sheet.download(RENATE_ANALYSIS_DIR, args.n))
        sheet_paths.append(args.renate_categorisation_sheet)
    for _sheet in args.renate_analysis_sheet:
        sheet = RenateAnalysisInv(_sheet)
        document_sets.append(sheet.download(RENATE_ANALYSIS_DIR, args.n))
        sheet_paths.append(_sheet)

    training_sets: list[AbstractDataset] = []
    test_sets: list[AbstractDataset] = []

    for documents in document_sets:
        dataset: DocumentDataset = DocumentDataset.from_documents(documents)
        dataset.shuffle()
        train, test = dataset.split(args.split)

        training_sets.append(train)
        test_sets.append(test)

    training_data: DocumentDataset = sum(training_sets, DocumentDataset([]))
    training_data.shuffle()

    ########################################################################################
    # LOAD MODEL
    ########################################################################################
    model = PageSequenceTagger(device=args.device)

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
    for test_set, sheet_path in zip(test_sets, sheet_paths, strict=True):
        print(f"Sheet: {sheet_path}", file=args.eval_output, flush=True)
        print(f"Sheet: {sheet_path}", file=args.test_output, flush=True)

        metrics = model.eval_(test_set, args.batch_size, args.test_output)

        for average, _metrics in groupby(
            sorted(metrics, key=lambda metric: metric.average is None),
            key=lambda m: m.average,
        ):
            if average is None:
                writer = csv.DictWriter(
                    args.eval_output,
                    fieldnames=["Metric"] + [label.name for label in Label],
                    delimiter="\t",
                )
                writer.writeheader()

                for metric in _metrics:
                    writer.writerow(
                        {"Metric": metric.__class__.__name__}
                        | {
                            label.name: f"{score:.4f}"
                            for label, score in zip(Label, metric.compute().tolist())
                        }
                    )
            else:
                for metric in _metrics:
                    score: float = metric.compute().item()
                    print(
                        f"{metric.__class__.__name__} ({average} average):\t{score:.4f}",
                        file=args.eval_output,
                        flush=True,
                    )
            args.eval_output.flush()

        print("=" * 80, file=args.eval_output, flush=True)
        print("=" * 80, file=args.test_output, flush=True)
