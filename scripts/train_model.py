import argparse
import csv
import logging
import random
from itertools import groupby
from pathlib import Path

from document_segmentation.model.page_sequence_tagger import PageSequenceTagger
from document_segmentation.pagexml.annotations.generale_missiven import GeneraleMissiven
from document_segmentation.pagexml.annotations.renate_analysis import (
    RenateAnalysis,
    RenateAnalysisInv,
)
from document_segmentation.pagexml.datamodel.inventory import Inventory
from document_segmentation.pagexml.datamodel.label import Label
from document_segmentation.settings import (
    GENERALE_MISSIVEN_SHEET,
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
        help="Output file for the model. Defaults to 'model.pt'.",
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

    random.seed(args.seed)

    ########################################################################################
    # LOAD ANNOTATION SHEETS AND DATA
    ########################################################################################

    training_inventories: list[Inventory] = []
    validation_inventories: dict[str, list[Inventory]] = {}

    if args.gm_sheet:
        sheet = GeneraleMissiven(args.gm_sheet)
        _inventories = list(sheet.all_annotated_inventories(n=args.n))

        random.shuffle(_inventories)
        split = int(len(_inventories) * args.split)
        training_inventories.extend(_inventories[:split])
        validation_inventories[args.gm_sheet.name] = _inventories[split:]

    if args.renate_categorisation_sheet:
        sheet = RenateAnalysis(args.renate_categorisation_sheet)
        _inventories = list(sheet.all_annotated_inventories(n=args.n))

        random.shuffle(_inventories)
        split = int(len(_inventories) * args.split)
        training_inventories.extend(_inventories[:split])
        validation_inventories[args.renate_categorisation_sheet.name] = _inventories[
            split:
        ]

    _inventories = []
    sheet_name = "renate_analysis_inv"
    validation_inventories[sheet_name] = []

    for sheet in args.renate_analysis_sheet:
        sheet = RenateAnalysisInv(sheet)
        _inventories.extend(list(sheet.all_annotated_inventories(n=args.n)))

    random.shuffle(_inventories)
    split = int(len(_inventories) * args.split)
    training_inventories.extend(_inventories[:split])
    validation_inventories[sheet_name] = _inventories[split:]

    ########################################################################################
    # LOAD OR TRAIN MODEL
    ########################################################################################
    model = PageSequenceTagger(device=args.device)

    model.train_(training_inventories, validation_inventories, epochs=args.epochs)
    model.save(args.model_file)

    logging.debug(str(model))
    model.wandb_run.finish()

    ########################################################################################
    # EVALUATE MODEL
    # Log to local files, skip W&B logging
    ########################################################################################

    model.wandb_run = None

    for name, validation in validation_inventories.items():
        print(f"Sheet: {name}", file=args.eval_output)
        print(f"Sheet: {name}", file=args.test_output)

        results = model.eval_(validation, name)
        metrics = results[:4]
        table = results[4]

        for average, _metrics in groupby(metrics, key=lambda m: m.average):
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
        print("=" * 80, file=args.eval_output)

        results[4].to_csv(
            args.test_output, sep="\t", index=False, header=True, mode="a"
        )
        print("=" * 80, file=args.test_output)
