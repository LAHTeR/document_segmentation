import argparse
import csv
import logging
import random
from itertools import chain, groupby
from pathlib import Path
from typing import Iterable

import torch
import wandb

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
        type=str,
        default=GENERALE_MISSIVEN_SHEET,
        help="The sheet with input annotations (Generale Missiven).",
    )
    arg_parser.add_argument(
        "--renate-categorisation-sheet",
        type=str,
        default=RENATE_TANAP_CATEGORISATION_SHEET,
        help="The sheet with input annotations (Appendix F Renate Analysis).",
    )
    arg_parser.add_argument(
        "--renate-analysis-sheet",
        nargs="*",
        type=str,
        default=RENATE_ANALYSIS_SHEETS,
        help="The sheet with input annotations (Entire inventories from Renate's Analyses).",
    )
    arg_parser.add_argument(
        "--max-inventories",
        type=int,
        required=False,
        help="The maximum number of inventories to use per category. By default, down-sample to the smallest category.",
    )

    arg_parser.add_argument("--split", type=float, default=0.8, help="Train/val split.")

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

    inventories: dict[str, Iterable[Inventory]] = {}

    max_inventories: int = args.max_inventories

    if args.renate_analysis_sheet:
        inventories["renate_analysis_inv"] = [
            inventory
            for file in args.renate_analysis_sheet[:max_inventories]
            for inventory in RenateAnalysisInv(Path(file)).all_annotated_inventories()
        ]
        if max_inventories is None:
            # assuming this is the smallest category, so we down-sample to this
            max_inventories = len(inventories["renate_analysis_inv"])

    if args.gm_sheet:
        filepath = Path(args.gm_sheet)
        sheet = GeneraleMissiven(filepath)
        inventories[filepath.name] = list(
            sheet.all_annotated_inventories(n=max_inventories)
        )

    if args.renate_categorisation_sheet:
        filepath = Path(args.renate_categorisation_sheet)
        sheet = RenateAnalysis(filepath)
        inventories[filepath.name] = list(
            sheet.all_annotated_inventories(n=max_inventories)
        )

    training_inventories = {}
    validation_inventories = {}

    for sheet, _inventories in inventories.items():
        random.shuffle(_inventories)
        split = int(len(_inventories) * args.split)

        training_inventories[sheet] = _inventories[:split]
        validation_inventories[sheet] = _inventories[split:]

    ########################################################################################
    # LOAD OR TRAIN MODEL
    ########################################################################################
    model = PageSequenceTagger(device=args.device)

    weights = Inventory.total_class_weights(
        validation_inventories["renate_analysis_inv"]
    )
    best_model = model.train_(
        list(chain(*training_inventories.values())),
        validation_inventories,
        epochs=args.epochs,
        weights=weights,
    )
    torch.save(best_model, args.model_file)

    logging.debug(str(model))

    if model.wandb_run:
        artifact = wandb.Artifact(
            name=model.__class__.__name__ + "_model", type="model"
        )
        artifact.add_file(str(args.model_file))
        model.wandb_run.log_artifact(artifact)

        model.wandb_run.finish()

    ########################################################################################
    # EVALUATE MODEL
    # Log to local files, skip W&B logging
    ########################################################################################

    model.wandb_run = None
    model.load(args.model_file)

    print("Evaluating best model...")

    for name, validation in validation_inventories.items():
        print(f"Sheet: {name}", file=args.eval_output)
        print(f"Sheet: {name}", file=args.test_output)

        if validation:
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
                                for label, score in zip(
                                    Label, metric.compute().tolist()
                                )
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
        else:
            logging.warning(f"Empty validation set for '{name}', skipping.")
        print("=" * 80, file=args.eval_output)

        results[4].to_csv(
            args.test_output, sep="\t", index=False, header=True, mode="a"
        )
        print("=" * 80, file=args.test_output)
