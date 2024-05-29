import argparse
import random
from pathlib import Path

import torch

from document_segmentation.model.document_classifier import DocumentClassifier
from document_segmentation.model.page_learner import AbstractPageLearner
from document_segmentation.pagexml.annotations.renate_analysis import (
    RenateAnalysis,
    RenateAnalysisInv,
)
from document_segmentation.pagexml.datamodel.document import Document
from document_segmentation.settings import (
    RENATE_ANALYSIS_SHEETS,
    RENATE_TANAP_CATEGORISATION_SHEET,
)

if __name__ == "__main__":
    ########################################################################################
    # PARSE ARGUMENTS
    ########################################################################################
    arg_parser = argparse.ArgumentParser(
        description="Train a document classification model"
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
    arg_parser.add_argument("--split", type=float, default=0.8, help="Train/val split.")

    arg_parser.add_argument(
        "--max-documents",
        "--max",
        type=int,
        required=False,
        help="The maximum number of documents to read.",
    )
    arg_parser.add_argument(
        "--model-file",
        type=Path,
        default=Path("classifier_model.pt"),
        help="Output file for the model. Defaults to 'model.pt'.",
    )
    arg_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
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

    training_data: list[Document] = []
    validation_data: dict[str, list[Document]] = dict()

    sheet = RenateAnalysis(sheet_file=args.renate_categorisation_sheet)
    docs = list(sheet.documents(n=args.max_documents))
    train, validation = AbstractPageLearner.split(docs, split=args.split)
    training_data.extend(train)
    validation_data[args.renate_categorisation_sheet.name] = validation

    if args.renate_categorisation_sheet:
        docs = []
        for inv_sheet in args.renate_analysis_sheet:
            docs.extend(
                sheet.documents_from_sheet(
                    RenateAnalysisInv(sheet_file=inv_sheet), n=args.max_documents
                )
            )

        train, validation = AbstractPageLearner.split(docs, split=args.split)

        training_data.extend(train)
        validation_data["renate_analysis_inv"] = validation

    # TODO: add Generale Missiven

    model = DocumentClassifier()
    best_model = model.train_(training_data, validation_data, epochs=args.epochs)

    torch.save(best_model, args.model_file)
