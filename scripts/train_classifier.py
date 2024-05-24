import argparse
import random
from pathlib import Path

import torch

from document_segmentation.model.document_classifier import DocumentClassifier
from document_segmentation.pagexml.annotations.renate_analysis import RenateAnalysis
from document_segmentation.pagexml.datamodel.document import Document
from document_segmentation.settings import RENATE_TANAP_CATEGORISATION_SHEET

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

    sheet = RenateAnalysis(sheet_file=args.renate_categorisation_sheet)
    documents: list[Document] = list(sheet.documents(n=args.max_documents))

    assert all(doc.label for doc in documents), "Some documents have no label."

    # TODO: add Generale Missiven
    # TODO: add entire inventories, including front matters

    random.shuffle(documents)
    split = int(len(documents) * 0.8)
    training_data = documents[:split]
    validation_data = documents[split:]

    model = DocumentClassifier()
    best_model = model.train_(training_data, validation_data, epochs=args.epochs)

    torch.save(best_model, args.model_file)
