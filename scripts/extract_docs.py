import argparse

from document_segmentation.model.page_sequence_tagger import PageSequenceTagger
from document_segmentation.pagexml.datamodel.inventory import Inventory
from document_segmentation.pagexml.datamodel.page import Page

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use model to segment inventories into documents."
    )

    parser.add_argument(
        "--model",
        type=argparse.FileType("rb"),
        required=True,
        help="The model to use for labelling.",
    )
    parser.add_argument(
        "--inventory",
        type=str,
        required=True,
        help="The inventory to segment in the format '<inv_nr>_<inv_part>'.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=argparse.FileType("wt"),
        default="-",
        help="The output file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        help="The device to use for the model (e.g. 'cuda:0'). Defaults to 'cpu'.",
    )

    args = parser.parse_args()

    model = PageSequenceTagger(device=args.device)
    model.load(args.model)

    inv = args.inventory
    inv_nr: list[str] = inv.split("_") if "_" in inv else [inv, ""]

    inventory = Inventory.load_or_download(*inv_nr)
    docs: list[list[Page]] = model.predict_documents(inventory)

    for doc in docs:
        print(
            f"Document from scan number {doc[0].scan_nr} to {doc[-1].scan_nr}.",
            file=args.output,
        )
