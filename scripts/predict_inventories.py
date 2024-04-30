import argparse
import logging

from requests import HTTPError
from tqdm import tqdm

from document_segmentation.model.page_sequence_tagger import PageSequenceTagger
from document_segmentation.pagexml.datamodel.inventory import Inventory

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
        nargs="+",
        help="The inventories to segment in the format '<inv_nr>_<inv_part>'.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=argparse.FileType("wt"),
        required=True,
        help="The output file.",
    )
    parser.add_argument(
        "--format",
        choices=["google", "wandb"],
        required=False,
        help="The output format; add platform-specific formatting.",
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

    is_first_inventory: bool = True

    for inv in tqdm(args.inventory, desc="Predicting", unit="inventory"):
        inv_nr: list[str] = inv.split("_") if "_" in inv else [inv, ""]
        inventory: Inventory = None

        try:
            inventory = Inventory.load_or_download(*inv_nr)
        except HTTPError as e:
            logging.error(f"Failed to download inventory: {e}")
            continue

        results = model.predict(inventory)

        if args.format == "google":
            MODE = 3
            results["Thumbnail"] = (
                results["Thumbnail"]
                .dropna()
                .apply(lambda url: f'=IMAGE("{url}"; {MODE})')
            )
        elif args.format == "wandb":
            raise NotImplementedError()

        results.to_csv(
            args.output,
            mode="w" if is_first_inventory else "a",
            header=is_first_inventory,
        )

        is_first_inventory = False
