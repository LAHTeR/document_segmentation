import argparse
from typing import Iterable

import pandas as pd
import torch
from tqdm import tqdm

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
        "--device",
        type=str,
        default="cpu",
        help="The device to use for the model (e.g. 'cuda:0'). Defaults to 'cpu'.",
    )

    args = parser.parse_args()

    # TODO: auto-determine the device
    model = torch.load(args.model, map_location=torch.device(args.device)).to_device(
        args.device
    )

    inventory_nrs: list[list[str]] = [
        inv.split("_") if "_" in inv else [inv, ""] for inv in args.inventory
    ]
    inventories: Iterable[Inventory] = (
        Inventory.load_or_download(*inv_nr) for inv_nr in inventory_nrs
    )

    results: pd.DataFrame = pd.concat(
        model.predict(inventory)
        for inventory in tqdm(
            inventories, total=len(args.inventory), desc="Predicting", unit="inventory"
        )
    )
    results.to_csv(args.output)
