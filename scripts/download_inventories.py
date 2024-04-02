"""
Download specific inventories OR
Download all inventories specified in sheet(s)
"""

import argparse
import logging
from pathlib import Path

from requests import HTTPError
from tqdm import tqdm

from document_segmentation.pagexml.annotations.generale_missiven import GeneraleMissiven
from document_segmentation.pagexml.annotations.renate_analysis import (
    RenateAnalysis,
    RenateAnalysisInv,
)
from document_segmentation.pagexml.datamodel.inventory import Inventory
from document_segmentation.settings import (
    DATA_DIR,
    GENERALE_MISSIVEN_SHEET,
    RENATE_ANALYSIS_SHEETS,
    RENATE_TANAP_CATEGORISATION_SHEET,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download inventories")

    parser.add_argument(
        "--inventory",
        type=str,
        nargs="*",
        help="Inventory or inventories to download in the format '<inv_nr>_<inv_part>'.",
    )

    parser.add_argument(
        "--gm-sheet",
        type=Path,
        required=False,
        help="The sheet with input annotations in 'Generale Missiven' format.",
    )

    parser.add_argument(
        "--renate-categorisation-sheet",
        type=Path,
        required=False,
        help="The sheet with input annotations in 'Renate's TANAP Categorisation' format.",
    )
    parser.add_argument(
        "--renate-analysis-sheet",
        nargs="*",
        type=Path,
        required=False,
        help="The sheet with input annotations in 'Renate's inventory annotation' format.",
    )

    parser.add_argument(
        "--all-sheets",
        action="store_true",
        help=f"Download all inventories specified in the sheets found in '{DATA_DIR}'.",
    )

    args = parser.parse_args()

    if args.inventory is not None:
        for inv in tqdm(args.inventory, desc="Downloading inventories", unit="inv"):
            inv_nr, inv_part = inv.split("_") if "_" in inv else (inv, "")
            try:
                Inventory.download(int(inv_nr), inv_part)
            except FileExistsError as e:
                logging.warning(f"Skipping inventory {inv}: {e}")
            except HTTPError as e:
                raise ValueError(f"Failed to download inventory {inv}: {e}") from e

    sheets = []

    if args.all_sheets:
        sheets.append(GeneraleMissiven(GENERALE_MISSIVEN_SHEET))
        sheets.append(RenateAnalysis(RENATE_TANAP_CATEGORISATION_SHEET))
        sheets.extend([RenateAnalysisInv(sheet) for sheet in RENATE_ANALYSIS_SHEETS])
    else:
        if args.gm_sheet is not None:
            sheets.append(GeneraleMissiven(args.gm_sheet))
        if args.renate_categorisation_sheet is not None:
            sheets.append(RenateAnalysis(args.renate_categorisation_sheet))
        if args.renate_analysis_sheet:
            sheets.extend(
                [RenateAnalysis(sheet) for sheet in args.renate_analysis_sheet]
            )

    for sheet in sheets:
        sheet.download_inventories()
