import abc
import logging
from itertools import islice
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests
from tqdm import tqdm

from ...settings import INVENTORY_DIR, MIN_REGION_TEXT_LENGTH
from ..datamodel.inventory import Inventory
from ..datamodel.label import Label


class Sheet(abc.ABC):
    """Abstract class for reading sheet annotations."""

    _INDEX_COLUMN = "Document_ID"
    _INV_NR_COLUMN = "Inv.nr. Nationaal Archief (1.04.02)"
    _START_PAGE_COLUMN = "Begin scan"
    _LAST_PAGE_COLUMN = "End scan"
    _DEEL_VAN_INVENTARIS_COL = "Part of the inv.nr."
    _SKIP_MESSAGE = ""

    _VALID_INVENTORY_PARTS = {"A", "B", "C"}

    def __init__(self, *, inventory_dir: Path = INVENTORY_DIR) -> None:
        """Initialize the sheet.

        Args:
            inventory_dir (Path, optional): The directory where the inventories are stored.
                Defaults to INVENTORY_DIR.
        """
        super().__init__()

        self._inventory_dir = inventory_dir

        self._dtypes = {
            self._INDEX_COLUMN: str,
            self._INV_NR_COLUMN: pd.Int64Dtype(),
            self._DEEL_VAN_INVENTARIS_COL: str,
            self._START_PAGE_COLUMN: pd.Int64Dtype(),
            self._LAST_PAGE_COLUMN: pd.Int64Dtype(),
        }
        self._dropna = {
            self._INV_NR_COLUMN,
            self._START_PAGE_COLUMN,
            self._LAST_PAGE_COLUMN,
        }

    def __len__(self):
        return len(self._data)

    def inventory_numbers(self) -> Iterable[tuple[int, str]]:
        key_fields = [self._INV_NR_COLUMN, self._DEEL_VAN_INVENTARIS_COL]

        for keys, _ in self._data.groupby(key_fields, dropna=False):
            yield keys

    def inventories(self) -> Iterable[Inventory]:
        """Retrieve all inventories in the sheet.

        Returns:
            Iterable[Inventory]: The inventories in the sheet.
        """
        for inv_nr, part in self.inventory_numbers():
            try:
                yield Inventory.load_or_download(inv_nr, part, self._inventory_dir)
            except requests.HTTPError as e:
                logging.error(f"Failed to load inventory {inv_nr}: {e}")

    def annotate_inventory(self, inventory: Inventory) -> Inventory:
        """Annotate the inventory with the labels from the sheet in-place.

        Args:
            inventory (Inventory): The inventory to annotate.
        Returns:
            Inventory: The annotated inventory.
        """
        inventory_rows = self._data.loc[
            (self._data[self._INV_NR_COLUMN] == inventory.inv_nr)
            & (self._data[self._DEEL_VAN_INVENTARIS_COL] == inventory.inventory_part)
        ]
        if len(inventory_rows) == 0:
            logging.warning(
                f"Inventory {inventory.inv_nr} not found in the sheet. Skipping."
            )

        for idx, row in inventory_rows.iterrows():
            begin_scan = row[self._START_PAGE_COLUMN]
            end_scan = row[self._LAST_PAGE_COLUMN]

            try:
                inventory.annotate_scan(begin_scan, Label.BEGIN)
                inventory.annotate_scan(end_scan, Label.END)
                for scan_nr in range(begin_scan + 1, end_scan):
                    inventory.annotate_scan(scan_nr, Label.IN)
            except ValueError as e:
                logging.error(str(e))
        return inventory

    def all_annotated_inventories(
        self,
        n: Optional[int] = None,
        *,
        min_region_text_length=MIN_REGION_TEXT_LENGTH,
        skip_errors: bool = True,
    ) -> Iterable[Inventory]:
        """Load, label, and preprocess all inventories in the sheet.

        Args:
            n (Optional[int], optional): The number of inventories to load.
                Defaults to None.
            min_region_text_length ([type], optional): The minimum length of text in a region.
                Defaults to MIN_REGION_TEXT_LENGTH.
            skip_errors (bool, optional): If True (default), errors are logged, otherwise they are raised.
        """

        for inventory in tqdm(
            islice(self.inventories(), n),
            desc="Loading Inventories",
            total=n or len(list(self.inventory_numbers())),
            unit="inventory",
        ):
            self.annotate_inventory(inventory)

            try:
                for labelled in inventory.labelled_inventories():
                    yield labelled.remove_short_regions(min_region_text_length)
            except ValueError as e:
                if skip_errors:
                    logging.error(str(e))
                else:
                    raise e
