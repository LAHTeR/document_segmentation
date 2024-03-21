import abc
import logging
from typing import Iterable, Optional

import pandas as pd
import requests
from tqdm import tqdm

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

    def __init__(self) -> None:
        super().__init__()

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

    def inventories(self, *, n: Optional[int] = None) -> Iterable[Inventory]:
        """Retrieve all inventories in the sheet.

        Args:
            n (int, optional): The maximum number of inventories to retrieve.
        Returns:
            Iterable[Inventory]: The inventories in the sheet.
        """
        keys = [self._INV_NR_COLUMN, self._DEEL_VAN_INVENTARIS_COL]

        count = 0
        for key, _ in tqdm(
            self._data.groupby(keys, dropna=False),
            desc="Loading inventories",
            total=n or self._data[keys].nunique(),
            unit="inventory",
        ):
            inv_nr, part = key

            try:
                count += 1
                yield Inventory.load_or_download(inv_nr, part if pd.notna(part) else "")
            except requests.HTTPError as e:
                logging.error(f"Failed to load inventory {inv_nr}: {e}")
            if n is not None and count >= n:
                break

    def annotate_inventory(self, inventory: Inventory) -> Inventory:
        """Annotate the inventory with the labels from the sheet in-place.

        Args:
            inventory (Inventory): The inventory to annotate.
        Returns:
            Inventory: The annotated inventory.
        """
        inventory_rows = self._data.loc[
            self._data[self._INV_NR_COLUMN] == inventory.inv_nr
        ]
        if len(inventory_rows) == 0:
            logging.warning(
                f"Inventory {inventory.inv_nr} not found in the sheet. Skipping."
            )

        for idx, row in inventory_rows.iterrows():
            begin_scan = row[self._START_PAGE_COLUMN]
            end_scan = row[self._LAST_PAGE_COLUMN]

            if all(page.label == Label.UNK for page in inventory[begin_scan:end_scan]):
                try:
                    inventory[begin_scan].label = Label.BEGIN
                    inventory[end_scan].label = Label.END
                    for page in range(begin_scan + 1, end_scan):
                        inventory[page].label = Label.IN
                except IndexError:
                    logging.warning(
                        f"Pages between {begin_scan} and {end_scan} not found in inventory {inventory.inv_nr}_{inventory.inventory_part} of length {len(inventory)}. Skipping."
                    )
            else:
                raise ValueError(
                    f"Pages between {begin_scan} and {end_scan} already have labels."
                )
        return Inventory
