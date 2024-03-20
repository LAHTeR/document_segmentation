import abc
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from ...settings import INVENTORY_DIR, SERVER_PASSWORD, SERVER_USERNAME
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

    def _filter_row(self, row: pd.Series) -> bool:
        """Filter a row from the sheet.

        Args:
            row (pd.Series): The row to filter.
        Returns:
            bool: Whether to filter out the row.
            str: The reason for filtering out the row.
        """
        return False, None

    def set_labels(
        self, target_dir: Path = INVENTORY_DIR, n: int = None
    ) -> Iterable[Inventory]:
        """Download the data annotated in the sheet, and store as Json files..

        Args:
            target_dir (Path): The directory to store the downloaded data.
            n (int): The maximum number of documents to download.
                Defaults to None (all documents).
        """

        with requests.Session() as session:
            session.auth = (SERVER_USERNAME, SERVER_PASSWORD)

            for idx, row in self._data.head(n).iterrows():
                inv_nr = row[self._INV_NR_COLUMN]

                filter, reason = self._filter_row(row)
                if filter:
                    logging.warning(
                        f"Skipping row with inventory number {str(inv_nr)} due to reason: '{reason}'"
                    )
                    continue

                part = row.get(self._DEEL_VAN_INVENTARIS_COL)
                if pd.isna(part) or part not in self._VALID_INVENTORY_PARTS:
                    part = ""

                inventory = Inventory.load_or_download(inv_nr, part, target_dir)

                begin_scan = row[self._START_PAGE_COLUMN]
                end_scan = row[self._LAST_PAGE_COLUMN]

                inventory[begin_scan].label = Label.BEGIN
                inventory[end_scan].label = Label.END
                for page in range(begin_scan + 1, end_scan):
                    inventory[page].label = Label.IN

                yield inventory
