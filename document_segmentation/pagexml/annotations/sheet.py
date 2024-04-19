import abc
import logging
from itertools import islice
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests
from tqdm import tqdm

from ...settings import (
    INVENTORY_DIR,
    MAX_INVENTORY_SIZE,
    MIN_INVENTORY_SIZE,
    MIN_REGION_TEXT_LENGTH,
    SERVER_PASSWORD,
    SERVER_USERNAME,
)
from ..datamodel.inventory import Inventory
from ..datamodel.label import Label
from ..datamodel.page import Page


class Sheet(abc.ABC):
    """Abstract class for reading sheet annotations."""

    _INDEX_COLUMN = "Document_ID"
    _INV_NR_COLUMN = "Inv.nr. Nationaal Archief (1.04.02)"
    _START_PAGE_COLUMN = "Begin scan"
    _LAST_PAGE_COLUMN = "End scan"
    _DEEL_VAN_INVENTARIS_COL = "Part of the inv.nr."
    _SKIP_MESSAGE = ""

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

    def download_inventories(
        self, *, username: str = SERVER_USERNAME, password: str = SERVER_PASSWORD
    ):
        session = requests.Session()
        session.auth = (SERVER_USERNAME, SERVER_PASSWORD)

        for inv_nr, part in tqdm(
            list(self.inventory_numbers()), desc="Downloading inventories", unit="inv"
        ):
            inv_str = f"{inv_nr}_{part}"
            try:
                Inventory.download(
                    inv_nr, part, target_directory=self._inventory_dir, session=session
                )
            except FileExistsError as e:
                logging.info(f"Skipping inventory {inv_str}: {e}")
            except requests.HTTPError as e:
                logging.error(f"Failed to load inventory {inv_str}: {e}")

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
        """Annotate the inventory with the labels from the sheet and generate new inventory.

        Pages between BEGIN and END scans are labelled as IN.

        Before the first and after each document, an empty page is added.

        Args:
            inventory (Inventory): The inventory to annotate.
        Returns:
            Inventory: The annotated inventory.
        """
        annotated_rows: pd.DataFrame = self._data.loc[
            self._data[self._INV_NR_COLUMN] == inventory.inv_nr
        ]

        if inventory.inventory_part:
            annotated_rows = annotated_rows.loc[
                annotated_rows[self._DEEL_VAN_INVENTARIS_COL]
                == inventory.inventory_part
            ]

        if len(annotated_rows) == 0:
            logging.warning(
                f"No entries found for inventory {inventory} in sheet '{self}'. Skipping."
            )

        documents: list[list[Page]] = []
        for _, row in annotated_rows.sort_values(self._START_PAGE_COLUMN).iterrows():
            begin_scan = row[self._START_PAGE_COLUMN]
            end_scan = row[self._LAST_PAGE_COLUMN]

            try:
                document: list[Page] = (
                    [inventory.get_scan(begin_scan).annotate(Label.BEGIN)]
                    + [
                        inventory.get_scan(scan_nr).annotate(Label.IN)
                        for scan_nr in range(begin_scan + 1, end_scan)
                    ]
                    + [inventory.get_scan(end_scan).annotate(Label.END)]
                )
                # remove redundant BEGIN_END pages
                redundant: list[int] = []
                for i, page in enumerate(document[:-1]):
                    if page.label == Label.END_BEGIN:
                        next_page = document[i + 1]
                        if next_page.scan_nr == page.scan_nr:
                            assert (
                                next_page.label == Label.END_BEGIN
                            ), f"Expected two subsequent END_BEGIN labels, got '{page}', '{next_page}'"
                            redundant.append(i)
                for i in reversed(redundant):
                    document.pop(i)

                documents.append(document)
            except ValueError as e:
                logging.error(str(e))

        # Merge documents to inventory with empty pages in between
        if documents:
            pages: list[Page] = []
            last_scan: int = 0
            for document in documents:
                if last_scan == 0 or document[0].scan_nr - 1 > last_scan:
                    # Insert empty page between documents
                    pages.append(
                        Page(
                            label=Label.OUT,
                            scan_nr=last_scan + 1,
                            external_ref="",
                            regions=[],
                        )
                    )
                pages.extend(document)
                last_scan = pages[-1].scan_nr if pages else 0

            # Insert empty page at the end
            pages.append(
                Page(
                    label=Label.OUT, scan_nr=last_scan + 1, external_ref="", regions=[]
                )
            )
        else:
            pages = []

        return Inventory(
            inv_nr=inventory.inv_nr,
            inventory_part=inventory.inventory_part,
            pages=pages,
        )

    def preprocess(
        self, inventory: Inventory, min_region_text_length, max_size: int
    ) -> Inventory:
        return (
            self.annotate_inventory(inventory)
            .remove_short_regions(min_chars=min_region_text_length)
            .empty_unlabelled()
            .head(max_size)
        )

    def all_annotated_inventories(
        self,
        n: Optional[int] = None,
        *,
        min_region_text_length=MIN_REGION_TEXT_LENGTH,
        skip_errors: bool = True,
        max_size: int = MAX_INVENTORY_SIZE,
        min_size: int = MIN_INVENTORY_SIZE,
    ) -> Iterable["Inventory"]:
        """Load, label, and preprocess all inventories in the sheet.

        Args:
            n (Optional[int], optional): The number of inventories to load.
                Defaults to None.
            min_region_text_length ([type], optional): The minimum length of text in a region.
                Defaults to MIN_REGION_TEXT_LENGTH.
            skip_errors (bool, optional): If True (default), errors are logged, otherwise they are raised.
            max_size (Optional[int], optional): The maximum number of pages per inventory. Larger inventories are cut off.
                Defaults to MAX_INVENTORY_SIZE. Set to 0 or None to disable.
        """
        inventories = (
            inventory
            for inventory in self.inventories()
            if (not MIN_INVENTORY_SIZE) or (len(inventory) >= min_size)
        )
        for inventory in tqdm(
            islice(inventories, n),
            desc=f"Loading Inventories ({self.__class__.__name__})",
            total=n or len(list(self.inventory_numbers())),
            unit="inventory",
        ):
            try:
                preprocessed: Inventory = self.preprocess(
                    inventory, min_region_text_length, max_size
                )
                if (not MIN_INVENTORY_SIZE) or (len(preprocessed) >= min_size):
                    yield preprocessed
                else:
                    logging.warning(
                        f"Inventory {inventory} is too small after preprocessing. Skipping."
                    )
            except ValueError as e:
                if skip_errors:
                    logging.error(str(e))
                else:
                    raise e
