import abc
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import pandas as pd
import requests
from requests import HTTPError
from tqdm import tqdm

from ...settings import SERVER_PASSWORD, SERVER_USERNAME
from ..datamodel.document import Document
from ..datamodel.label import Label
from ..datamodel.page import Page
from ..inventory import InventoryReader


class Sheet(abc.ABC):
    """Abstract class for sheet annotations."""

    _INDEX_COLUMN = "Document_ID"
    _INV_NR_COLUMN = "Inv.nr. Nationaal Archief (1.04.02)"
    _START_PAGE_COLUMN = "Begin scan"
    _LAST_PAGE_COLUMN = "Eind scan"
    _DEEL_VAN_INVENTARIS_COL = "Part of the inv.nr."
    _SEPARATOR = ";"

    _VALID_INVENTORY_PARTS = {"A", "B", "C"}
    _SKIP_MESSAGE = "Niet gedigitaliseerd."

    def __len__(self):
        return len(self._data)

    def _filter_row(self, row: pd.Series) -> bool:
        """Filter rows from the spreadsheet.

        Args:
            row: A row from the spreadsheet.

        Returns:
            bool: True if the row should be included, False otherwise.
        """
        return True

    def to_documents(
        self, *, skip_errors: bool = True, n: int = None, skip_ids: set[str] = None
    ) -> Iterable[Document]:
        """ "Generate a Document for each row in the spreadsheet.

        Args:
            skip_errors (bool, optional): If True, skip inventories that failed to download.
                Otherwise, raise the exception. Defaults to True.
            n: The number of documents to generate. Defaults to None (all documents).
            skip_ids: A set of IDs to skip. Defaults to None.
        """
        inventory = None

        with TemporaryDirectory() as cache_directory, requests.Session() as session:
            session.auth = (SERVER_USERNAME, SERVER_PASSWORD)

            for idx, row in (
                self._data.head(n).sort_values(by=self._INV_NR_COLUMN).iterrows()
            ):
                if skip_ids and idx in skip_ids:
                    continue

                inv_nr = row[self._INV_NR_COLUMN]
                filter, reason = self._filter_row(row)
                if not filter:
                    tqdm.write(
                        f"Skipping row with inventory number {str(inv_nr)} due to status message: '{reason}'"
                    )
                    continue

                begin_scan = row[self._START_PAGE_COLUMN]
                end_scan = row[self._LAST_PAGE_COLUMN]

                part = row[self._DEEL_VAN_INVENTARIS_COL]
                if pd.isna(part) or part not in self._VALID_INVENTORY_PARTS:
                    part = None

                if (
                    inventory is None
                    or inventory._inv_nr != str(inv_nr)
                    or inventory._inventory_part != part
                ):
                    inventory = InventoryReader(
                        inv_nr,
                        inventory_part=part,
                        cache_directory=Path(cache_directory),
                        session=session,
                    )

                pages: list[Page] = []
                for page_number in range(begin_scan, end_scan + 1):
                    try:
                        page_xml = inventory.pagexml(page_number)
                    except HTTPError as e:
                        if skip_errors:
                            tqdm.write(
                                f"Skipping inventory '{inventory.inv_nr}' due to error: {str(e)}"
                            )
                            break
                        else:
                            raise e

                    if page_number == begin_scan:
                        label = Label.BEGIN
                    elif page_number == end_scan:
                        label = Label.END
                    else:
                        label = Label.IN

                    pages.append(Page.from_pagexml(label, page_number, page_xml))

                yield Document(
                    id=idx, inventory_nr=inv_nr, inventory_part=part, pages=pages
                )
