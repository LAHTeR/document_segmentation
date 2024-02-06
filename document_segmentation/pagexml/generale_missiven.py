import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import pandas as pd
import requests
from requests import HTTPError
from tqdm import tqdm

from ..settings import GENERALE_MISSIVEN_SHEET, SERVER_PASSWORD, SERVER_USERNAME
from .datamodel import Document, Label, Page
from .inventory import InventoryReader


class GeneraleMissiven:
    """Helper class to handle the Generale Missiven spreadsheet."""

    _VALID_INVENTORY_PARTS = {"A", "B", "C"}
    _SKIP_MESSAGE = "Niet gedigitaliseerd."

    _INDEX_COLUMN = "ID"
    _INV_NR_COLUMN = "Inv.nr. Nationaal Archief (1.04.02)"
    _START_PAGE_COLUMN = "Beginscan"
    _LAST_PAGE_COLUMN = "Eindscan"
    _DEEL_VAN_INVENTARIS_COL = "Deel v. inventarisnummer"
    _STATUS_COLUMN = "Problemen gevonden tijdens handmatige check:"
    _SEPARATOR = ";"

    def __init__(self, sheet_file: Path = GENERALE_MISSIVEN_SHEET) -> None:
        """Load the spreadsheet into memory.

        Args:
            sheet_file (Path, optional): Path to the spreadsheet. Defaults to settings.GENERALE_MISSIVEN_SHEET.
        """
        self._data = pd.read_csv(
            sheet_file,
            sep=self._SEPARATOR,
            index_col=self._INDEX_COLUMN,
            dtype={
                self._INDEX_COLUMN: str,
                self._INV_NR_COLUMN: pd.Int64Dtype(),
                self._DEEL_VAN_INVENTARIS_COL: str,
                self._START_PAGE_COLUMN: pd.Int64Dtype(),
                self._LAST_PAGE_COLUMN: pd.Int64Dtype(),
            },
        ).dropna(
            subset={
                self._INV_NR_COLUMN,
                self._START_PAGE_COLUMN,
                self._LAST_PAGE_COLUMN,
            }
        )

    def __len__(self):
        return len(self._data)

    def _pagexml(self, id, page):
        logging.warning(
            "DEPRECATED: This method instantiates an InventoryReader for every page."
        )

        inv_nr = self._data.loc[id, self._INV_NR_COLUMN]

        return InventoryReader(inv_nr).pagexml(page)

    def pagexml_first_page(self, id):
        """Get the PageXML for the first page of a row in the spreadsheet.

        Args:
            id: The ID of the row in the spreadsheet.
        """
        return self._pagexml(id, self._data.loc[id, self._START_PAGE_COLUMN])

    def pagexml_last_page(self, id):
        """Get the PageXML for the last page of a row in the spreadsheet.

        Args:
            id: The ID of the row in the spreadsheet.
        """
        return self._pagexml(id, self._data.loc[id, self._LAST_PAGE_COLUMN])

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
                if row[self._STATUS_COLUMN] == self._SKIP_MESSAGE:
                    tqdm.write(
                        f"Skipping row with inventory number {str(inv_nr)} due to status message: '{row[self._STATUS_COLUMN]}'"
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
