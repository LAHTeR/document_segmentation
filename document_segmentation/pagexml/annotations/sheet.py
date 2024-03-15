import abc
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import pandas as pd
import requests
from pagexml.model.physical_document_model import PageXMLScan
from requests import HTTPError
from tqdm import tqdm

from ...settings import SERVER_PASSWORD, SERVER_USERNAME
from ..datamodel.document import Document
from ..datamodel.label import Label
from ..datamodel.page import Page
from ..inventory import InventoryReader


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

    def download(self, target_dir: Path, n: int = None) -> Iterable[Document]:
        """Download the data annotated in the sheet, and store as Json files..

        Args:
            target_dir (Path): The directory to store the downloaded data.
            n (int): The maximum number of documents to download.
                Defaults to None (all documents).
        """

        target_dir.mkdir(parents=True, exist_ok=True)

        with TemporaryDirectory() as cache_directory, requests.Session() as session:
            session.auth = (SERVER_USERNAME, SERVER_PASSWORD)

            inventory = None

            for idx, row in (
                self._data.head(n).sort_values(by=self._INV_NR_COLUMN).iterrows()
            ):
                target_file: Path = (target_dir / str(idx)).with_suffix(".json")
                try:
                    yield Document.from_json_file(target_file)
                except FileNotFoundError:
                    inv_nr = row[self._INV_NR_COLUMN]
                    assert inv_nr is not None, "Inventory number is None."

                    filter, reason = self._filter_row(row)
                    if filter:
                        tqdm.write(
                            f"Skipping row with inventory number {str(inv_nr)} due to reason: '{reason}'"
                        )
                    else:
                        begin_scan = row[self._START_PAGE_COLUMN]
                        end_scan = row[self._LAST_PAGE_COLUMN]

                        part = row.get(self._DEEL_VAN_INVENTARIS_COL)
                        if pd.isna(part) or part not in self._VALID_INVENTORY_PARTS:
                            part = None

                        if (
                            inventory._inv_nr != str(inv_nr)
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
                                page_xml: PageXMLScan = inventory.pagexml(page_number)
                            except HTTPError as e:
                                logging.error(
                                    f"Skipping inventory '{inv_nr}' due to error: {str(e)}"
                                )
                                break

                            if page_number == begin_scan:
                                label = Label.BEGIN
                            elif page_number == end_scan:
                                label = Label.END
                            else:
                                label = Label.IN

                            pages.append(Page.from_pagexml(label, inv_nr, page_xml))

                        document = Document(
                            id=str(idx),
                            inventory_nr=inv_nr,
                            inventory_part=part,
                            pages=pages,
                        )

                        with target_file.open("xt") as f:
                            f.write(document.model_dump_json())
                            f.write("\n")
                        yield document
