import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import pandas as pd

from ...settings import RENATE_TANAP_CATEGORISATION_SHEET
from ..datamodel.document import Document
from ..datamodel.label import Label
from ..datamodel.page import Page
from ..inventory import InventoryReader
from .sheet import Sheet


class RenateAnalysis(Sheet):
    def __init__(self, sheet: Path = RENATE_TANAP_CATEGORISATION_SHEET) -> None:
        """Helper class to handle the 'Renate Analysis' sheet.

        Args:
            sheet (Path, optional): Paths to the spreadsheet.
                Defaults to settings.RENATE_TANAP_CATEGORISATION_SHEET.
        """
        super().__init__()

        self._data = pd.read_excel(
            sheet, dtype=self._dtypes, index_col=self._INDEX_COLUMN
        ).dropna(subset=self._dropna)


class RenateAnalysisInv(Sheet):
    _INDEX_COLUMN = "Scan File_Name"
    _PAGE_COLUMN = "Page"
    _LABEL_COLUMN = "TANAP Boundaries"

    def __init__(self, path: Path) -> None:
        super().__init__()

        self._data = pd.read_excel(path, index_col=self._INDEX_COLUMN).fillna("")
        self._data[self._LABEL_COLUMN] = self._data[self._LABEL_COLUMN].str.replace(
            "START", "BEGIN"
        )

        self._id = path.stem
        self._inv_nr = int(self._id[-4:])

    def download(self, target_dir: Path, n: int = None) -> Iterable[Document]:
        target_dir.mkdir(parents=True, exist_ok=True)

        with TemporaryDirectory() as cache_directory:
            inventory = InventoryReader(
                self._inv_nr, cache_directory=Path(cache_directory)
            )
            fallback_label = "OUT"
            pages = []

            for idx, row in self._data.head(n).iterrows():
                page = int(idx[-4:])
                page_xml = inventory.pagexml(page)

                document_file = target_dir / f"{idx}.json"
                try:
                    yield Document.from_json_file(document_file)
                except FileNotFoundError:
                    try:
                        label = Label[row[self._LABEL_COLUMN].strip() or fallback_label]
                    except KeyError as e:
                        raise ValueError(
                            f"Invalid label '{row[self._LABEL_COLUMN]}' in inventory '{self._id}' for page '{idx}'."
                        ) from e

                    pages.append(Page.from_pagexml(label, self._inv_nr, page_xml))

                    if label == Label.BEGIN:
                        logging.info("Beginning of document.")
                        fallback_label = "IN"
                    elif label == Label.END:
                        logging.info("End of document.")

                        document = Document(
                            id=idx, inventory_nr=self._inv_nr, pages=pages.copy()
                        )
                        with document_file.open("xt") as f:
                            f.write(document.model_dump_json())
                            f.write("\n")

                        yield document

                        pages = []
                        fallback_label = "OUT"
