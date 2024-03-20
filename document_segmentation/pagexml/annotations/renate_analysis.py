from pathlib import Path
from typing import Iterable

import pandas as pd

from document_segmentation.pagexml.datamodel.inventory import Inventory

from ...settings import RENATE_TANAP_CATEGORISATION_SHEET
from ..datamodel.label import Label
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
        self._inventory = Inventory.load_or_download(self.int(self._id[-4:]))

    def set_labels(self) -> Iterable[Inventory]:
        default_label = "OUT"

        for idx, row in self._data.iterrows():
            page = int(idx[-4:])
            label = Label[row[self._LABEL_COLUMN].strip() or default_label]

            # Subtract 1 from page number because they start counting at 1
            self._inventory[page - 1].label = label

            if label == Label.BEGIN:
                default_label = "IN"
            elif label == Label.END:
                default_label = "OUT"

        yield self._inventory
