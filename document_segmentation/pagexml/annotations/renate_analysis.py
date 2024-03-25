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

        self._data[self._DEEL_VAN_INVENTARIS_COL] = (
            self._data[self._DEEL_VAN_INVENTARIS_COL].replace({"0": pd.NA}).fillna("")
        )


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
        self._inventory = Inventory.load_or_download(int(self._id[-4:]), "")

    def inventories(self) -> Iterable[Inventory]:
        yield self._inventory

    def inventory_numbers(self) -> Iterable[tuple[int, str]]:
        return [(self._inventory.inv_nr, self._inventory.inventory_part)]

    def annotate_inventory(self, inventory: Inventory) -> "Inventory":
        # TODO: handle multiple labels per scan
        default_label = "OUT"

        for idx, row in self._data.iterrows():
            scan_nr = int(idx[-4:])
            label = Label[row[self._LABEL_COLUMN].strip() or default_label]
            inventory.annotate_scan(scan_nr, label)

            if label == Label.BEGIN:
                default_label = "IN"
            elif label == Label.END:
                default_label = "OUT"

        return inventory
