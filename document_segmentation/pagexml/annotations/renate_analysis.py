import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from ...settings import (
    INVENTORY_DIR,
    MIN_REGION_TEXT_LENGTH,
    RENATE_TANAP_CATEGORISATION_SHEET,
)
from ..datamodel.inventory import Inventory
from ..datamodel.label import Label
from ..datamodel.page import Page
from .sheet import Sheet


class RenateAnalysis(Sheet):
    def __init__(
        self,
        sheet_file: Path = RENATE_TANAP_CATEGORISATION_SHEET,
        *,
        inventory_dir: Path = INVENTORY_DIR,
    ) -> None:
        """Helper class to handle the 'Renate Analysis' sheet.

        Args:
            sheet_file (Path, optional): Paths to the spreadsheet.
                Defaults to settings.RENATE_TANAP_CATEGORISATION_SHEET.
        """
        super().__init__(inventory_dir=inventory_dir)

        self._data = pd.read_excel(
            sheet_file, dtype=self._dtypes, index_col=self._INDEX_COLUMN
        ).dropna(subset=self._dropna)

        self._data[self._DEEL_VAN_INVENTARIS_COL] = (
            self._data[self._DEEL_VAN_INVENTARIS_COL].replace({"0": pd.NA}).fillna("")
        )


class RenateAnalysisInv(Sheet):
    _INDEX_COLUMN = "Scan File_Name"
    _PAGE_COLUMN = "Page"
    _LABEL_COLUMN = "TANAP Boundaries"

    def __init__(
        self, sheet_file: Path, *, inventory_dir: Path = INVENTORY_DIR
    ) -> None:
        super().__init__(inventory_dir=inventory_dir)

        self._data = pd.read_excel(sheet_file, index_col=self._INDEX_COLUMN).fillna("")
        self._data[self._LABEL_COLUMN] = self._data[self._LABEL_COLUMN].str.replace(
            "START", "BEGIN"
        )

        self._id = sheet_file.stem
        self._inventory = Inventory.load_or_download(
            int(self._id[-4:]), "", self._inventory_dir
        )

    def inventories(self) -> Iterable[Inventory]:
        yield self._inventory

    def inventory_numbers(self) -> Iterable[tuple[int, str]]:
        return [(self._inventory.inv_nr, self._inventory.inventory_part)]

    def annotate_inventory(self, inventory: Inventory) -> "Inventory":
        in_doc: bool = False
        """Inside a document or not."""
        pre_annotation: bool = True
        """Not seen any annotation yet"""

        for idx, row in self._data.iterrows():
            scan_nr = int(idx[-4:])

            has_text: bool = (
                len(inventory.get_scan(scan_nr).text()) >= MIN_REGION_TEXT_LENGTH
            )

            if annotation := row[self._LABEL_COLUMN].strip():
                # sheet provides annotation for page (BEGIN or END)
                label: Label = Label[annotation]
                pre_annotation = False

                message = f"Unexpected label: '{label.name}' for scan '{scan_nr}' for inventory '{inventory}' in sheet '{self}."

                if label == Label.BEGIN:
                    if in_doc:
                        label = Label.END_BEGIN
                    in_doc = True
                elif label == Label.END:
                    if not in_doc:
                        logging.warning(message)
                    in_doc = False
                else:
                    raise ValueError(message)
            elif pre_annotation:
                # In the beginning of the inventory
                if has_text:
                    if in_doc:
                        label = Label.IN
                    else:
                        label = Label.BEGIN
                        in_doc = True
                else:
                    if in_doc:
                        label = Label.END
                        in_doc = False
                    else:
                        label = Label.OUT
            else:
                # No annotation for scan, but not in the beginning of the inventory
                if in_doc:
                    label = Label.IN
                else:
                    label = Label.OUT

            inventory.annotate_scan(scan_nr, label)

        for scan in inventory.pages:
            unlabelled: list[Page] = [
                page for page in inventory.pages if page.label == Label.UNK
            ]
            if unlabelled:
                logging.error(
                    f"Removing un-annotated pages in inventory {inventory}: {unlabelled}"
                )
                for page in unlabelled:
                    inventory.remove_scan(page.scan_nr)

        return inventory
