import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from ...settings import INVENTORY_DIR, RENATE_TANAP_CATEGORISATION_SHEET
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

    def annotated_rows(self, inventory: Inventory) -> pd.DataFrame:
        # TODO: use child documents where available
        return super().annotated_rows(inventory)


class RenateAnalysisInv(Sheet):
    _INDEX_COLUMN = "Scan File_Name"
    _PAGE_COLUMN = "Page"
    _LABEL_COLUMN = "TANAP Boundaries"
    _SUB_DOC_COLUMN = "Subdocument boundaries"

    def __init__(
        self, sheet_file: Path, *, inventory_dir: Path = INVENTORY_DIR
    ) -> None:
        super().__init__(inventory_dir=inventory_dir)

        self._data = pd.read_csv(
            sheet_file,
            delimiter=";",
            index_col=self._INDEX_COLUMN,
        ).fillna("")

        self._id = sheet_file.stem
        self._inventory = Inventory.load_or_download(
            int(self._id[-4:]), "", self._inventory_dir
        )

    def inventories(self) -> Iterable[Inventory]:
        yield self._inventory

    def inventory_numbers(self) -> Iterable[tuple[int, str]]:
        return [(self._inventory.inv_nr, self._inventory.inventory_part)]

    def annotate_inventory(self, inventory: Inventory = None) -> "Inventory":
        """Assume all pages in a document have been annotated."""

        inventory = inventory or self._inventory
        if inventory != self._inventory:
            raise ValueError(
                f"Inventory {inventory} does not match the expected inventory {self._inventory}."
            )

        in_doc: bool = False
        """Inside a document or not."""

        for idx, row in self._data.sort_index().iterrows():
            scan_nr = int(idx[-4:])

            # Sub-document annotation overrides main document annotation
            annotation = (
                row[self._SUB_DOC_COLUMN].strip() or row[self._LABEL_COLUMN].strip()
            )

            if annotation and not annotation.startswith("SAME AS"):
                # sheet provides annotation for page (BEGIN or END)
                label: Label = Label[annotation.replace("/", "_")]

                error_message = f"Unexpected label: '{annotation}' ('{label.name}') for scan '{scan_nr}' for inventory '{inventory}' in sheet '{self}."

                if label == Label.BEGIN:
                    if in_doc:
                        label = Label.END_BEGIN
                    in_doc = True
                elif label == Label.END:
                    if not in_doc:
                        logging.error(error_message)
                    in_doc = False
                elif label == Label.END_BEGIN:
                    if annotation.startswith("END") and not in_doc:
                        raise ValueError(error_message)
                    in_doc = annotation[-5:] in {"START", "BEGIN"}
                else:
                    raise ValueError(error_message)
            else:
                # No annotation for scan
                label = Label.IN if in_doc else Label.OUT

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
