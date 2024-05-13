import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from tqdm import tqdm

from ...settings import INVENTORY_DIR, RENATE_TANAP_CATEGORISATION_SHEET
from ..datamodel.document import Document
from ..datamodel.inventory import Inventory
from ..datamodel.label import Label, Tanap
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

        # self._categories = pd.read_excel(
        #     sheet_file, sheet_name="Categoriecodes", index_col="Code"
        # )
        # self._tanap = pd.read_excel(sheet_file, sheet_name="TANAP", index_col="ID")

    def annotated_rows(self, inventory: Inventory) -> pd.DataFrame:
        # TODO: use child documents where available
        return super().annotated_rows(inventory)

    def documents(self, *, n: Optional[int] = None) -> Iterable[Document]:
        categories = self._data.loc[
            pd.notna(self._data["Code TANAP document category"])
        ].head(n)

        for _, row in tqdm(
            categories.iterrows(),
            desc="Reading docs",
            total=len(categories),
            unit="row",
        ):
            inventory = Inventory.load_or_download(
                row[self._INV_NR_COLUMN], row[self._DEEL_VAN_INVENTARIS_COL]
            )
            yield Document(
                pages=inventory.get_scans(
                    row[self._START_PAGE_COLUMN], row[self._LAST_PAGE_COLUMN] + 1
                ),
                label=Tanap(int(row["Code TANAP document category"])),
            )


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
        """Inside a document?"""

        for idx, row in self._data.sort_index().iterrows():
            scan_nr = int(idx[-4:])

            # Sub-document annotation overrides main document annotation
            annotation = (
                row[self._SUB_DOC_COLUMN].strip() or row[self._LABEL_COLUMN].strip()
            )

            if annotation and not annotation.startswith("SAME AS"):
                # sheet provides annotation for page (BEGIN or END)
                if "START" in annotation or "END" in annotation:
                    label = Label.BOUNDARY
                    # is last element of the annotation "START" or "END"?
                    in_doc = annotation.split("/")[-1] == "START"
            elif in_doc:
                label = Label.IN
            else:
                label = Label.OUT

            inventory.annotate_scan(scan_nr, label)

        unlabelled: list[Page] = [
            page for page in inventory.pages if page.label == Label.UNK
        ]
        for page in unlabelled:
            logging.warning(
                f"Removing un-annotated page in inventory {inventory}: {page}"
            )
            inventory.remove_scan(page.scan_nr)
        return inventory
