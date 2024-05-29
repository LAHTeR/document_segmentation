import logging
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from tqdm import tqdm

from ...settings import INVENTORY_DIR, RENATE_TANAP_CATEGORISATION_SHEET
from ..datamodel.document import Document
from ..datamodel.inventory import Inventory
from ..datamodel.label import SequenceLabel, Tanap
from ..datamodel.page import Page
from .sheet import Sheet


class RenateAnalysis(Sheet):
    _RUBRIEK_REGEX = re.compile(r"RUBRIEK:([^;]+);")

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

        self._categories = pd.read_excel(sheet_file, sheet_name="Categoriecodes")
        self._tanap = pd.read_excel(sheet_file, sheet_name="TANAP", index_col="ID")

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
                inv_nr=inventory.inv_nr,
                inventory_part=inventory.inventory_part,
                label=Tanap(int(row["Code TANAP document category"])),
                pages=inventory.get_scans(
                    row[self._START_PAGE_COLUMN], row[self._LAST_PAGE_COLUMN] + 1
                ),
            )

    def documents_from_sheet(
        self, sheet: "RenateAnalysisInv", *, n: int = None
    ) -> Iterable[Document]:
        """Return the documents with annotations from the Renate Analysis sheet.

        Args:
            sheet (RenateAnalysis): The Renate Analysis sheet to loop up TANAP catagories.

        Returns:
            Iterable[Document]: The documents with annotations.
        """
        count = 0
        for tanap_id, rows in sheet._data.groupby(by="TANAP ID", dropna=False):
            if pd.notna(tanap_id):
                # document rows
                pages = sheet._inventory.get_scans(
                    int(rows.index.min()[-4:]), int(rows.index.max()[-4:]) + 1
                )
                try:
                    category = self._tanap_category(int(tanap_id))
                except ValueError as e:
                    logging.error(
                        f"Could not get TANAP category for TANAP ID {tanap_id}: {e}"
                    )
                    continue
                yield Document(
                    inv_nr=sheet._inventory.inv_nr,
                    inventory_part=sheet._inventory.inventory_part,
                    label=category,
                    pages=pages,
                )

            else:
                # Non-document rows
                for idx, row in rows.iterrows():
                    non_doc_type = row["Type of non-document page"]
                    if pd.notna(non_doc_type) and non_doc_type != "Empty":
                        if non_doc_type not in RenateAnalysisInv.NON_DOC_TYPES:
                            raise ValueError(
                                f"Unknown non-document type '{non_doc_type}' in '{self}' for '{row}'"
                            )

                        _, page = sheet._inventory.get_scan(int(idx[-4:]))
                        yield Document(
                            inv_nr=sheet._inventory.inv_nr,
                            inventory_part=sheet._inventory.inventory_part,
                            label=Tanap.FRONT_MATTER,
                            pages=[page],
                        )
            count += 1
            if n and count >= n:
                break

    def _tanap_category(self, tanap_doc_id: int) -> Tanap:
        row = self._tanap.loc[tanap_doc_id]
        _type = row["TYPE"]

        if pd.isna(_type):
            raise ValueError(f"Could not find type for TANAP ID {tanap_doc_id}.")

        if match := re.match(self._RUBRIEK_REGEX, row["TYPE"].strip()):
            rubriek: str = match.group(1)
            category = self._categories.loc[
                self._categories["Titel"].str.strip() == rubriek
            ]
        else:
            raise ValueError(f"Could not find rubriek in {row['RUBRIEK']}")

        if not len(category) == 1:
            raise ValueError(f"Could not find category for rubriek {rubriek}.")
        return Tanap(int(category.iloc[0]["Code"]))


class RenateAnalysisInv(Sheet):
    _INDEX_COLUMN = "Scan File_Name"
    _PAGE_COLUMN = "Page"
    _LABEL_COLUMN = "TANAP Boundaries"
    _SUB_DOC_COLUMN = "Subdocument boundaries"

    NON_DOC_TYPES: set[str] = {
        "Cover",
        "Empty",
        "Table of contents",
        "Folio/page number",
        "Section title page",
        "Document title page",
        "Small note",
    }

    def __init__(
        self, sheet_file: Path, *, inventory_dir: Path = INVENTORY_DIR
    ) -> None:
        super().__init__(inventory_dir=inventory_dir)

        self._data = pd.read_csv(
            sheet_file,
            delimiter=";",
            index_col=self._INDEX_COLUMN,
            thousands=".",
            dtype={self._LABEL_COLUMN: str, self._SUB_DOC_COLUMN: str},
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
                    label = SequenceLabel.BOUNDARY
                    # is last element of the annotation "START" or "END"?
                    in_doc = annotation.split("/")[-1] == "START"
            elif in_doc:
                label = SequenceLabel.IN
            else:
                label = SequenceLabel.OUT

            inventory.annotate_scan(scan_nr, label)

        unlabelled: list[Page] = [
            page for page in inventory.pages if page.label == SequenceLabel.UNK
        ]
        for page in unlabelled:
            logging.warning(
                f"Removing un-annotated page in inventory {inventory}: {page}"
            )
            inventory.remove_scan(page.scan_nr)
        return inventory
