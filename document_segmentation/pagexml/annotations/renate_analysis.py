import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from ...settings import INVENTORY_DIR, RENATE_TANAP_CATEGORISATION_SHEET
from ..datamodel.inventory import Inventory
from ..datamodel.label import Combined, FrontMatter, Label, Tanap
from ..datamodel.page import Page
from .sheet import Sheet


class RenateAnalysis(Sheet):
    _RUBRIEK_PATTERN = re.compile(r"^RUBRIEK:(.*);ARCHIEFSTUK")

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
            sheet_file,
            dtype=self._dtypes,
            index_col=self._INDEX_COLUMN,
        ).dropna(subset=self._dropna | {"ID in TANAP database"})

        self._data[self._DEEL_VAN_INVENTARIS_COL] = (
            self._data[self._DEEL_VAN_INVENTARIS_COL].replace({"0": pd.NA}).fillna("")
        )

        self._categories = pd.read_excel(
            sheet_file,
            sheet_name="Categoriecodes",
            index_col="Code",
            dtype={"Code": str},
        )
        if not self._categories.index.is_unique:
            raise ValueError("Duplicate category codes in 'Categoriecodes' sheet.")

        self._tanap = pd.read_excel(sheet_file, sheet_name="TANAP", index_col="ID")

    def annotated_rows(self, inventory: Inventory) -> pd.DataFrame:
        # TODO: use child documents where available
        return super().annotated_rows(inventory)

    def combined_label(self, label: Label, row: int = None) -> Combined:
        if label == Label.OUT:
            return label.combined(FrontMatter.EMPTY)
        else:
            tanap_id: int = int(row["ID in TANAP database"])
            category: Tanap = self.tanap_category(tanap_id)
            return label.combined(category)

    def tanap_id(self, inv_nr, *, page_nr: int = None) -> int:
        """Look up the TANAP ID for a given inventory number.

        Args:
            inv_nr (int): The inventory number.
        Returns:
            int: The TANAP ID.
        Raises:
            RuntimeError: If the TANAP ID cannot be found.
        """
        tanap_id = self._tanap.loc[self._tanap["INVENTARISNUMMER"] == inv_nr]
        if len(tanap_id) == 1:
            return tanap_id.index
        elif len(tanap_id) > 1:
            if page_nr is not None:
                raise ValueError(
                    f"Multiple TANAP IDs found for inventory number {inv_nr}"
                )
        else:
            raise ValueError(f"No TANAP ID for inventory number {inv_nr}")

    def tanap_category(self, tanap_id: int) -> Tanap:
        """Look up the TANAP category for a given TANAP ID.

        Args:
            tanap_id (int): The TANAP ID.
        Returns:
            Tanap: The category.
        Raises:
            RuntimeError: If the category code cannot be found.
        """
        rubric: str
        tanap_category: Tanap
        row = self._tanap.iloc[tanap_id]

        if match := self._RUBRIEK_PATTERN.match(row["TYPE"]):
            rubric = match.group(1).strip()
        else:
            raise RuntimeError(
                f"Failed to extract rubric from {row['TYPE']} for TANAP ID {tanap_id}"
            )

        if category := self._categories.loc[self._categories == rubric]:
            try:
                tanap_category: Tanap = Tanap(int(category["Code"]))
            except ValueError as e:
                raise RuntimeError(
                    f"Invalid category code '{category['Code']}' for TANAP ID {tanap_id}"
                ) from e
        else:
            raise RuntimeError(
                f"Failed to find category for rubric '{rubric}' for TANAP ID {tanap_id}"
            )
        return tanap_category


class RenateAnalysisInv(Sheet):
    _INDEX_COLUMN = "Scan File_Name"
    _PAGE_COLUMN = "Page"
    _LABEL_COLUMN = "TANAP Boundaries"
    _SUB_DOC_COLUMN = "Subdocument boundaries"
    _TYPE_COLUMN = "Type of non-document page"

    # TODO: how to look up TANAP categories for TANAP documents that are not in the RenateAnalysis?

    def __init__(
        self,
        sheet_file: Path,
        *,
        ra_sheet: RenateAnalysis,
        inventory_dir: Path = INVENTORY_DIR,
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

        self._ra_sheet = ra_sheet

    def combined_label(
        self, label: Label, *, front_matter: FrontMatter = FrontMatter.EMPTY
    ) -> Combined:
        if label is Label.OUT:
            return label.combined(front_matter)
        else:
            tanap_id = self._ra_sheet.tanap_id(self._inventory.inv_nr)
            category: Tanap = self._ra_sheet.tanap_category(tanap_id)
            return label.combined(label, category)

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
                front_matter = FrontMatter.EMPTY

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
            elif in_doc:
                # No annotation for scan
                label = Label.IN
            else:
                if _type := row[self._TYPE_COLUMN].strip():
                    front_matter = FrontMatter[_type.upper().replace(" ", "_")]

                label = Label.OUT

            combined: Combined = self.combined_label(label, front_matter=front_matter)
            inventory.annotate_scan(scan_nr, combined)

        # Remove unlabelled pages
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
