from pathlib import Path

import pandas as pd

from ...settings import GENERALE_MISSIVEN_SHEET, INVENTORY_DIR
from .sheet import Sheet


class GeneraleMissiven(Sheet):
    """Helper class to handle the Generale Missiven spreadsheet."""

    _INDEX_COLUMN = "ID"
    _START_PAGE_COLUMN = "Beginscan"
    _LAST_PAGE_COLUMN = "Eindscan"
    _DEEL_VAN_INVENTARIS_COL = "Deel v. inventarisnummer"
    _STATUS_COLUMN = "Problemen gevonden tijdens handmatige check:"
    _SKIP_MESSAGE = "Niet gedigitaliseerd."

    _SEPARATOR = ";"

    def __init__(
        self,
        sheet_file: Path = GENERALE_MISSIVEN_SHEET,
        inventory_dir: Path = INVENTORY_DIR,
    ) -> None:
        """Load the spreadsheet into memory.

        Args:
            sheet_file (Path, optional): Path to the spreadsheet. Defaults to settings.GENERALE_MISSIVEN_SHEET.
            inventory_dir (Path, optional): The directory where the inventories are stored.
        """
        super().__init__(inventory_dir=inventory_dir)

        self._data = pd.read_csv(
            sheet_file,
            sep=self._SEPARATOR,
            index_col=self._INDEX_COLUMN,
            dtype=self._dtypes,
        ).dropna(subset=self._dropna)

        self._data[self._DEEL_VAN_INVENTARIS_COL] = self._data[
            self._DEEL_VAN_INVENTARIS_COL
        ].fillna("")
