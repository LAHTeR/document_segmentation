from pathlib import Path

import pandas as pd

from ..settings import GENERALE_MISSIVEN_SHEET
from .inventory import Inventory


class GeneraleMissiven:
    """Helper class to handle the Generale Missiven spreadsheet."""

    _INDEX_COLUMN = "ID"
    _INV_NR_COLUMN = "Inv.nr. Nationaal Archief (1.04.02)"
    _START_PAGE_COLUMN = "Beginscan"
    _LAST_PAGE_COLUMN = "Eindscan"
    _SEPARATOR = ";"

    def __init__(self, sheet_file: Path = GENERALE_MISSIVEN_SHEET) -> None:
        """Load the spreadsheet into memory.

        Args:
            sheet_file (Path, optional): Path to the spreadsheet. Defaults to settings.GENERALE_MISSIVEN_SHEET.
        """
        self._data = pd.read_csv(
            sheet_file,
            sep=self._SEPARATOR,
            index_col=self._INDEX_COLUMN,
            dtype={
                self._INV_NR_COLUMN: pd.Int16Dtype(),
                self._START_PAGE_COLUMN: pd.Int16Dtype(),
                self._LAST_PAGE_COLUMN: pd.Int16Dtype(),
            },
        )

    def _pagexml(self, id, page):
        inv_nr = self._data.loc[id, self._INV_NR_COLUMN]

        return Inventory(inv_nr).pagexml(page)

    def pagexml_first_page(self, id):
        """Get the PageXML for the first page of a row in the spreadsheet.

        Args:
            id: The ID of the row in the spreadsheet.
        """
        return self._pagexml(id, self._data.loc[id, self._START_PAGE_COLUMN])

    def pagexml_last_page(self, id):
        """Get the PageXML for the last page of a row in the spreadsheet.

        Args:
            id: The ID of the row in the spreadsheet.
        """
        return self._pagexml(id, self._data.loc[id, self._LAST_PAGE_COLUMN])
