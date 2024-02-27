import logging
from pathlib import Path

import pandas as pd

from ...settings import GENERALE_MISSIVEN_SHEET
from ..inventory import InventoryReader
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

    def __init__(self, sheet_file: Path = GENERALE_MISSIVEN_SHEET) -> None:
        """Load the spreadsheet into memory.

        Args:
            sheet_file (Path, optional): Path to the spreadsheet. Defaults to settings.GENERALE_MISSIVEN_SHEET.
        """
        super().__init__()

        self._data = pd.read_csv(
            sheet_file,
            sep=self._SEPARATOR,
            index_col=self._INDEX_COLUMN,
            dtype=self._dtypes,
        ).dropna(subset=self._dropna)

    def _pagexml(self, id, page):
        logging.warning(
            "DEPRECATED: This method instantiates an InventoryReader for every page."
        )

        inv_nr = self._data.loc[id, self._INV_NR_COLUMN]

        return InventoryReader(inv_nr).pagexml(page)

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
