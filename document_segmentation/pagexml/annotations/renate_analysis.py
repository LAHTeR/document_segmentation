from pathlib import Path

import pandas as pd

from ...settings import RENATE_TANAP_CATEGORISATION_SHEET
from .sheet import Sheet


class RenateAnalysis(Sheet):
    def __init__(self, sheet: Path = RENATE_TANAP_CATEGORISATION_SHEET) -> None:
        """Helper class to handle the 'Renate Analysis' sheet.

        Args:
            sheet (Path, optional): Paths to the spreadsheet.
                Defaults to settings.RENATE_TANAP_CATEGORISATION_SHEET.
        """
        super().__init__()

        self._data = pd.read_excel(sheet, dtype=self._dtypes).dropna(
            subset=self._dropna
        )
