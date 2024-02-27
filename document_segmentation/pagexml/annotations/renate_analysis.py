from pathlib import Path

import pandas as pd

from ...settings import RENATE_TANAP_CATEGORISATION_SHEET
from .sheet import Sheet


class RenateAnalysis(Sheet):
    def __init__(self, sheet: list[Path] = RENATE_TANAP_CATEGORISATION_SHEET) -> None:
        """Load the spreadsheet into memory.

        Args:
            files (list[Path], optional): Paths to the spreadsheet. Defaults to settings.RENATE_ANALYSIS_SHEETS.
        """
        super().__init__()

        self._data = pd.read_excel(sheet, dtype=self._dtypes).dropna(
            subset=self._dropna
        )
