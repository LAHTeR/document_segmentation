import logging
from pathlib import Path
from typing import Union

import pandas as pd
from tqdm import tqdm

from document_segmentation.pagexml.inventory import Inventory

from ..settings import TEST_SHEET


class TestSet:
    INDEX_COLUMN = "Document_ID"
    INV_NR_COLUMN = "Inv.nr. Nationaal Archief (1.04.02)"
    BEGIN_PAGE_COLUMN = "Begin scan"
    END_PAGE_COLUMN = "End scan"

    def __init__(self, *, input_file: Union[str, Path] = TEST_SHEET) -> None:
        self._data: pd.DataFrame = pd.read_excel(
            input_file,
            index_col=TestSet.INDEX_COLUMN,
            dtype={
                "ID in TANAP database": pd.Int64Dtype(),
                "Inv.nr. Nationaal Archief (1.04.02)": str,
                "Part of the inv.nr.": pd.Int64Dtype(),
                "Begin scan": pd.Int64Dtype(),
                "End scan": pd.Int64Dtype(),
                "Code TANAP document category": str,
            },
        ).dropna(subset=[TestSet.INV_NR_COLUMN, TestSet.BEGIN_PAGE_COLUMN])

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    def recall(self):
        recalled: int = 0
        total: int = 0

        for index, row in tqdm(self.data.iterrows(), total=len(self.data), unit="row"):
            inv_nr, begin_page, end_page = row[
                [
                    TestSet.INV_NR_COLUMN,
                    TestSet.BEGIN_PAGE_COLUMN,
                    TestSet.END_PAGE_COLUMN,
                ]
            ]
            if pd.isna(inv_nr) or pd.isna(begin_page):
                logging.warning(
                    f"Skipping row {index} because both '{TestSet.INV_NR_COLUMN}' ({row[TestSet.INV_NR_COLUMN]}) and '{TestSet.BEGIN_PAGE_COLUMN}' ({TestSet.BEGIN_PAGE_COLUMN}) must have values."
                )
            else:
                total += 1
                inventory = Inventory(str(inv_nr))
                document_begin_page = inventory.pagexml(begin_page)

                document_types = document_begin_page.document_type()
                if document_types:
                    print(f"{index}: {document_types}")
                recalled += bool(document_types)
        print(f"Found {recalled}/{total} document types ({recalled/total}).")
