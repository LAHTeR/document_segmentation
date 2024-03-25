import pytest

from document_segmentation import settings
from document_segmentation.pagexml.annotations.generale_missiven import GeneraleMissiven
from document_segmentation.pagexml.annotations.renate_analysis import (
    RenateAnalysis,
    RenateAnalysisInv,
)
from document_segmentation.pagexml.datamodel.label import Label


class TestSheet:
    INV_COLUMN = "Inv.nr. Nationaal Archief (1.04.02)"

    @pytest.mark.parametrize(
        "sheet_class, length, min_inv, max_inv",
        [(RenateAnalysis, 78, 1060, 8935), (GeneraleMissiven, 914, 1068, 7957)],
    )
    def test_init(self, sheet_class, length, min_inv, max_inv):
        """Read the sheet and check the length and inventory numbers."""

        sheet = sheet_class()

        assert len(sheet) == length
        assert sheet._data[self.INV_COLUMN].min(skipna=False) == min_inv
        assert sheet._data[self.INV_COLUMN].max(skipna=False) == max_inv

        assert not sheet._data[self.INV_COLUMN].hasnans


class TestRenateAnalysisInv:
    @pytest.fixture()
    def test_sheet(self):
        return settings.DATA_DIR / "Analysis Renate 1547.xlsx"

    def test_init(self, test_sheet):
        """Test the initialization of the RenateAnalysisInv class."""
        sheet = RenateAnalysisInv(test_sheet)

        for label in sheet._data[RenateAnalysisInv._LABEL_COLUMN]:
            try:
                Label[label]
            except KeyError:
                assert label == ""

        assert len(sheet) == 690
        assert sheet._id == "Analysis Renate 1547"
        assert sheet._inv_nr == 1547
