from pathlib import Path

import pytest

from document_segmentation import settings
from document_segmentation.pagexml.annotations.renate_analysis import (
    RenateAnalysis,
    RenateAnalysisInv,
)
from document_segmentation.pagexml.datamodel.inventory import Inventory
from document_segmentation.pagexml.datamodel.label import Label

from ...conftest import DATA_DIR


class TestRenateAnalysis:
    @pytest.fixture(scope="session")
    def test_sheet(self) -> Path:
        return RenateAnalysis()

    def test_annotate_inventory(self, test_sheet):
        expected_labels = {1: Label.OUT, 114: Label.END_BEGIN, 115: Label.OUT}

        for page in test_sheet.annotate_inventory(
            Inventory.load(2542, "", DATA_DIR)
        ).pages:
            assert page.label == expected_labels[page.scan_nr]

    def test_preprocess(self, test_sheet):
        """Test the preprocessing of the RenateAnalysis sheet."""
        min_region_text_length = 20
        max_size = 1024
        inventory = Inventory.load(2542, "", DATA_DIR)
        assert len(inventory) == 2052

        expected_labels = [Label.OUT, Label.END_BEGIN, Label.OUT]
        expected_scan_nrs = [1, 114, 115]

        preprocessed: Inventory = test_sheet.preprocess(
            inventory, min_region_text_length, max_size
        )

        assert preprocessed.labels() == expected_labels
        assert [page.scan_nr for page in preprocessed.pages] == expected_scan_nrs


@pytest.mark.skipif(
    not (settings.SERVER_USERNAME and settings.SERVER_PASSWORD),
    reason="No server credentials.",
)
class TestRenateAnalysisInv:
    # FIXME: Mock servers requests

    @pytest.fixture()
    def sheet_file(self) -> Path:
        return settings.ANNOTATIONS_DIR / "Analysis Renate 1547.csv"

    def test_init(self, sheet_file, tmp_path):
        """Test the initialization of the RenateAnalysisInv class."""
        sheet = RenateAnalysisInv(sheet_file, inventory_dir=tmp_path)

        for label in sheet._data[RenateAnalysisInv._LABEL_COLUMN]:
            try:
                Label[label]
            except KeyError:
                assert label == ""

        assert len(sheet) == 690
        assert sheet._id == "Analysis Renate 1547"
        assert sheet.inventory_numbers() == [(1547, "")]

    def test_annotate_inventory(self, sheet_file, tmp_path):
        """Test the annotation of an inventory with the RenateAnalysisInv class."""
        expected_labels = (
            [Label.OUT] * 14
            + [Label.BEGIN]
            + [Label.IN] * 3
            + [Label.END]
            + [Label.OUT] * 7
            + [Label.BEGIN]
            + [Label.IN] * 68
            + [Label.END]
            + [Label.OUT] * 2
            + [Label.BEGIN]
            + [Label.IN] * 6
            + [Label.END]
            + [Label.BEGIN]
            + [Label.END_BEGIN]
            + [Label.IN]
            + [Label.END_BEGIN]
            + [Label.IN]
            + [Label.END]
            + [Label.OUT] * 6
            + [Label.BEGIN]
            + [Label.IN] * 115
            + [Label.END]
            + [Label.OUT]
            + [Label.BEGIN]
            + [Label.IN] * 2
            + [Label.END_BEGIN]
            + [Label.IN] * 3
            + [Label.END_BEGIN]
            + [Label.IN]
            + [Label.END_BEGIN]
            + [Label.IN] * 3
            + [Label.END_BEGIN]
            + [Label.IN] * 5
            + [Label.END_BEGIN]
            + [Label.IN] * 3
            + [Label.END_BEGIN]
            + [Label.IN]
            + [Label.END_BEGIN] * 2
            + [Label.IN]
            + [Label.END]
            + [Label.BEGIN]
            + [Label.IN] * 5
            + [Label.END]
            + [Label.OUT] * 2
        )

        sheet = RenateAnalysisInv(sheet_file, inventory_dir=tmp_path)
        annotated = sheet.annotate_inventory()

        assert annotated.inv_nr == 1547
        assert len(annotated) == 688

        for page, label in zip(annotated.pages, expected_labels):
            assert page.label == label
