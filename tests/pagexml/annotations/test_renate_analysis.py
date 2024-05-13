from pathlib import Path

import pytest

from document_segmentation import settings
from document_segmentation.pagexml.annotations.renate_analysis import (
    RenateAnalysis,
    RenateAnalysisInv,
)
from document_segmentation.pagexml.datamodel.inventory import Inventory
from document_segmentation.pagexml.datamodel.label import Label, Tanap

from ...conftest import DATA_DIR


class TestRenateAnalysis:
    @pytest.fixture(scope="session")
    def test_sheet(self) -> Path:
        return RenateAnalysis()

    def test_annotate_inventory(self, test_sheet):
        expected_labels = {1: Label.OUT, 114: Label.BOUNDARY, 115: Label.OUT}

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

        expected_labels = [Label.OUT, Label.BOUNDARY, Label.OUT]
        expected_scan_nrs = [1, 114, 115]

        preprocessed: Inventory = test_sheet.preprocess(
            inventory, min_region_text_length, max_size
        )

        assert preprocessed.labels() == expected_labels
        assert [page.scan_nr for page in preprocessed.pages] == expected_scan_nrs

    def test_documents(self, test_sheet):
        expected_labels = 3 * [Tanap.DAGREGISTERS]
        expected_lengths = [60, 42, 68]

        for document, label, length in zip(
            test_sheet.documents(), expected_labels, expected_lengths
        ):
            assert document.label == label
            assert len(document) == length


class TestRenateAnalysisInv:
    @pytest.fixture()
    def sheet_file(self) -> Path:
        return settings.ANNOTATIONS_DIR / "Analysis Renate 1547.csv"

    @pytest.mark.parametrize("mock_request", [1547], indirect=True)
    def test_init(self, sheet_file, tmp_path, mock_request):
        """Test the initialization of the RenateAnalysisInv class."""
        sheet = RenateAnalysisInv(sheet_file, inventory_dir=tmp_path)

        assert len(sheet) == 690
        assert sheet._id == "Analysis Renate 1547"
        assert sheet.inventory_numbers() == [(1547, "")]

    @pytest.mark.parametrize("mock_request", [1547], indirect=True)
    def test_annotate_inventory(self, sheet_file, tmp_path, mock_request):
        """Test the annotation of an inventory with the RenateAnalysisInv class."""
        expected_labels = (
            [Label.OUT] * 14
            + [Label.BOUNDARY]
            + [Label.IN] * 3
            + [Label.BOUNDARY]
            + [Label.OUT] * 7
            + [Label.BOUNDARY]
            + [Label.IN] * 68
            + [Label.BOUNDARY]
            + [Label.OUT] * 2
            + [Label.BOUNDARY]
            + [Label.IN] * 6
            + [Label.BOUNDARY]
            + [Label.BOUNDARY]
            + [Label.BOUNDARY]
            + [Label.IN]
            + [Label.BOUNDARY]
            + [Label.IN]
            + [Label.BOUNDARY]
            + [Label.OUT] * 6
            + [Label.BOUNDARY]
            + [Label.IN] * 115
            + [Label.BOUNDARY]
            + [Label.OUT]
            + [Label.BOUNDARY]
            + [Label.IN] * 2
            + [Label.BOUNDARY]
            + [Label.IN] * 3
            + [Label.BOUNDARY]
            + [Label.IN]
            + [Label.BOUNDARY]
            + [Label.IN] * 3
            + [Label.BOUNDARY]
            + [Label.IN] * 5
            + [Label.BOUNDARY]
            + [Label.IN] * 3
            + [Label.BOUNDARY]
            + [Label.IN]
            + [Label.BOUNDARY] * 2
            + [Label.IN]
            + [Label.BOUNDARY]
            + [Label.BOUNDARY]
            + [Label.IN] * 5
            + [Label.BOUNDARY]
            + [Label.OUT] * 2
        )

        sheet = RenateAnalysisInv(sheet_file, inventory_dir=tmp_path)
        annotated = sheet.annotate_inventory()

        assert annotated.inv_nr == 1547
        assert len(annotated) == 688

        for page, label in zip(annotated.pages, expected_labels):
            assert page.label == label
