from pathlib import Path

import pytest

from document_segmentation import settings
from document_segmentation.pagexml.annotations.renate_analysis import (
    RenateAnalysis,
    RenateAnalysisInv,
)
from document_segmentation.pagexml.datamodel.inventory import Inventory
from document_segmentation.pagexml.datamodel.label import SequenceLabel, Tanap

from ...conftest import DATA_DIR


class TestRenateAnalysis:
    @pytest.fixture(scope="session")
    def test_sheet(self) -> Path:
        return RenateAnalysis()

    def test_annotate_inventory(self, test_sheet):
        expected_labels = {
            1: SequenceLabel.OUT,
            114: SequenceLabel.BOUNDARY,
            115: SequenceLabel.OUT,
        }

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

        expected_labels = [SequenceLabel.OUT, SequenceLabel.BOUNDARY, SequenceLabel.OUT]
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
            [SequenceLabel.OUT] * 14
            + [SequenceLabel.BOUNDARY]
            + [SequenceLabel.IN] * 3
            + [SequenceLabel.BOUNDARY]
            + [SequenceLabel.OUT] * 7
            + [SequenceLabel.BOUNDARY]
            + [SequenceLabel.IN] * 68
            + [SequenceLabel.BOUNDARY]
            + [SequenceLabel.OUT] * 2
            + [SequenceLabel.BOUNDARY]
            + [SequenceLabel.IN] * 6
            + [SequenceLabel.BOUNDARY]
            + [SequenceLabel.BOUNDARY]
            + [SequenceLabel.BOUNDARY]
            + [SequenceLabel.IN]
            + [SequenceLabel.BOUNDARY]
            + [SequenceLabel.IN]
            + [SequenceLabel.BOUNDARY]
            + [SequenceLabel.OUT] * 6
            + [SequenceLabel.BOUNDARY]
            + [SequenceLabel.IN] * 115
            + [SequenceLabel.BOUNDARY]
            + [SequenceLabel.OUT]
            + [SequenceLabel.BOUNDARY]
            + [SequenceLabel.IN] * 2
            + [SequenceLabel.BOUNDARY]
            + [SequenceLabel.IN] * 3
            + [SequenceLabel.BOUNDARY]
            + [SequenceLabel.IN]
            + [SequenceLabel.BOUNDARY]
            + [SequenceLabel.IN] * 3
            + [SequenceLabel.BOUNDARY]
            + [SequenceLabel.IN] * 5
            + [SequenceLabel.BOUNDARY]
            + [SequenceLabel.IN] * 3
            + [SequenceLabel.BOUNDARY]
            + [SequenceLabel.IN]
            + [SequenceLabel.BOUNDARY] * 2
            + [SequenceLabel.IN]
            + [SequenceLabel.BOUNDARY]
            + [SequenceLabel.BOUNDARY]
            + [SequenceLabel.IN] * 5
            + [SequenceLabel.BOUNDARY]
            + [SequenceLabel.OUT] * 2
        )

        sheet = RenateAnalysisInv(sheet_file, inventory_dir=tmp_path)
        annotated = sheet.annotate_inventory()

        assert annotated.inv_nr == 1547
        assert len(annotated) == 688

        for page, label in zip(annotated.pages, expected_labels):
            assert page.label == label
