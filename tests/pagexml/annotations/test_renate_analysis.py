from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pytest

from document_segmentation import settings
from document_segmentation.pagexml.annotations.renate_analysis import (
    DocumentTypeSheet,
    RenateAnalysis,
    RenateAnalysisInv,
)
from document_segmentation.pagexml.datamodel.inventory import Inventory
from document_segmentation.pagexml.datamodel.label import (
    DocumentType,
    SequenceLabel,
    Tanap,
)

from ...conftest import DATA_DIR


class TestDocumentTypeSheet:
    @pytest.fixture(scope="session")
    def sheet(self):
        return DocumentTypeSheet.from_file(settings.DOCUMENT_TYPE_TANAP_MAPPING_FILE)

    @pytest.mark.parametrize(
        "tanap_category,expected_document_type,expected_exception",
        [
            ("1.1", DocumentType.DAGREGISTER, does_not_raise()),
            ("14.2", DocumentType.MISCELLANEOUS, does_not_raise()),
            ("INVALID", DocumentType.MISCELLANEOUS, pytest.raises(ValueError)),
        ],
    )
    def test_document_type(
        self, sheet, tanap_category, expected_document_type, expected_exception
    ):
        with expected_exception:
            sheet.document_type(tanap_category) == expected_document_type


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

    # FIXME: mock request
    @pytest.mark.skipif(
        not (settings.SERVER_USERNAME and settings.SERVER_PASSWORD),
        reason="No server credentials",
    )
    def test_documents(self, test_sheet):
        expected_labels = 3 * [DocumentType.DAGREGISTER]
        expected_lengths = [60, 42, 68]

        for document, label, length in zip(
            test_sheet.documents(), expected_labels, expected_lengths
        ):
            assert document.label == label
            assert len(document) == length

    def test_documents_from_sheet(self, test_sheet, tmp_path):
        sheet = RenateAnalysisInv(
            settings.ANNOTATIONS_DIR / "Analysis Renate 1547.csv",
            inventory_dir=tmp_path,
        )

        docs = list(test_sheet.documents_from_sheet(sheet))
        doc_scan_nr_1 = docs[16]
        assert (
            doc_scan_nr_1.inv_nr,
            doc_scan_nr_1.label,
            len(doc_scan_nr_1.pages),
        ) == (1547, DocumentType.FRONT_MATTER, 1)

    @pytest.mark.parametrize(
        "tanap_id,expected_category",
        [
            (1, Tanap.BRIEVEN_BINNEN),
            (5084.0, Tanap.BRIEVEN_NEDERLAND),
            (83347, Tanap.STUKKEN_SCHEPEN),
            (170132, Tanap.STUKKEN_BOEKHOUDING),
        ],
    )
    def test_tanap_category(self, test_sheet, tanap_id: int, expected_category: Tanap):
        assert test_sheet._tanap_category(tanap_id) == expected_category

    @pytest.mark.parametrize(
        "tanap_doc_id,expected,expected_exception",
        [
            (1, DocumentType.BRIEF, does_not_raise()),
            (12, DocumentType.REGISTER, pytest.raises(ValueError)),
            (170132, DocumentType.RENDEMENT, does_not_raise()),
        ],
    )
    def test_document_type(
        self, test_sheet, tanap_doc_id: int, expected: DocumentType, expected_exception
    ):
        with expected_exception:
            assert test_sheet._document_type(tanap_doc_id) == expected


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
