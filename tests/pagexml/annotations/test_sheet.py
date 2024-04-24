from itertools import islice
from pathlib import Path

import pytest

from document_segmentation import settings
from document_segmentation.pagexml.annotations.generale_missiven import GeneraleMissiven
from document_segmentation.pagexml.annotations.renate_analysis import (
    RenateAnalysis,
    RenateAnalysisInv,
)
from document_segmentation.pagexml.datamodel.inventory import Inventory
from document_segmentation.pagexml.datamodel.label import Label

from ...conftest import DATA_DIR, GENERALE_MISSIVEN_CSV


class TestSheet:
    INV_COLUMN = "Inv.nr. Nationaal Archief (1.04.02)"

    @pytest.mark.parametrize(
        "sheet, length, min_inv, max_inv",
        [(RenateAnalysis(), 132, 1060, 8935), (GeneraleMissiven(), 914, 1068, 7957)],
    )
    def test_init(self, sheet, length, min_inv, max_inv):
        """Read the sheet and check the length and inventory numbers."""

        assert len(sheet) == length
        assert sheet._data[self.INV_COLUMN].min(skipna=False) == min_inv
        assert sheet._data[self.INV_COLUMN].max(skipna=False) == max_inv

        assert not sheet._data[self.INV_COLUMN].hasnans


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
    def test_sheet(self) -> Path:
        return settings.DATA_DIR / "Analysis Renate 1547.csv"

    def test_init(self, test_sheet, tmp_path):
        """Test the initialization of the RenateAnalysisInv class."""
        sheet = RenateAnalysisInv(test_sheet, inventory_dir=tmp_path)

        for label in sheet._data[RenateAnalysisInv._LABEL_COLUMN]:
            try:
                Label[label]
            except KeyError:
                assert label == ""

        assert len(sheet) == 690
        assert sheet._id == "Analysis Renate 1547"
        assert sheet.inventory_numbers() == [(1547, "")]


class TestGeneraleMissiven:
    # FIXME: Mock servers requests

    @pytest.fixture()
    def test_sheet(self, tmp_path):
        return GeneraleMissiven(GENERALE_MISSIVEN_CSV, inventory_dir=tmp_path)

    def test_annotate_inventory(self, test_sheet):
        expected_labels = {1: Label.OUT, 919: Label.END_BEGIN, 920: Label.OUT}

        for page in test_sheet.annotate_inventory(
            Inventory.load(1105, "", DATA_DIR)
        ).pages:
            assert page.label == expected_labels[page.scan_nr]

    @pytest.mark.skipif(
        not (settings.SERVER_USERNAME and settings.SERVER_PASSWORD),
        reason="No server credentials.",
    )
    def test_inventory_numbers(self, test_sheet):
        n = 5

        assert list(islice(test_sheet.inventory_numbers(), n)) == [
            (1068, ""),
            (1070, ""),
            (1071, ""),
            (1072, ""),
            (1073, ""),
        ]

    @pytest.mark.skipif(
        not (settings.SERVER_USERNAME and settings.SERVER_PASSWORD),
        reason="No server credentials.",
    )
    def test_inventories(self, test_sheet):
        expected_inv_nrs = [1068, 1070, 1071, 1072, 1073]
        expected_inv_parts = [""] * 5
        expected_lengths = [1024, 1322, 688, 942, 702]

        for inventory, inv_nr, inv_part, length in zip(
            test_sheet.inventories(),
            expected_inv_nrs,
            expected_inv_parts,
            expected_lengths,
        ):
            assert inventory.inv_nr == inv_nr
            assert inventory.inventory_part == inv_part
            assert len(inventory) == length

    @pytest.mark.skipif(
        not (settings.SERVER_USERNAME and settings.SERVER_PASSWORD),
        reason="No server credentials.",
    )
    def test_all_annotated_inventories(self, test_sheet):
        n = 2

        expected_inv_nrs = [1068, 1070]
        expected_inv_parts = [""] * 2
        expected_lengths = [90, 96]

        for inventory, inv_nr, inv_part, length in zip(
            test_sheet.all_annotated_inventories(n=n),
            expected_inv_nrs,
            expected_inv_parts,
            expected_lengths,
        ):
            assert all(page.label != Label.UNK for page in inventory.pages)
            assert inventory.inv_nr == inv_nr
            assert inventory.inventory_part == inv_part
            assert (
                len(inventory) == length
            ), f"Inventory {inventory} has {len(inventory)} pages, expected {length}."
