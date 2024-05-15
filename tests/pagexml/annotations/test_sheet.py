from itertools import islice

import pytest

from document_segmentation import settings
from document_segmentation.pagexml.annotations.generale_missiven import GeneraleMissiven
from document_segmentation.pagexml.annotations.renate_analysis import RenateAnalysis
from document_segmentation.pagexml.datamodel.inventory import Inventory
from document_segmentation.pagexml.datamodel.label import Label

from ...conftest import DATA_DIR, GENERALE_MISSIVEN_CSV


class TestSheet:
    INV_COLUMN = "Inv.nr. Nationaal Archief (1.04.02)"

    @pytest.mark.parametrize(
        "sheet, length, min_inv, max_inv",
        [(RenateAnalysis(), 162, 1055, 10426), (GeneraleMissiven(), 914, 1068, 7957)],
    )
    def test_init(self, sheet, length, min_inv, max_inv):
        """Read the sheet and check the length and inventory numbers."""

        assert len(sheet) == length
        assert sheet._data[self.INV_COLUMN].min(skipna=False) == min_inv
        assert sheet._data[self.INV_COLUMN].max(skipna=False) == max_inv

        assert not sheet._data[self.INV_COLUMN].hasnans


class TestGeneraleMissiven:
    # FIXME: Mock servers requests

    @pytest.fixture()
    def test_sheet(self, tmp_path):
        return GeneraleMissiven(GENERALE_MISSIVEN_CSV, inventory_dir=tmp_path)

    def test_annotate_inventory(self, test_sheet):
        expected_labels = {1: Label.OUT, 919: Label.BOUNDARY, 920: Label.OUT}

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
