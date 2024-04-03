import logging

import pytest

from document_segmentation.pagexml.datamodel.inventory import Inventory
from document_segmentation.pagexml.datamodel.label import Label
from document_segmentation.pagexml.datamodel.page import Page


class TestInventory:
    @pytest.mark.parametrize(
        "inv_nr, inventory_part, pages, expected_inv_nr, expected_inventory_part, expected_pages",
        [
            (1201, "A", [], 1201, "A", []),
            (1201, "", [], 1201, "", []),
            (1201, "1", [], 1201, "", []),
        ],
    )
    def test_init(
        self,
        inv_nr,
        inventory_part,
        pages,
        expected_inv_nr,
        expected_inventory_part,
        expected_pages,
    ):
        inventory = Inventory(inv_nr=inv_nr, inventory_part=inventory_part, pages=pages)
        assert inventory.inv_nr == expected_inv_nr
        assert inventory.inventory_part == expected_inventory_part
        assert inventory.pages == expected_pages

    def test_annotate_scan(self, tmp_path):
        inventory = Inventory(
            inv_nr=1201,
            inventory_part="A",
            pages=[
                Page(
                    label=Label.UNK,
                    scan_nr=1,  # Scans start at 1
                    doc_id="doc1",
                    external_ref="ref1",
                    regions=[],
                ),
                Page(
                    label=Label.UNK,
                    scan_nr=2,
                    doc_id="doc1",
                    external_ref="ref1",
                    regions=[],
                ),
                Page(
                    label=Label.UNK,
                    scan_nr=3,
                    doc_id="doc1",
                    external_ref="ref1",
                    regions=[],
                ),
            ],
        )

        expected_labels = [Label.UNK, Label.BEGIN, Label.UNK]
        inventory.annotate_scan(2, Label.BEGIN)

        for page, expected_label in zip(inventory.pages, expected_labels):
            assert page.label == expected_label

        # END and BEGIN on the same page:
        inventory.annotate_scan(2, Label.END)
        assert inventory.pages[1].label == Label.END_BEGIN

        # Invalid scan numbers:
        with pytest.raises(ValueError):
            inventory.annotate_scan(0, Label.END)
            inventory.annotate_scan(4, Label.END)

    def test_download(self, mock_request, tmp_path):
        inventory = Inventory.download(
            inv_nr=1201, inventory_part="", target_directory=tmp_path
        )

        # Assert that this does not trigger another remote (mock) request:
        assert (
            Inventory.load_or_download(
                inv_nr=1201, inventory_part="", inventory_dir=tmp_path
            )
            == inventory
        )

    @pytest.mark.parametrize(
        "inv_nr, inv_part, expected, expected_logs",
        [
            (1201, "", "1201.json", []),
            (1201, "A", "1201_A.json", []),
            (1201, "1", "1201.json", ["Removing invalid inventory part: '1'"]),
        ],
    )
    def test_localfile(
        self, tmp_path, caplog, inv_nr, inv_part, expected, expected_logs
    ):
        with caplog.at_level(logging.WARNING):
            assert (
                Inventory.local_file(inv_nr, inv_part, tmp_path) == tmp_path / expected
            )
            assert caplog.messages == expected_logs, "Unexpected log messages"
