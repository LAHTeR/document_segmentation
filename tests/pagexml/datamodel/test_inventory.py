import pytest

from document_segmentation.pagexml.datamodel.inventory import Inventory
from document_segmentation.pagexml.datamodel.label import Label
from document_segmentation.pagexml.datamodel.page import Page


class TestInventory:
    def test_annotate_scan(self, tmp_path):
        inventory = Inventory(
            inv_nr=1201,
            inventory_part="A",
            pages=[
                Page(
                    label=Label.UNK,
                    scan_nr=1,
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

        with pytest.raises(ValueError):
            inventory.annotate_scan(0, Label.END)
            inventory.annotate_scan(4, Label.END)

    def test_download(self, mock_request, tmp_path):
        inventory = Inventory.download(
            inv_nr=1201, inventory_part="", target_directory=tmp_path
        )
        assert inventory.inv_nr == 1201
        assert inventory.inventory_part == ""
        assert len(inventory.pages) == 806

        assert (
            Inventory.load_or_download(
                inv_nr=1201, inventory_part="", inventory_dir=tmp_path
            )
            == inventory
        )

    def test_localfile(self, tmp_path):
        assert Inventory.local_file(1201, "A", tmp_path) == tmp_path / "1201_A.json"
        assert Inventory.local_file(1201, "", tmp_path) == tmp_path / "1201.json"
