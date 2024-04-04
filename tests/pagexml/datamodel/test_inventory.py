import logging
from uuid import UUID

import pytest
import requests_mock

from document_segmentation.pagexml.datamodel.inventory import (
    Inventory,
    ThumbnailDownloader,
)
from document_segmentation.pagexml.datamodel.label import Label
from document_segmentation.pagexml.datamodel.page import Page

from ...conftest import TEST_THUMBNAIL_FILE


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

    # TODO: test for files with inventory parts (both valid ("A" and invalid ("1"))
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
        "inventory_nr, doc_id, expected",
        [
            (
                1557,
                "NL-HaNA_1.04.02_1557_0026.jpg",
                "https://www.nationaalarchief.nl/onderzoeken/archief/1.04.02/invnr/1557/file/NL-HaNA_1.04.02_1557_0026",
            ),
            (
                1557,
                None,
                "https://www.nationaalarchief.nl/onderzoeken/archief/1.04.02/invnr/1557/file/NL-HaNA_1.04.02_1557_0026",
            ),
        ],
    )
    def test_link(self, inventory_nr, doc_id, expected):
        page = Page(
            label=Label.UNK,
            scan_nr=26,
            doc_id=doc_id,
            external_ref="test_ref",
            regions=[],
        )

        inventory = Inventory(inv_nr=inventory_nr, inventory_part="", pages=[page])
        assert inventory.link(page) == expected

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


class TestThumbnailDownloader:
    def test_download(self, tmp_path):
        thumbnail_downloader = ThumbnailDownloader(
            mapping={"1": UUID("db776fa8-9d77-45ca-8e85-e1c48406a555")},
            thumbnails_dir=tmp_path,
        )
        page = Page(
            label=Label.UNK,
            scan_nr=1,
            external_ref="aa84f770-f5d7-40ac-bfda-db3d06f204c9",
            regions=[],
        )
        inventory = Inventory(inv_nr=1, inventory_part="", pages=[page])

        expected_url = "https://service.archief.nl/iipsrv?IIIF=/db/77/6f/a8/9d/77/45/ca/8e/85/e1/c4/84/06/a5/55/aa84f770-f5d7-40ac-bfda-db3d06f204c9.jp2/full/100,/0/default.jpg"

        with requests_mock.Mocker() as mocker:
            mocker.get(expected_url, content=TEST_THUMBNAIL_FILE.open("rb").read())

            target_file = thumbnail_downloader.download(inventory, page, size="100,")
        assert target_file.is_file()
