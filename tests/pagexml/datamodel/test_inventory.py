import logging
from contextlib import nullcontext as does_not_raise
from uuid import UUID

import pytest
import requests_mock

from document_segmentation.pagexml.datamodel.inventory import (
    Inventory,
    ThumbnailDownloader,
)
from document_segmentation.pagexml.datamodel.label import Label
from document_segmentation.pagexml.datamodel.page import Page
from document_segmentation.pagexml.datamodel.region import Region

from ...conftest import DATA_DIR, TEST_THUMBNAIL_FILE


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

        expected_labels = [Label.UNK, Label.BOUNDARY, Label.UNK]
        inventory.annotate_scan(2, Label.BOUNDARY)

        for page, expected_label in zip(inventory.pages, expected_labels):
            assert page.label == expected_label

        # END and BEGIN on the same page:
        inventory.annotate_scan(2, Label.BOUNDARY)
        assert inventory.pages[1].label == Label.BOUNDARY

        # Invalid scan numbers:
        with pytest.raises(ValueError):
            inventory.annotate_scan(0, Label.BOUNDARY)
            inventory.annotate_scan(4, Label.BOUNDARY)

    # TODO: test for files with inventory parts (both valid ("A" and invalid ("1"))
    @pytest.mark.parametrize("mock_request", [1201], indirect=True)
    def test_download(self, mock_request, tmp_path):
        inventory = Inventory.download(
            inv_nr=1201, inventory_part="", target_directory=tmp_path
        )

        # Assert that this does not trigger another remote request
        assert (
            Inventory.load_or_download(
                inv_nr=1201, inventory_part="", inventory_dir=tmp_path
            )
            == inventory
        )

    @pytest.mark.parametrize(
        "inventory, expected",
        [
            (
                Inventory(inv_nr=1201, inventory_part="", pages=[]),
                Inventory(inv_nr=1201, inventory_part="", pages=[]),
            ),
            (
                Inventory(
                    inv_nr=1201,
                    inventory_part="",
                    pages=[
                        Page(
                            label=Label.UNK,
                            scan_nr=1,
                            external_ref="test_ref",
                            regions=[
                                Region(id="test_id", types=[], coordinates=[], lines=[])
                            ],
                        )
                    ],
                ),
                Inventory(
                    inv_nr=1201,
                    inventory_part="",
                    pages=[
                        Page(
                            label=Label.OUT,
                            scan_nr=1,
                            external_ref="test_ref",
                            regions=[],
                        )
                    ],
                ),
            ),
        ],
    )
    def test_empty_unlabelled(self, inventory, expected):
        assert inventory.empty_unlabelled().pages == expected.pages
        assert inventory.empty_unlabelled() == expected

    @pytest.mark.parametrize(
        "inv_nr, inv_part, expected",
        [(1201, "", "1201"), (1201, "A", "1201A"), (1201, "1", "1201"), (1, "", "1")],
    )
    def test_full_inv_nr(self, inv_nr, inv_part, expected):
        assert (
            Inventory(inv_nr=inv_nr, inventory_part=inv_part, pages=[]).full_inv_nr()
            == expected
        )

    @pytest.mark.parametrize(
        "inventory_nr, expected_length",
        [(1105, 1092), (2542, 2052)],
    )
    def test_get_scan(self, inventory_nr, expected_length):
        inventory = Inventory.load(inventory_nr, "", DATA_DIR)

        for scan_nr in range(1, expected_length + 1):
            assert inventory.get_scan(scan_nr).scan_nr == scan_nr
            assert inventory.get_scan(scan_nr) == inventory.pages[scan_nr - 1]

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
            (
                1,
                "NL-HaNA_1.04.02_0001_0026.jpg",
                "https://www.nationaalarchief.nl/onderzoeken/archief/1.04.02/invnr/0001/file/NL-HaNA_1.04.02_0001_0026",
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
        "inventory_nr, expected_length, expected_exception",
        [
            (1105, 1092, does_not_raise()),
            (2542, 2052, does_not_raise()),
            (1, 1092, pytest.raises(FileNotFoundError)),
        ],
    )
    def test_load(self, inventory_nr, expected_length, expected_exception):
        with expected_exception:
            inventory = Inventory.load(inventory_nr, "", DATA_DIR)

            assert inventory.inv_nr == inventory_nr
            assert inventory.inventory_part == ""
            assert len(inventory) == expected_length

    @pytest.mark.parametrize(
        "inv_nr, inv_part, expected, expected_logs",
        [
            (1201, "", "1201.json", []),
            (1201, "A", "1201_A.json", []),
            (1201, "1", "1201.json", ["Removing invalid inventory part: '1'"]),
            (1, "A", "0001_A.json", []),
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

    @pytest.mark.parametrize(
        "pages, max_length, expected",
        [
            ([], 10, []),
            (
                [Page(label=Label.OUT, scan_nr=1, external_ref="test_ref", regions=[])]
                * 2,
                1,
                [Page(label=Label.OUT, scan_nr=1, external_ref="test_ref", regions=[])],
            ),
            (
                [Page(label=Label.IN, scan_nr=1, external_ref="test_ref", regions=[])]
                * 2,
                1,
                [Page(label=Label.IN, scan_nr=1, external_ref="test_ref", regions=[])]
                * 2,
            ),
            (
                [Page(label=Label.OUT, scan_nr=1, external_ref="test_ref", regions=[])]
                * 2,
                2,
                [Page(label=Label.OUT, scan_nr=1, external_ref="test_ref", regions=[])]
                * 2,
            ),
            (
                [Page(label=Label.OUT, scan_nr=1, external_ref="test_ref", regions=[])]
                * 2
                + [Page(label=Label.IN, scan_nr=2, external_ref="test_ref", regions=[])]
                + [
                    Page(
                        label=Label.OUT, scan_nr=3, external_ref="test_ref", regions=[]
                    )
                ]
                * 2,
                1,
                [
                    Page(
                        label=Label.OUT, scan_nr=1, external_ref="test_ref", regions=[]
                    ),
                    Page(
                        label=Label.IN, scan_nr=2, external_ref="test_ref", regions=[]
                    ),
                    Page(
                        label=Label.OUT, scan_nr=3, external_ref="test_ref", regions=[]
                    ),
                ],
            ),
        ],
    )
    def test_remove_empty_pages(self, pages, max_length, expected):
        assert (
            Inventory(inv_nr=1201, inventory_part="", pages=pages)
            .remove_empty_pages(max_length=max_length)
            .pages
            == expected
        )

    @pytest.mark.parametrize(
        "inventories, expected",
        [
            ([], [0.0] * 4),
            (
                [
                    Inventory(
                        inv_nr=1,
                        inventory_part="",
                        pages=[
                            Page(
                                label=Label.OUT,
                                scan_nr=1,
                                external_ref="test_ref",
                                regions=[],
                            )
                        ],
                    )
                ],
                [0.0, 1.0, 1.0, 0.5],
            ),
            (
                [
                    Inventory(
                        inv_nr=1,
                        inventory_part="",
                        pages=[
                            Page(
                                label=Label.OUT,
                                scan_nr=1,
                                external_ref="test_ref",
                                regions=[],
                            )
                        ],
                    ),
                    Inventory(
                        inv_nr=1,
                        inventory_part="",
                        pages=[
                            Page(
                                label=Label.OUT,
                                scan_nr=1,
                                external_ref="test_ref",
                                regions=[],
                            ),
                            Page(
                                label=Label.BOUNDARY,
                                scan_nr=1,
                                external_ref="test_ref",
                                regions=[],
                            ),
                        ],
                    ),
                ],
                [0.0, 1.5, 3.0, 1.0],
            ),
        ],
    )
    def test_total_class_weights(self, inventories, expected):
        assert Inventory.total_class_weights(inventories) == expected


class TestThumbnailDownloader:
    @pytest.fixture
    def thumbnail_downloader(self, tmp_path):
        return ThumbnailDownloader(
            mapping={"1": UUID("db776fa8-9d77-45ca-8e85-e1c48406a555")},
            thumbnails_dir=tmp_path,
        )

    def test_thumbnail_url(self, thumbnail_downloader):
        page = Page(
            label=Label.UNK,
            scan_nr=1,
            external_ref="aa84f770-f5d7-40ac-bfda-db3d06f204c9",
            regions=[],
        )
        inventory = Inventory(inv_nr=1, inventory_part="", pages=[page])
        assert (
            thumbnail_downloader.thumbnail_url(inventory, page)
            == "https://service.archief.nl/iipsrv?IIIF=/db/77/6f/a8/9d/77/45/ca/8e/85/e1/c4/84/06/a5/55/aa84f770-f5d7-40ac-bfda-db3d06f204c9.jp2/full/,200/0/default.jpg"
        )

    def test_download(self, thumbnail_downloader):
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
