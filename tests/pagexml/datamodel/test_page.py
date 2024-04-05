import logging
from zipfile import ZipFile

import pytest
from pagexml.parser import parse_pagexml_file

from document_segmentation.pagexml.datamodel.label import Label
from document_segmentation.pagexml.datamodel.page import Page
from document_segmentation.pagexml.datamodel.region import Region

from ...conftest import TEST_FILE


class TestPage:
    def test_enum_from_int(self):
        json = (
            '{ "label": 1, "regions": [], "scan_nr": 617 , "external_ref": "test_ref"}'
        )

        expected_page = Page(
            label=Label.BEGIN, regions=[], scan_nr=617, external_ref="test_ref"
        )
        page = Page.model_validate_json(json)

        assert page == expected_page
        assert isinstance(
            page.label, Label
        ), f"label should be of type Label, but was {type(page.label)}"

    def test_from_pagexml(self, tmp_path):
        expected_page = {
            "label": Label.BEGIN,
            "scan_nr": 1,
            "doc_id": "NL-HaNA_1.04.02_1201_0001.jpg",
            "external_ref": "a37e52e8-c9af-4b0f-8390-df846024830a",
        }
        expected_region_ids = [
            "region_7d00fe73-664c-4a16-84e2-e57325fec161_2",
            "region_ec90613d-f499-4c1b-881a-64f704e53278_1",
            "region_7baba970-ed60-472e-9685-daf77faa77ba",
            "region_8dc74729-31bc-4ea8-a0e6-65220043ab90",
            "ce76645e-b134-48bf-b78a-95ab081cb360",
        ]

        filename = "pagexml/1201/NL-HaNA_1.04.02_1201_0001.xml"

        with ZipFile(TEST_FILE, "r") as zip_ref:
            pagexml = parse_pagexml_file(zip_ref.extract(filename, tmp_path))

            page = Page.from_pagexml(Label.BEGIN, 1, pagexml)

            for key, value in expected_page.items():
                assert getattr(page, key) == value
            assert [region.id for region in page.regions] == expected_region_ids

    @pytest.mark.parametrize(
        "scan_nr, inv_nr, expected",
        [
            (1, "0001", "NL-HaNA_1.04.02_0001_0001"),
            (26, "1557", "NL-HaNA_1.04.02_1557_0026"),
        ],
    )
    def test_guess_doc_id(self, caplog, scan_nr, inv_nr, expected):
        with caplog.at_level(logging.WARNING):
            page = Page(
                label=Label.UNK,
                scan_nr=scan_nr,
                doc_id=None,
                external_ref="test_ref",
                regions=[],
            )

            assert page.guess_doc_id(inv_nr) == expected
            assert caplog.messages == []

    @pytest.mark.parametrize(
        "page, expected, expected_warnings",
        [
            (
                Page(label=Label.UNK, scan_nr=1, external_ref="test_ref", regions=[]),
                Page(label=Label.OUT, scan_nr=1, external_ref="test_ref", regions=[]),
                [],
            ),
            (
                Page(
                    label=Label.UNK,
                    scan_nr=1,
                    external_ref="test_ref",
                    regions=[Region(id="test_id", types=[], coordinates=[], lines=[])],
                ),
                Page(label=Label.OUT, scan_nr=1, external_ref="test_ref", regions=[]),
                [],
            ),
            (
                Page(label=Label.IN, scan_nr=1, external_ref="test_ref", regions=[]),
                Page(label=Label.OUT, scan_nr=1, external_ref="test_ref", regions=[]),
                ["Emptying page with label 'IN'."],
            ),
        ],
    )
    def test_empty(self, caplog, page, expected, expected_warnings):
        with caplog.at_level(logging.WARNING):
            assert page.empty() == expected
        assert caplog.messages == expected_warnings
