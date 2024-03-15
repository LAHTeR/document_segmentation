from zipfile import ZipFile

from pagexml.parser import parse_pagexml_file

from document_segmentation.pagexml.datamodel.label import Label
from document_segmentation.pagexml.datamodel.page import Page

from ...conftest import TEST_FILE


class TestPage:
    def test_enum_from_int(self):
        json = '{ "label": 0, "regions": [], "scan_nr": 617 }'

        expected_page = Page(label=Label.BEGIN, regions=[], scan_nr=617)
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
