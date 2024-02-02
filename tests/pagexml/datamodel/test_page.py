from document_segmentation.pagexml.datamodel import Label, Page


class TestPage:
    def test_enum_from_int(self):
        json = '{ "label": 1, "regions": [], "scan_nr": 617 }'

        expected_page = Page(label=Label.BEGIN, regions=[], scan_nr=617)
        page = Page.model_validate_json(json)

        assert page == expected_page
        assert isinstance(
            page.label, Label
        ), f"label should be of type Label, but was {type(page.label)}"
