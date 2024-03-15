from document_segmentation.pagexml.datamodel.document import Document
from document_segmentation.pagexml.datamodel.label import Label
from document_segmentation.pagexml.datamodel.page import Page


class TestDocument:
    def test_enum_from_int(self):
        json = """
        {
            "id": "1",
            "inventory_nr": 1068,
            "inventory_part": null,
            "pages": [
                { "label": 0, "regions": [], "scan_nr": 617, "external_ref": "test_ref" }
            ]
        }
        """

        expected_document = Document(
            id="1",
            inventory_nr=1068,
            inventory_part=None,
            pages=[
                Page(
                    label=Label.BEGIN, regions=[], scan_nr=617, external_ref="test_ref"
                )
            ],
        )
        document = Document.model_validate_json(json)

        assert document == expected_document
        assert all(isinstance(page.label, Label) for page in document.pages)
