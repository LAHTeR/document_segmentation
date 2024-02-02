from document_segmentation.pagexml.datamodel import Document, Label, Page


class TestDocument:
    def test_enum_from_int(self):
        json = """
        {
            "id": "1",
            "inventory_nr": 1068,
            "inventory_part": null,
            "pages": [
                { "label": 1, "regions": [], "scan_nr": 617 }
            ]
        }
        """

        expected_document = Document(
            id="1",
            inventory_nr=1068,
            inventory_part=None,
            pages=[Page(label=Label.BEGIN, regions=[], scan_nr=617)],
        )
        document = Document.model_validate_json(json)

        assert document == expected_document
        assert all(isinstance(page.label, Label) for page in document.pages)
