class TestPageXML:
    def test_get_regions_from_top(self, inventory):
        assert [
            region.id for region in inventory.pagexml(511).get_regions_from_top()
        ] == [
            "region_61f2a686-bc6d-409a-aec4-b92fa830119a_14",
            "region_2d536fc2-188e-4b01-97aa-6b6187945abe_11",
            "region_5a0fe245-92bb-411e-8804-a99e99a6479e",
            "region_8f535eac-d06f-4842-865d-ea9df03c14ac_7",
            "b1d13588-c64a-441e-b482-cc45b41b1654",
            "region_7ab4cd3f-9e5a-4732-bc45-3c9266bada33_5",
            "region_56da241a-d10c-48e2-a0d7-3286d3f32a31",
            "region_1ce769da-f81f-4d2b-8aa1-439d7e4783cc",
            "region_c9b813fa-3420-43b7-94e3-1955be2292e4_4",
            "df4036ac-b072-407c-9494-e242aaf9e2e7",
            "region_84bf87ff-da52-403d-83bf-e25e89957aef_18",
            "region_19707202-b3f0-4d5a-aa3f-27e95bc3428b_12",
            "e16896bd-13f5-461e-b3ad-a690b1b41109",
        ]

    def test_top_paragraph(self, inventory):
        assert (
            inventory.pagexml(511).top_paragraph().id
            == "region_2d536fc2-188e-4b01-97aa-6b6187945abe_11"
        )

    def test_document_type(self, inventory):
        # TODO: define tests that do find a document type
        assert list(inventory.pagexml(511).document_type()) == []
