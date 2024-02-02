import pytest

from document_segmentation.pagexml.datamodel import Region, RegionType


class TestRegion:
    @pytest.mark.parametrize(
        "region",
        [
            Region(id="test_id", types=[], coordinates=[], lines=[]),
            Region(
                id="test_id",
                types=[RegionType.TEXT_REGION],
                coordinates=[(1, 2), (3, 4)],
                lines=["line1", "line2"],
            ),
        ],
    )
    def test_hash(self, region):
        assert hash(region)


class TestRegionType:
    @pytest.mark.parametrize(
        "region_type,expected",
        [
            (RegionType.CATCH_WORD, 0),
            (RegionType.HEADER, 1),
            (RegionType.MARGINALIA, 2),
            (RegionType.PAGE_NUMBER, 3),
            (RegionType.PAGEXML_DOC, 4),
            (RegionType.PARAGRAPH, 5),
            (RegionType.PHYSICAL_STRUCTURE_DOC, 6),
            (RegionType.SIGNATURE_MARK, 7),
            (RegionType.TEXT_REGION, 8),
        ],
    )
    def test_index(self, region_type, expected):
        assert region_type.index() == expected
