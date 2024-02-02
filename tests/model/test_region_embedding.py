import pytest

from document_segmentation.model.region_embedding import RegionEmbedding
from document_segmentation.pagexml.datamodel import Region, RegionType


class TestRegionEmbedding:
    @pytest.mark.parametrize(
        "regions,expected_size",
        [
            ([], (0, 777)),
            (
                [
                    Region(
                        id="test_id",
                        types=[RegionType.HEADER],
                        lines=["test_line"],
                        coordinates=[],
                    )
                ],
                (1, 777),
            ),
        ],
    )
    def test_forward(self, regions, expected_size):
        assert RegionEmbedding()(regions).size() == expected_size
