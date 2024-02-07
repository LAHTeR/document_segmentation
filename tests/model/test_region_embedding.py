import pytest

from document_segmentation.model.region_embedding import RegionEmbedding
from document_segmentation.pagexml.datamodel import Region, RegionType


@pytest.fixture
def region_embedding(scope="Session"):
    return RegionEmbedding()


class TestRegionEmbedding:
    @pytest.mark.parametrize(
        "regions,expected_size",
        [
            ([], (0, 784)),
            (
                [
                    Region(
                        id="test_id",
                        types=[RegionType.HEADER],
                        lines=["test_line"],
                        coordinates=[],
                    )
                ],
                (1, 784),
            ),
            (
                [
                    Region(
                        id="test_id_1",
                        types=[RegionType.HEADER],
                        lines=["test_line_1"],
                        coordinates=[],
                    ),
                    Region(
                        id="test_id_2",
                        types=[RegionType.HEADER],
                        lines=["test_line_2"],
                        coordinates=[],
                    ),
                ],
                (2, 784),
            ),
        ],
    )
    def test_forward(self, region_embedding, regions, expected_size):
        assert region_embedding(regions).size() == expected_size
