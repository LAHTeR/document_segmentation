import pytest

from document_segmentation.model.region_embedding import RegionEmbedding
from document_segmentation.pagexml.datamodel import Region, RegionType
from document_segmentation.settings import REGION_EMBEDDING_OUTPUT_SIZE


@pytest.fixture
def region_embedding(scope="Session"):
    return RegionEmbedding()


class TestRegionEmbedding:
    @pytest.mark.parametrize(
        "regions",
        [
            ([]),
            (
                [
                    Region(
                        id="test_id",
                        types=[RegionType.HEADER],
                        lines=["test_line"],
                        coordinates=[],
                    )
                ]
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
                ]
            ),
        ],
    )
    def test_forward(self, region_embedding, regions):
        assert region_embedding(regions).size() == (
            len(regions),
            REGION_EMBEDDING_OUTPUT_SIZE,
        )
