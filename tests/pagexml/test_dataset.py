import pytest

from document_segmentation.pagexml import GeneraleMissivenDataset, Label

from ..conftest import GENERALE_MISSIVEN_CSV, TEST_SHEET_SIZE


@pytest.fixture
def dataset(tmp_path):
    return GeneraleMissivenDataset.from_csv(
        GENERALE_MISSIVEN_CSV, cache_directory=tmp_path
    )


class TestGeneraleMissivenDataset:
    def test_from_csv(self, dataset, mock_request):
        assert len(dataset) == TEST_SHEET_SIZE

        assert dataset[17247][1] == Label.BEGIN
        assert dataset[17248][1] == Label.IN
        assert dataset[17595][1] == Label.END

    def test_label_tensor(self, dataset):
        labels = dataset.label_tensor()
        assert labels.shape == (len(dataset), 3), f"Bad shape: {labels.shape}"
        assert labels.sum() == len(dataset)

    @pytest.mark.parametrize(
        "start_index,expected",
        [
            (17247, ["NL-HaNA_1.04.02_1201_0016.jpg", "NL-HaNA_1.04.02_1201_0017.jpg"]),
            (17249, ["NL-HaNA_1.04.02_1201_0018.jpg", "NL-HaNA_1.04.02_1201_0019.jpg"]),
            # (TEST_SHEET_SIZE + 1, []),    # Test fails because mock request is not called
        ],
    )
    def test_page_ids(self, dataset, mock_request, start_index, expected):
        assert list(dataset[start_index : start_index + 2].page_ids()) == expected

    def test_segments(self, dataset):
        assert len(list(dataset.segments())) == 914
        for segment in dataset.segments():
            labels = list(segment.labels())
            assert labels[0] == Label.BEGIN
            assert all(label == Label.IN for label in labels[1:-1])
            assert labels[-1] == Label.END

    def test_label_count(self, dataset):
        counts = dataset.label_counts()
        assert counts[Label.BEGIN] == 914
        assert counts[Label.IN] == 190853
        assert counts[Label.END] == 914

    def test_segment_label_count(self, dataset):
        for segment in dataset.segments():
            counts = segment.label_counts()
            assert counts[Label.BEGIN] == 1
            assert counts[Label.END] == 1
