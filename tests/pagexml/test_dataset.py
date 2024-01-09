from contextlib import nullcontext as does_not_raise

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

    def test_inverse_frequency(self, dataset):
        assert dataset.inverse_frequencies() == pytest.approx(
            [210.81072210065645, 1.0095780522181994, 210.81072210065645]
        )


class TestLabel:
    @pytest.mark.parametrize(
        "scores,expected,expected_exception",
        [
            ([0.9, 0.1, 0], {"BEGIN": 0.9, "IN": 0.1, "END": 0}, does_not_raise()),
            ([0.1] * 5, None, pytest.raises(ValueError)),
            ([0.1] * 2, None, pytest.raises(ValueError)),
        ],
    )
    def test_map_scores(self, scores, expected, expected_exception):
        with expected_exception:
            assert Label.map_scores(scores) == expected
