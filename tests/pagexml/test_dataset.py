import pytest
import requests_mock

from document_segmentation.pagexml import GeneraleMissivenDataset, Label

from ..conftest import DATA_DIR, GENERALE_MISSIVEN_CSV


@pytest.fixture
def dataset(tmp_path):
    return GeneraleMissivenDataset.from_csv(
        GENERALE_MISSIVEN_CSV, cache_directory=tmp_path
    )


class TestGeneraleMissivenDataset:
    TEST_SHEET_SIZE = 192681
    TEST_INV_NR = "1201"

    def test_from_csv(self, dataset):
        assert len(dataset) == self.TEST_SHEET_SIZE

        url = f"https://hucdrive.huc.knaw.nl/HTR/obp-v2-pagexml-leon-metadata-trimmed-2023-11/{self.TEST_INV_NR}.zip"
        with requests_mock.Mocker() as m:
            test_file = (DATA_DIR / self.TEST_INV_NR).with_suffix(".zip")
            m.get(url, content=test_file.open("rb").read())

            assert dataset[17247][1] == Label.BEGIN
            assert dataset[17248][1] == Label.IN
            assert dataset[17595][1] == Label.END

            assert m.called, "Request was not made."
            assert m.request_history[0].url == url, "Request URL does not match."

    def test_label_tensor(self, dataset):
        labels = dataset.label_tensor()
        assert labels.shape == (len(dataset), 3), f"Bad shape: {labels.shape}"
        assert labels.sum() == len(dataset)

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
