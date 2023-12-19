from contextlib import nullcontext as does_not_raise

import pytest
import requests_mock

from document_segmentation.pagexml.inventory import Inventory
from document_segmentation.pagexml.pagexml import PageXML

from ..conftest import DATA_DIR


class TestInventory:
    def test_local_file(self, inventory):
        assert inventory.local_file == DATA_DIR / "1201.zip"

    def test_download(self, tmp_path):
        test_file = DATA_DIR / "1201.zip"
        inventory = Inventory("1201", cache_directory=tmp_path)
        url = "https://hucdrive.huc.knaw.nl/HTR/obp-v2-pagexml-leon-metadata-trimmed-2023-11/1201.zip"

        with requests_mock.Mocker() as m:
            m.get(url, content=test_file.open("rb").read())

            assert inventory.download() == tmp_path / "1201.zip"

            assert m.called, "Request was not made."
            assert m.call_count == 1, "Request was made more than once."
            assert (
                m.request_history[0].method == "GET"
            ), "Request was not a GET request."
            assert m.request_history[0].url == url, "Request URL does not match."

        assert (
            tmp_path / "1201.zip"
        ).read_bytes() == test_file.read_bytes(), "Downloaded file does not match."

    @pytest.mark.parametrize(
        "page_nr,expected_error",
        [(511, does_not_raise()), (10000, pytest.raises(ValueError))],
    )
    def test_pagexml(self, inventory, page_nr, expected_error):
        with expected_error:
            assert isinstance(inventory.pagexml(page_nr), PageXML)
