from contextlib import nullcontext as does_not_raise

import pytest
from pagexml.model.physical_document_model import PageXMLScan

from document_segmentation.pagexml.inventory import Inventory

from ..conftest import DATA_DIR


class TestInventory:
    def test_local_file(self, inventory):
        assert inventory.local_file == DATA_DIR / "1201.zip"

    def test_download(self, tmp_path, mock_request):
        test_file = DATA_DIR / "1201.zip"
        inventory = Inventory("1201", cache_directory=tmp_path)

        assert inventory.download() == tmp_path / "1201.zip"
        assert (
            tmp_path / "1201.zip"
        ).read_bytes() == test_file.read_bytes(), "Downloaded file does not match."

    @pytest.mark.parametrize(
        "page_nr,expected_error",
        [(511, does_not_raise()), (10000, pytest.raises(ValueError))],
    )
    def test_pagexml(self, inventory, page_nr, expected_error):
        with expected_error:
            assert isinstance(inventory.pagexml(page_nr), PageXMLScan)
