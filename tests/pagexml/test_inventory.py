from contextlib import nullcontext as does_not_raise

import pytest

from document_segmentation.pagexml.pagexml import PageXML

from ..conftest import DATA_DIR


class TestInventory:
    def test_local_file(self, inventory):
        assert inventory.local_file == DATA_DIR / "1201.zip"

    def test_download(self, inventory):
        # TODO: test overwrite, download (mock request)
        assert inventory.download("username", "password") == DATA_DIR / "1201.zip"

    @pytest.mark.parametrize(
        "page_nr,expected_error",
        [(511, does_not_raise()), (10000, pytest.raises(ValueError))],
    )
    def test_pagexml(self, inventory, page_nr, expected_error):
        with expected_error:
            assert isinstance(inventory.pagexml(page_nr), PageXML)
