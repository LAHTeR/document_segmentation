import random

import pytest
from pagexml.model.physical_document_model import PageXMLScan

from document_segmentation.pagexml.inventory import InventoryReader


class TestInventory:
    def test_pagexml(self, inventory, tmp_path):
        inventory._pagexml_directory = tmp_path / inventory._inv_nr

        inventory_size: int = 806

        assert all(
            isinstance(inventory.pagexml(page_nr), PageXMLScan)
            for page_nr in random.sample(range(1, inventory_size + 1), 10)
        )

        with pytest.raises(ValueError):
            inventory.pagexml(inventory_size + 1)

    def test_del(self, tmp_path, inventory):
        inventory.pagexml(1)

        assert inventory.local_zip_file.exists()
        assert inventory._pagexml_directory.exists()

        inventory.__del__()

        assert not (tmp_path / "1201.zip").exists()
        assert not (tmp_path / "1201").exists()

    def test_context_manager(self, mock_request, tmp_path):
        with InventoryReader("1201", cache_directory=tmp_path) as inventory:
            assert inventory.local_zip_file.is_file()
            assert inventory._pagexml_directory.is_dir()

        assert not inventory.local_zip_file.exists()
        assert not inventory._pagexml_directory.exists()
