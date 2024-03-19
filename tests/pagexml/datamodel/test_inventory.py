import pytest

from document_segmentation.pagexml.datamodel.inventory import Inventory
from document_segmentation.settings import INVENTORY_DIR


class TestInventory:
    @pytest.mark.skip(reason="Testing actual download, should be mocked.")
    # TODO: mock the download
    # TODO: test with inventory part
    def test_download(self):
        inv_nr = 1547

        inventory = Inventory.download(inv_nr)

        with (INVENTORY_DIR / str(inv_nr)).with_suffix(".json").open("rt") as f:
            assert Inventory.model_validate_json(f.read()) == inventory
