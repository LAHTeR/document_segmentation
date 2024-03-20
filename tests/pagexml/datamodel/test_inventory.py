from document_segmentation.pagexml.datamodel.inventory import Inventory


class TestInventory:
    def test_download(self, mock_request, tmp_path):
        expected_pages = 806

        inventory = Inventory.download(inv_nr=1201, target_directory=tmp_path)

        assert len(inventory) == expected_pages

        with open(Inventory.local_file(1201, "", tmp_path), "rt") as f:
            _json = f.read()
            assert Inventory.model_validate_json(_json) == inventory
