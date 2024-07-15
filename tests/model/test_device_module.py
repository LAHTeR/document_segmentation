from document_segmentation.model.device_module import DeviceModuleMixIn


class TestDeviceModule:
    def test_get_device(self):
        assert DeviceModuleMixIn.get_device() in {"cuda", "mps", "cpu"}
