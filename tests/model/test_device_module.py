from document_segmentation.model.device_module import DeviceModule


class TestDeviceModule:
    def test_get_device(self):
        assert DeviceModule.get_device() in {"cuda", "mps", "cpu"}
