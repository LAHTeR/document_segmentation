from document_segmentation.model.util import get_device


def test_get_device():
    assert get_device() in {"cuda", "mps", "cpu"}
