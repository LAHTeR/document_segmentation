import torch


def get_device() -> str:
    """Get the device to use for Torch."""

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
