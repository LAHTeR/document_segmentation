import abc
import inspect
import logging
from typing import Optional

import torch
from torch import nn


class DeviceModule(abc.ABC):
    """A MixIn class for moving all Torch (sub-)modules to a device."""

    def to_device(self, device: Optional[str] = None):
        """Move the module and all its Torch modules to a device.

        Args:
            device: The device to move to.
        Returns:
            self
        """

        if device is None:
            device = DeviceModule.get_device()
        self._device = device

        logging.info(f"Using device: {self._device}")

        for name, module in inspect.getmembers(
            self, lambda x: isinstance(x, nn.Module)
        ):
            module.to(self._device)

        return self

    @staticmethod
    def get_device() -> str:
        """Get the device to use for Torch."""

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
