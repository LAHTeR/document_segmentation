import abc
import inspect
import logging
from types import ModuleType
from typing import Optional

import torch


class DeviceModuleMixIn(abc.ABC):
    """A MixIn class for moving all Torch (sub-)modules to a device."""

    # FIXME: only move to device if not already on the device
    def to_device(self, device: Optional[str] = None):
        """Move the module and all its Torch modules to a device.

        Sets self._device to the device used.

        Args:
            device: The device to move to. If not given, auto-detect available devices.
        Returns:
            self
        """

        self._device = device or DeviceModuleMixIn.get_device()

        logging.info(f"Using device: {self._device}")

        for name, module in inspect.getmembers(self):
            try:
                module.to_device(self._device)
                logging.info(
                    "Moving sub-modules of '%s' to device '%s'",
                    self.__class__.__name__,
                    self._device,
                )
            except AttributeError:
                pass

            try:
                module.to(self._device)
                logging.info(
                    "Moving module '%s' to device '%s'",
                    ".".join((self.__class__.__name__, name)),
                    self._device,
                )
            except AttributeError:
                pass

        return self

    @staticmethod
    def get_device(
        backends: list[ModuleType] = [torch.cuda, torch.backends.mps, torch.cpu],
    ) -> str:
        """Get the device to use for Torch. Returns the name of the first backend that is available.

        Args:
            backends: A list of modules to check for availability.
                Defaults to [torch.cuda, torch.backends.mps, torch.cpu].
        """

        # TODO: this does not work for multiple CUDA devices ("cuda" vs "cuda:0")
        return next(
            backend.__name__.split(".")[-1]
            for backend in backends
            if backend.is_available()
        )
