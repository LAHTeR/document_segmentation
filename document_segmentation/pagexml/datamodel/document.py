import torch
from annotated_types import MinLen
from typing_extensions import Annotated

from ...pagexml.datamodel.inventory import Inventory
from .label import Tanap
from .page import Page


class Document(Inventory):
    """A document with a label and a list of pages.

    Sub-classes the Inventory class.
    """

    label: Tanap
    pages: Annotated[list[Page], MinLen(1)]

    def label_tensor(self) -> torch.Tensor:
        """Get a tensor for the label of this document.

        Returns:
            tensor[int]: a Tensor of shape 1x1."""
        label_list: list[int] = [0] * len(Tanap)
        label_list[self.label] = 1
        return torch.Tensor(label_list)
