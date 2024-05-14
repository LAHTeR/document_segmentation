from enum import IntEnum

import torch


class Label(IntEnum):
    """Labels for pages in a sequence."""

    UNK = 0
    """No annotation available."""
    BOUNDARY = 1
    """The beginning or end of a document."""
    IN = 2
    """Part of a document."""
    OUT = 3

    def to_list(self) -> list[int]:
        """Convert the label to a list of integers.

        Returns:
            list[int]: A list of integers representing the label.
        """
        return [int(self == label) for label in Label]

    @staticmethod
    def to_tensor(labels: list["Label"]) -> torch.Tensor:
        """Convert a list of labels to a tensor.

        Args:
            labels (list[Label]): The labels to convert.

        Returns:
            torch.Tensor: A tensor of length (len(labels)).
        """
        return torch.Tensor([label.value for label in labels]).to(int)
