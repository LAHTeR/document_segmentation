from enum import IntEnum

import torch


class Label(IntEnum):
    """Labels for pages in a sequence."""

    UNK = 0
    """No annotation available."""
    BEGIN = 1
    START = 1
    """The beginning of a document."""
    IN = 2
    """Part of a document."""
    END = 3
    """The end of a document."""
    OUT = 4
    """Not part of any document."""
    END_BEGIN = 5
    END_START = 5
    START_END = 5
    START_END_START = 5
    """Page on which a document ends, and another one begins"""

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
