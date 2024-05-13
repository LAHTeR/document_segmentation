from enum import IntEnum, unique

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


@unique
class Tanap(IntEnum):
    """TANAP categories for documents."""

    DAGREGISTERS = 1
    RESOLUTIES = 2
    BRIEVEN_NEDERLAND = 3
    BRIEVEN_BATAVIA = 4
    BRIEVEN_BINNEN = 5
    BRIEVEN_OVERIG = 6
    WETGEVING = 7
    STUKKEN_BATAVIA = 8
    STUKKEN_ANDERE = 9
    STUKKEN_BEVOLKING = 10
    STUKKEN_HANDEL = 11
    STUKKEN_BOEKHOUDING = 12
    STUKKEN_SCHEPEN = 13
    STUKKEN_OVERIG = 14
