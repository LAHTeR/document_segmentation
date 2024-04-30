from enum import Enum, IntEnum, unique

import torch

LABEL_SEPARATOR = "_"


@unique
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
    END_START_END = 5
    START_END_START = 5
    END_START_END_START = 5
    START_END_START_END = 5
    """Page on which a document ends, and another one begins"""

    def to_list(self) -> list[int]:
        """Convert the label to a list of integers.

        Returns:
            list[int]: A list of integers representing the label.
        """
        return [int(self == label) for label in Label]

    def combined(self, other: Enum) -> "Combined":
        try:
            return Combined[LABEL_SEPARATOR.join([self.name, other.name])]
        except KeyError as e:
            raise ValueError(
                f"Invalid combination of label and other category: {self}, {other}"
            ) from e

    @staticmethod
    def to_tensor(labels: list["Label"]) -> torch.Tensor:
        """Convert a list of labels to a tensor.

        Args:
            labels (list[Label]): The labels to convert.

        Returns:
            torch.Tensor: A tensor of length (len(labels)).
        """
        return torch.Tensor([label.value for label in labels]).to(int)

    @classmethod
    def from_combined(cls, combined: Enum) -> "Label":
        label, _ = combined.name.split(LABEL_SEPARATOR)
        return cls[label]


@unique
class Tanap(IntEnum):
    """TANAP categories for documents."""

    DAGREGISTERS = 1
    RESOLUTIES = 2
    CORRESPONDENTIE_HEREN = 3
    CORRESPONDENTIE_GOUVERNEUR = 4
    CORRESPONDENTIE_KANTOREN = 5
    RAPPORTEN = 6
    WETGEVING = 7
    NOTULEN = 8
    MONSTERROLLEN = 10
    FACTUREN = 11
    GROOTBOEKEN = 12
    LIJSTEN = 13
    VERKLARINGEN = 14


@unique
class FrontMatter(IntEnum):
    """Front matter categories (out of documents)."""

    EMPTY = 0
    COVER = 1
    TABLE_OF_CONTENTS = 2
    SECTION_TITLE_PAGE = 3
    DOCUMENT_TITLE_PAGE = 4


Combined = IntEnum(
    "Combined",
    [Label.UNK.name]
    + [
        label.name + LABEL_SEPARATOR + tanap.name
        for tanap in Tanap
        for label in [Label.BEGIN, Label.IN, Label.END, Label.END_BEGIN]
    ]
    + [
        Label.OUT.name + LABEL_SEPARATOR + front_matter.name
        for front_matter in FrontMatter
    ],
)
