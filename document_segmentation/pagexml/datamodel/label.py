from enum import IntEnum, unique

import torch


@unique
class Label(IntEnum):
    @staticmethod
    def to_tensor(labels: list["Label"]) -> torch.Tensor:
        """Convert a list of labels to a tensor.

        Args:
            labels (list[Label]): The labels to convert.

        Returns:
            torch.Tensor: A tensor of length (len(labels)).
        """
        return torch.Tensor([label.value for label in labels]).to(torch.int64)

    def to_list(self) -> list[int]:
        """Convert the label to a list of integers.

        Returns:
            list[int]: A list of integers representing the label.
        """

        return [int(self == label) for label in self.__class__]


class SequenceLabel(Label):
    """Labels for pages in a sequence."""

    UNK = 0
    """No annotation available."""
    BOUNDARY = 1
    """The beginning or end of a document."""
    IN = 2
    """Part of a document."""
    OUT = 3


class Tanap(Label):
    """TANAP categories for documents."""

    UNK = 0
    """No annotation available."""

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

    FRONT_MATTER = 15
    """Front or back matter of a document."""


class DocumentType(Label):
    UNK = 0

    AKTE = 1
    APPOINTEMENT = 2
    BRIEF = 3
    DAGREGISTER = 4
    EIS = 5
    FACTUUR = 6
    GROOTBOEK = 7
    INSTRUCTIE = 8
    INVENTARIS = 9
    LIJST = 10
    LIQUIDATIE = 11
    MEMORIE = 12
    MISCELLANEOUS = 13
    NOTULEN = 14
    OVEREENKOMST = 15
    PAS = 16
    PROCESSTUK = 17
    RAPPORT = 18
    REGISTER = 19
    REKENING = 20
    REKEST = 21
    RENDEMENT = 22
    RESOLUTIE = 23
    SENTENTIE = 24
    VERKLARING = 25
    WETGEVING = 26

    FRONT_MATTER = 27
