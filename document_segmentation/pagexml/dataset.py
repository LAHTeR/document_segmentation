import logging
from collections import Counter
from enum import Enum, auto
from typing import Iterable, Union

import pandas as pd
import torch
from pagexml.model.physical_document_model import PageXMLScan
from torch.utils.data import Dataset

from ..settings import PAGEXML_CACHE_DIRECTORY
from .inventory import Inventory


class Label(Enum):
    """Labels for pages in a sequence."""

    BEGIN = auto()
    IN = auto()
    END = auto()
    # OUT = auto()

    @staticmethod
    def map_scores(scores: Iterable[float]) -> dict[str, float]:
        """Map a list of scores to a dictionary of label names and scores.

        Args:
            scores (Iterable[float]): List of scores with the same length as Label.
        Returns:
            dict[str, float]: Dictionary mapping label names to scores.
        """
        if len(scores) != len(Label):
            raise ValueError(
                f"Expected {len(Label)} scores, got {len(scores)}: {scores}"
            )
        return {label.name: score for label, score in zip(Label, scores)}


class PageXmlDataset(Dataset):
    """Dataset of PageXML files with labels."""

    def __init__(self, page_xmls: list[tuple[Inventory, int, Label]]):
        self._page_xmls: list[tuple[Inventory, int, Label]] = page_xmls

    def __repr__(self) -> str:
        return super().__repr__() + f"({len(self)} samples)"

    def __len__(self):
        return len(self._page_xmls)

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[tuple[PageXMLScan, Label], "PageXmlDataset"]:
        """Return a tuple of the PageXMLScan object and the label for the given index or a sub-dataset.

        Args:
            idx (int|slice): The index or a slice object.
        Returns:
            The PageXMLScan object and the label for the given index, or a sub-dataset if a slice object is given.
        """
        if isinstance(idx, slice):
            return self.__class__(self._page_xmls[idx])
        else:
            inventory, page_number, label = self._page_xmls[idx]
            return inventory.pagexml(page_number), label

    def batches(self, batch_size: int) -> Iterable["PageXmlDataset"]:
        """Return a generator over batches of the given size.

        Args:
            batch_size (int): The batch size.
        Returns:
            Iterable[tuple[PageXMLScan, Label]]: A generator over batches of the given size.
        """
        for i in range(0, len(self), batch_size):
            yield self[i : i + batch_size]

    def segments(self) -> Iterable["PageXmlDataset"]:
        """Return a iterable over sub-datasets each containing one complete segment."""
        # TBD: prepend and append previous/succeeding pages with O label?

        begin_idx: int = None
        end_idx: int = None

        for i, label in enumerate(self.labels()):
            if label == Label.BEGIN:
                begin_idx = i
            elif label == Label.IN:
                if begin_idx is None:
                    raise RuntimeError(
                        f"Found IN label without matching BEGIN label at index {i}: {self[i]}"
                    )
            elif label == Label.END:
                if begin_idx is None:
                    raise RuntimeError(
                        f"Found END label without matching BEGIN label at index {i}: {self[i]}"
                    )
                end_idx = i
                yield self[begin_idx : end_idx + 1]

                begin_idx = None
                end_idx = None

        if begin_idx is not None:
            logging.warning(
                f"Found BEGIN label without matching END label at index {begin_idx}: {self[begin_idx]}"
            )

    def find_inv_nr(self, inv_nr: str) -> list[int]:
        """Find all indices with the given inventory number.

        Args:
            inv_nr (str): Inventory number.
        Returns:
            list[int]: List of indices that match the given inventory number.
        """
        return [
            idx
            for idx, (inventory, _, _) in enumerate(self._page_xmls)
            if inventory.inv_nr == str(inv_nr)
        ]

    def page_xmls(self) -> Iterable[PageXMLScan]:
        """Get all PageXMLScan objects in this dataset.

        Returns:
            Iterable[PageXMLScan]: all PageXMLScan objects in the dataset.
        """
        return (page_xml for page_xml, _ in self)

    def labels(self) -> Iterable[Label]:
        """Get all labels in this dataset.

        Returns:
            Iterable[Label]: all labels in the dataset.
        """
        return (label for _, _, label in self._page_xmls)

    def page_ids(self) -> Iterable[str]:
        """Get all page IDs in this dataset.

        This requires downloading and parsing the Inventory/PageXML files.

        Returns:
            Iterable[str]: all page IDs in the dataset.
        """
        return (page_xml.id for page_xml in self.page_xmls())

    def label_tensor(self) -> torch.Tensor:
        """Get a tensor over all labels in this dataset.

        Returns:
            tensor[int]: a Tensor of shape (len(self), len(Label)).
        """
        t = torch.zeros(len(self), len(Label))
        for idx, label in enumerate(self.labels()):
            t[idx, label.value - 1] = 1
        assert t.sum() == len(self)
        return t

    def inverse_frequencies(self) -> list[float]:
        """Get the inverse frequency of each label in this dataset.

        Returns:
            list[float]: List of frequency of each label in dataset dividied by dataset length.
        """
        counts = Counter(self.labels())
        return [len(self) / counts[label] for label in Label]

    @classmethod
    def from_csv(cls, input_file, cache_directory=PAGEXML_CACHE_DIRECTORY):
        """Create a dataset from a CSV file.

        Args:
            input_file (str): Path to CSV file.
            cache_directory (str, optional): Directory to cache PageXML files. Defaults to PAGEXML_CACHE_DIRECTORY.
        """

        df = pd.read_csv(
            input_file, index_col=cls.INDEX_COL, sep=";", dtype={cls.INV_NR_COL: str}
        ).dropna(subset=[cls.INV_NR_COL, cls.BEGIN_PAGE_COL, cls.END_PAGE_COL])

        inventories: dict[int, Inventory] = {
            inventory: Inventory(inventory, cache_directory=cache_directory)
            for inventory in df[cls.INV_NR_COL].unique()
        }

        page_xmls: tuple[Inventory, int, Label] = []
        for index, row in df.iterrows():
            if pd.notna(row[cls.DEEL_VAN_INVENTARIS_COL]):
                logging.warning(
                    f"Document with id '{index}' is part '{row[cls.DEEL_VAN_INVENTARIS_COL]}' of inventory {row[cls.INV_NR_COL]}."
                )

            inv = inventories[row[cls.INV_NR_COL]]
            begin = int(row[cls.BEGIN_PAGE_COL])
            end = int(row[cls.END_PAGE_COL])

            if begin == end:
                logging.warning(
                    f"Document '{index}' has begin page equal to end page: {row}"
                )

            page_xmls.append((inv, begin, Label.BEGIN))
            page_xmls.extend((inv, page, Label.IN) for page in range(begin + 1, end))
            page_xmls.append((inv, end, Label.END))

        return cls(page_xmls)


class GeneraleMissivenDataset(PageXmlDataset):
    """Dataset of PageXML files with labels for the Generale Missiven sheet format."""

    INDEX_COL = "ID"
    INV_NR_COL = "Inv.nr. Nationaal Archief (1.04.02)"
    BEGIN_PAGE_COL = "Beginscan"
    END_PAGE_COL = "Eindscan"
    DEEL_VAN_INVENTARIS_COL = "Deel v. inventarisnummer"
