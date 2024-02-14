import logging
from collections import Counter
from pathlib import Path
from typing import Iterable, Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..pagexml.datamodel import Document, Label, Page


class PageDataset(Dataset):
    """Dataset of documents and pages.

    This Dataset implementation follows the logic of the models defined in document_segmentation.pagexml.model.
    """

    def __init__(self, pages: list[Page]) -> None:
        """Create a dataset from a list of pages.

        Args:
            pages (list[Page]): A list of pages.
        """
        super().__init__()

        self._pages = pages

    @property
    def pages(self) -> list[Page]:
        """Return the pages in this dataset.

        Returns:
            list[Page]: The pages in this dataset.
        """
        return self._pages

    def __repr__(self) -> str:
        return super().__repr__() + f"({len(self)} samples)"

    def __len__(self) -> int:
        """Return the number of pages in this dataset.

        Returns:
            int: The number of pages in this dataset.
        """
        return len(self._pages)

    def __getitem__(self, index) -> Union[Page, "PageDataset"]:
        """Return the page for the given index or a sub-dataset.

        Args:
            index (int): Index of the page.
                If this is a slice, return a sub-dataset.

        """
        if isinstance(index, slice):
            return self.__class__(self._pages[index])
        else:
            return self._pages[index]

    def doc_ids(self) -> list[str]:
        """Return the page IDs in this dataset.

        Returns:
            list[str]: The page IDs in this dataset.
        """
        return [page.doc_id for page in self._pages]

    def labels(self) -> list[Label]:
        """Return a list of labels for each page.

        Returns:
            list[Label]: A list of labels for each page.
        """
        return [page.label for page in self._pages]

    def label_tensor(self) -> torch.Tensor:
        """Get a tensor over all labels in this dataset.

        Returns:
            tensor[int]: a Tensor of shape (len(self), len(Label)).
        """

        t = torch.Tensor([label.to_list() for label in self.labels()])
        assert t.size() == (len(self), len(Label)), f"Bad shape: {t.size()}"
        return t

    def class_weights(self) -> list[float]:
        """Get the inverse frequency of each label in this dataset.

        Add 1 to each frequency to avoid division by zero.

        Returns:
            list[float]: List of frequency of each label in dataset dividied by dataset length.
        """
        counts = Counter(self.labels())
        return [len(self) / (counts[label] + 1) for label in Label]

    def batches(self, batch_size: int) -> Iterable["PageDataset"]:
        """Return a generator over batches of the given size.

        Args:
            batch_size (int): The batch size.
        Returns:
            Iterable[PageDataset]: A generator over batches of the given size.
        """
        for i in range(0, len(self), batch_size):
            yield self[i : i + batch_size]

    @classmethod
    def from_documents(cls, documents: Iterable[Document]):
        """Create a dataset from a collection of documents.

        Args:
            documents (Iterable[Document]): A collection of Document objects.
        """
        pages = []
        for document in documents:
            if document.pages:
                pages.extend(document.pages)
            else:
                logging.warning(f"No pages found in document {document}.")

        return cls(pages)

    @classmethod
    def from_json_files(cls, files: Iterable[Path]):
        """Create a dataset from a list of JSON files.

        Args:
            files (Iterable[Path]): A collection of JSON files.
        """
        return cls.from_documents(
            Document.model_validate_json(file.open("rt").read())
            for file in tqdm(list(files), unit="file", desc="Reading JSON files")
        )

    @classmethod
    def from_dir(cls, data_dir: Path, *, glob="*.json"):
        """Create a dataset from a directory of JSON files.

        Args:
            data_dir (Path): The directory containing the JSON files.
            glob (str, optional): The glob pattern to match the files. Defaults to "*.json".
        """
        return cls.from_json_files(data_dir.glob(glob))
