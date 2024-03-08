import abc
import logging
import random
from collections import Counter
from itertools import islice
from math import ceil
from pathlib import Path
from typing import Iterable, Optional, Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..pagexml.datamodel.document import Document
from ..pagexml.datamodel.label import Label
from ..pagexml.datamodel.page import Page
from ..pagexml.datamodel.region import Region
from ..settings import MIN_REGION_TEXT_LENGTH


class AbstractDataset(Dataset, abc.ABC):
    def __repr__(self) -> str:
        return super().__repr__() + f"({len(self)} samples)"

    def _class_counts(self) -> dict[Label, int]:
        return Counter(self.labels())

    @abc.abstractmethod
    def labels(self) -> list[Label]:
        return NotImplemented

    @abc.abstractmethod
    def balance(self, max_size: Optional[int] = None) -> "AbstractDataset":
        return NotImplemented

    @abc.abstractmethod
    def shuffle(self) -> "AbstractDataset":
        return NotImplemented

    def _indices(self, label: Label) -> list[int]:
        return [i for i, _label in enumerate(self.labels()) if _label == label]

    def class_weights(self) -> list[float]:
        """Get the inverse frequency of each label in this dataset.

        Applies add-one smoothing to avoid division by zero.

        Returns:
            list[float]: List of frequency of each label in dataset dividied by dataset length.
        """
        counts = self._class_counts()
        return [len(self) / (counts[label] + 1) for label in Label]

    def batches(
        self, batch_size: int, *, shuffle: bool = False
    ) -> Iterable["AbstractDataset"]:
        """Return a generator over batches of the given size.

        Args:
            batch_size (int): The batch size.
            shuffle (bool, optional): Whether to shuffle the dataset before batching. Defaults to False.
        Returns:
            Iterable[AbstractDataset]: A generator over batches of the given size.
        """
        for i in range(0, len(self), batch_size):
            yield self[i : i + batch_size]

    def label_tensor(self) -> torch.Tensor:
        """Get a tensor over all labels in this dataset.

        Returns:
            tensor[int]: a Tensor of shape (len(self), len(Label)).
        """

        return torch.Tensor([label.to_list() for label in self.labels()])


class DocumentDataset(AbstractDataset):
    """A dataset that wraps PageDatasets from different documents."""

    def __init__(self, page_datasets: list["PageDataset"] = None) -> None:
        super().__init__()

        self._page_datasets: list[PageDataset] = page_datasets or []

    def __add__(self, other: Dataset) -> "DocumentDataset":
        return self.__class__(self._page_datasets + other._page_datasets)

    def __len__(self) -> int:
        return len(self._page_datasets)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, self.__class__)
            and self._page_datasets == other._page_datasets
        )

    def __getitem__(self, index) -> Page:
        if isinstance(index, slice):
            return self.__class__(self._page_datasets[index])
        else:
            return self._page_datasets[index]

    def balance(self, max_size: Optional[int] = None) -> AbstractDataset:
        # FIXME:
        logging.warning("max_size is applied per document, not in total.")

        return self.__class__(
            [page_dataset.balance(max_size) for page_dataset in self._page_datasets]
        )

    def shuffle(self) -> "DocumentDataset":
        """Shuffle the order of the sub-datasets in place."""
        random.shuffle(self._page_datasets)

    def labels(self) -> list[Label]:
        """All labels for all sub-datasets."""
        return sum(
            [page_dataset.labels() for page_dataset in self._page_datasets], start=[]
        )

    def batches(
        self, batch_size: int, *, min_region_length: int = MIN_REGION_TEXT_LENGTH
    ) -> Iterable["PageDataset"]:
        """
        Return a generator over batches of the given size.

        Regions with fewer than `min_region_length` characters are removed on-the-fly.

        All pages in a batch belong to the same document.
        For documents shorter than the batch size, the batch consists of all pages of that document.

        Args:
            batch_size (int): The maximum batch size.
                Batches are based on documents, so they may be smaller.
            min_region_length (int, optional): The minimum number of characters in a region.
                Defaults to MIN_REGION_TEXT_LENGTH.
        Returns:
            Iterable[PageDataset]: A generator over PageDatasets of the given size (maximum).
        """

        for dataset in self._page_datasets:
            yield from dataset.remove_short_regions(min_region_length).batches(
                batch_size
            )

    def n_batches(self, batch_size: int) -> int:
        """Compute the number of batches of the given size."""
        return sum(
            ceil(len(page_dataset) / batch_size) for page_dataset in self._page_datasets
        )

    def split(self, portion: float) -> tuple["DocumentDataset", "DocumentDataset"]:
        """Split the dataset into two parts.

        Args:
            portion (float): The portion of the dataset to take in the first part (e.g. 0.8).
        Returns:
            tuple[DocumentDataset, DocumentDataset]: The two parts of the dataset.
        """
        split = int(len(self._page_datasets) * portion)
        return self.__class__(self._page_datasets[:split]), self.__class__(
            self._page_datasets[split:]
        )

    @classmethod
    def from_documents(cls, documents: Iterable[Document]) -> "DocumentDataset":
        """Create a dataset from a collection of documents."""
        return cls([PageDataset(document.pages) for document in documents])

    @classmethod
    def from_json_files(cls, files: Iterable[Path]):
        """Create a dataset from a list of JSON files.

        The Json files are supposed to contain a list of documents as defined in the `Document` class.

        Args:
            files (Iterable[Path]): A collection of JSON files.
        """
        return cls.from_documents(
            Document.model_validate_json(file.open("rt").read())
            for file in tqdm(list(files), unit="file", desc="Reading JSON files")
        )

    @classmethod
    def from_dir(cls, data_dir: Path, *, glob="*.json", n: int = None):
        """Create a dataset from a directory that contains JSON files.

        Args:
            data_dir (Path): The directory containing the JSON files.
            glob (str, optional): The glob pattern to match the files. Defaults to "*.json".
        """
        return cls.from_json_files(islice(data_dir.glob(glob), n))


class PageDataset(AbstractDataset):
    """Dataset of documents and pages.

    This Dataset implementation follows the logic of the models defined in document_segmentation.pagexml.model.
    """

    def __init__(self, pages: list[Page] = None) -> None:
        """Create a dataset from a list of pages.

        Args:
            pages (list[Page]): A list of pages.
        """
        super().__init__()

        self._pages = pages or []

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self._pages == other._pages

    def __add__(self, other: Dataset) -> "PageDataset":
        """Concatenate two datasets."""
        return self.__class__(self._pages + other._pages)

    @property
    def pages(self) -> list[Page]:
        """Return the pages in this dataset.

        Returns:
            list[Page]: The pages in this dataset.
        """
        return self._pages

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

    def regions(self) -> Iterable[Region]:
        """Return a generator over all regions in this dataset."""
        for page in self._pages:
            yield from page.regions

    def region_labels(self) -> Iterable[Label]:
        """Return the label of the page for each region."""
        for page in self._pages:
            yield from [page.label] * len(page.regions)

    def remove_short_regions(
        self, min_chars: int = MIN_REGION_TEXT_LENGTH
    ) -> "PageDataset":
        """Create a filtered dataset in which all page regions with fewer than `min_chars` characters are removed.

        Args:
            min_chars (int, optional): The minimum number of characters in a region.
                Defaults to settings.MIN_REGION_TEXT_LENGTH.
        """
        return self.__class__(
            [page.filter_short_regions(min_chars) for page in self._pages]
        )

    def balance(self, max_size: Optional[int] = None) -> "RegionDataset":
        """Balance the dataset by keeping a maximum number of samples per class.

        If a class has fewer samples than `max_size`, all samples are kept.
        Empty classes are ignored.

        Args:
            max_size (Optional[int], optional): The maximum number of samples to take from each class.
                If None (default), take the minimum non-zero number of samples from any class.
        Returns:
            RegionDataset: A new dataset with balanced classes.
                The samples are added per class, so the new dataset should be shuffled.
        """
        if max_size is None:
            counts = self._class_counts()
            if counts:
                max_size = min([value for value in counts.values() if value > 0])
            else:
                max_size = 0

        new_pages = []

        for label in Label:
            sample: list[int] = self._indices(label)
            if len(sample) > max_size:
                sample = random.sample(sample, max_size)
            new_pages.extend([self._pages[i] for i in sample])

        return self.__class__(new_pages)

    def shuffle(self) -> "PageDataset":
        indices = list(range(len(self)))
        random.shuffle(indices)
        return self.__class__([self._pages[index] for index in indices])

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


class RegionDataset(AbstractDataset):
    def __init__(self, regions: list[Region], labels: list[Label]) -> None:
        if len(regions) != len(labels):
            raise ValueError(f"Got {len(regions)} regions, but {len(labels)} labels.")

        self._labels: list[Label] = labels
        self._regions: list[Region] = regions

    def __hash__(self):
        raise NotImplementedError()

    def __eq__(self, other) -> bool:
        return self._labels == other._labels and self._regions == other._regions

    def __len__(self) -> int:
        """Return the number of pages in this dataset.

        Returns:
            int: The number of pages in this dataset.
        """
        return len(self._regions)

    def __getitem__(self, index) -> Union[Region, "PageDataset"]:
        """Return the page for the given index or a sub-dataset.

        Args:
            index (int): Index of the page.
                If this is a slice, return a sub-dataset.

        """
        if isinstance(index, slice):
            return self.__class__(self._regions[index], self._labels[index])
        else:
            return self._regions[index]

    def labels(self) -> list[Label]:
        return self._labels

    def balance(self, max_size: Optional[int] = None) -> "RegionDataset":
        """Balance the dataset by keeping a maximum number of sample per class.

        If a class has fewer samples than `max_size`, all samples are kept.
        Empty classes are ignored.

        Args:
            max_size (Optional[int], optional): The maximum number of samples to take from each class.
                If None (default), take the minimum non-zero number of samples from any class.
        Returns:
            RegionDataset: A new dataset with balanced classes.
                The samples are added per class, so the new dataset should be shuffled.
        """
        if max_size is None:
            counts = self._class_counts()
            if counts:
                max_size = min([value for value in counts.values() if value > 0])
            else:
                max_size = 0

        new_regions = []
        new_labels = []

        for label in Label:
            sample: list[int] = self._indices(label)
            if len(sample) > max_size:
                sample = random.sample(sample, max_size)
            new_regions.extend([self._regions[i] for i in sample])
            new_labels.extend([self._labels[i] for i in sample])

        return self.__class__(new_regions, new_labels)

    def regions(self) -> list[Region]:
        return self._regions

    def sample(self, k) -> "RegionDataset":
        indices = random.sample(range(len(self)), k)
        regions = [self._regions[index] for index in indices]
        labels = [self._labels[index] for index in indices]
        return self.__class__(regions, labels)

    def shuffle(self) -> "RegionDataset":
        indices = list(range(len(self)))
        random.shuffle(indices)
        regions = [self._regions[index] for index in indices]
        labels = [self._labels[index] for index in indices]
        return self.__class__(regions, labels)

    def remove_empty(self, min_chars: int = 1) -> "RegionDataset":
        """Generate a new dataset only with regions with more than `min_chars` characters.

        Args:
            min_chars (int, optional): The minimum number of characters in a region. Defaults to 1.

        """
        keep_indices: list[int] = [
            i for i, region in enumerate(self._regions) if len(region) >= min_chars
        ]
        regions = [self._regions[i] for i in keep_indices]
        labels = [self._labels[i] for i in keep_indices]

        return self.__class__(regions, labels)

    @classmethod
    def from_page_dataset(cls, page_dataset: PageDataset):
        return cls(list(page_dataset.regions()), list(page_dataset.region_labels()))
