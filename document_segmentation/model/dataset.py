import logging
import random
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Iterable, Optional, Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..pagexml.datamodel.document import Document
from ..pagexml.datamodel.label import Label
from ..pagexml.datamodel.page import Page
from ..pagexml.datamodel.region import Region
from ..settings import MAX_REGIONS_PER_PAGE


class AbstractDataset(Dataset):
    def __repr__(self) -> str:
        return super().__repr__() + f"({len(self)} samples)"

    def _class_counts(self) -> dict[Label, int]:
        return Counter(self.labels())

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
    ) -> Iterable["PageDataset"]:
        """Return a generator over batches of the given size.

        Args:
            batch_size (int): The batch size.
            shuffle (bool, optional): Whether to shuffle the dataset before batching. Defaults to False.
        Returns:
            Iterable[PageDataset]: A generator over batches of the given size.
        """
        for i in range(0, len(self), batch_size):
            yield self[i : i + batch_size]

    def label_tensor(self) -> torch.Tensor:
        """Get a tensor over all labels in this dataset.

        Returns:
            tensor[int]: a Tensor of shape (len(self), len(Label)).
        """

        return torch.Tensor([label.to_list() for label in self.labels()])


class PageDataset(AbstractDataset):
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

    @property
    def max_regions(self) -> int:
        return len(self) * MAX_REGIONS_PER_PAGE

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
        for page in self._pages:
            yield from page.regions

    def region_labels(self) -> Iterable[Label]:
        """Return the label of the page for each region."""
        for page in self._pages:
            yield from [page.label] * len(page.regions)

    # def region_labels(self) -> Iterable[Label]:
    #     for page in self._pages:
    #         if page.regions:
    #             if page.label.name == Label.BEGIN.name:
    #                 yield Label.BEGIN
    #                 yield from [Label.IN] * (len(page.regions) - 1)
    #             elif page.label.name == Label.IN.name:
    #                 yield from [Label.IN] * len(page.regions)
    #             elif page.label.name == Label.END.name:
    #                 yield from [Label.IN] * (len(page.regions) - 1)
    #                 yield Label.END
    #             elif page.label.name == Label.OUT.name:
    #                 yield from [Label.OUT] * len(page.regions)

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
    def from_dir(cls, data_dir: Path, *, glob="*.json", n: int = None):
        """Create a dataset from a directory of JSON files.

        Args:
            data_dir (Path): The directory containing the JSON files.
            glob (str, optional): The glob pattern to match the files. Defaults to "*.json".
        """
        return cls.from_json_files(islice(data_dir.glob(glob), n))


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

    def _indices(self, label: Label) -> list[int]:
        return [i for i, _label in enumerate(self._labels) if _label == label]

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

    def labels(self) -> list[Label]:
        return self._labels

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
