import gzip
import json
import logging
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Counter, Iterable, Optional
from uuid import UUID

import PIL.Image
import requests
import torch
from pagexml.parser import parse_pagexml_file
from pydantic import BaseModel, PositiveInt, ValidationError, field_validator
from torch.utils.data import Dataset

from ...settings import (
    DEFAULT_BASE_PATH,
    DEFAULT_SERVER,
    DEFAULT_THUMBNAIL_SIZE,
    INV_NR_UUID_MAPPING_FILE,
    INVENTORY_DIR,
    MAX_EMPTY_SEQUENCE,
    MIN_REGION_TEXT_LENGTH,
    SERVER_PASSWORD,
    SERVER_USERNAME,
    THUMBNAILS_DIR,
)
from .label import Label
from .page import Page


class Inventory(BaseModel, Dataset):
    """Represents an inventory of a collection of scans."""

    inv_nr: PositiveInt
    inventory_part: str = ""
    pages: list[Page]

    @field_validator("inventory_part")
    @classmethod
    def validate_inv_part(cls, value: str) -> str:
        """Remove invalid inventory parts.


        Currently, letters are allowed (e.g. A, B, C, I, II) and empty strings. Integers are invalid.
        """

        default_value = ""

        if len(value) == 0:
            # Empty string
            pass
        else:
            try:
                int(value)
            except ValueError:
                # Not an integer
                pass
            else:
                logging.warning("Removing invalid inventory part: '%s'", value)
                value = default_value
        return value

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Inventory)
            and self.inv_nr == other.inv_nr
            and self.inventory_part == other.inventory_part
            and self.pages == other.pages
        )

    def __len__(self) -> int:
        """Return the number of pages in the inventory."""
        return len(self.pages)

    def __getitem__(self, idx: int) -> Page:
        """Return the page at the given index."""
        return self.pages[idx]

    def __repr__(self) -> str:
        return f"Inventory(inv_nr={self.inv_nr}, inventory_part={self.inventory_part}, pages={len(self.pages)} pages)"

    def __str__(self) -> str:
        return self.__repr__()

    def annotate_scan(self, scan_nr: int, label: Label):
        try:
            _, page = self.get_scan(scan_nr)
            page.annotate(label)
        except IndexError as e:
            raise ValueError(f"Scan {scan_nr} not in inventory ({str(self)})") from e

    def get_documents(self) -> list[list[Page]]:
        """Get the documents in the inventory based on page labels.

        Returns:
            list[list[Page]]: A list of documents, where each document is a list of pages.
        """

        documents: list[list[Page]] = []
        if self.pages[0].label in {Label.BOUNDARY, Label.IN}:
            logging.warning(
                f"First page {self.pages[0]} of inventory {self} is part of a document."
            )
            doc = [self.pages[0]]
        else:
            doc = None

        # first and last pages ([0],[-1]) are never part of a document
        for prev, page in zip(self.pages[:-2], self.pages[1:-1], strict=True):
            # validate current state
            if prev.label == Label.OUT and doc is not None:
                raise RuntimeError(
                    f"Page {page.scan_nr} is inside a document, but no document has been started."
                )

            if page.label == Label.UNK:
                logging.warning(
                    f"Unlabelled page {page.scan_nr} in inventory {self.inv_nr}"
                )

            # process page label
            if page.label == Label.BOUNDARY:
                if prev.label == Label.OUT:
                    # Start of new document
                    doc = [page]
                elif prev.label == Label.IN:
                    # End of document
                    doc.append(page)
                    documents.append(doc)
                    doc = None
                elif prev.label == Label.BOUNDARY:
                    if doc is not None:
                        documents.append(doc)  # finish previous document
                    doc = [page]  # start new document
            if page.label == Label.IN:
                if doc is None:
                    raise ValueError(
                        f"Page {page.scan_nr} is inside a document, but no document has been started. Previous: {prev}"
                    )
                if prev.label == Label.OUT:
                    logging.error(f"Invalid label sequence: {prev} -> {page}.")
                    doc = []
                doc.append(page)
            elif page.label == Label.OUT:
                if prev.label == Label.IN:
                    logging.error(f"Invalid label sequence: {prev} -> {page}")
                    documents.append(doc)
                    doc = None
                elif doc and prev.label == Label.BOUNDARY:
                    # Previous page was a single page document
                    documents.append(doc)
                    doc = None

        for doc in documents:
            assert doc, f"Empty document in inventory {self.inv_nr}: {documents}"
        return documents

    def get_scan(self, scan_nr: int) -> tuple[int, Page]:
        """Get the page with the given scan number."""
        for i, page in enumerate(self.pages):
            if page.scan_nr == scan_nr:
                return i, page
        else:
            raise IndexError(f"Scan {scan_nr} not in inventory ({str(self)})")

    def get_scans(self, start_scan_nr: int, end_scan_nr: int) -> list[Page]:
        """Get the pages with the given scan numbers.

        This assumes that the scans in the given range are a consecutive sequence.

        Args:
            start_scan_nr (int): The start scan number (inclusive).
            end_scan_nr (int): The end scan number (exclusive).

        Returns:
            list[Page]: The pages with the given scan numbers.

        Raises:
            RuntimeError: If the difference between start and end scan numbers is not equal to the number of consecutive pages.
        """
        start_index, _ = self.get_scan(start_scan_nr)
        pages: list[Page] = self.pages[
            start_index : start_index + (end_scan_nr - start_scan_nr)
        ]
        if not pages[-1].scan_nr == end_scan_nr - 1:
            raise RuntimeError(
                f"Scan nr of last page does not match requested end scan nr {end_scan_nr}: {pages[-1]}"
            )
        return pages

    def head(self, n: int) -> "Inventory":
        if (not n) or (len(self) <= n):
            return self
        else:
            return Inventory(
                inv_nr=self.inv_nr,
                inventory_part=self.inventory_part,
                pages=self.pages[:n],
            )

    def has_labels(self) -> bool:
        return any(page.label != Label.UNK for page in self.pages)

    def full_inv_nr(self, *, delimiter: str = "") -> str:
        """Return the full inventory number plus part if applicable as a string.

        Args:
            delimiter (str, optional): The delimiter to use between the inventory number and part. Defaults to "".
        Returns:
            str: The full inventory number plus inventory part, e.g. 1574 or 1574A
        """
        return delimiter.join([str(self.inv_nr), self.inventory_part]).rstrip(delimiter)

    def labels(self) -> list[Label]:
        return [page.label for page in self.pages]

    def class_counts(self) -> Counter[Label]:
        return Counter(self.labels())

    def class_weights(self) -> list[float]:
        """Get the inverse frequency of each label in this dataset.

        Applies add-one smoothing to avoid division by zero.

        Returns:
            list[float]: List of frequency of each label in dataset divided by dataset length.
        """
        counts = self.class_counts()
        weights = [len(self) / (counts[label] + 1) for label in Label]
        weights[Label.UNK] = 0.0
        return weights

    def labelled(self) -> list[Page]:
        """Return the labelled pages in the inventory."""
        return [page for page in self.pages if page.label != Label.UNK]

    def labelled_inventories(self) -> Iterable["Inventory"]:
        # TODO: remove this method
        """Split the inventory into segments.

        Each segment will have a continuous sequence of labelled pages.
        Non-labelled pages are cut out.

        Returns:
            Iterable[Inventory]: An iterable of Inventories with labelled pages; singleton if all pages are labelled.
        """
        logging.warning("Deprecated method labelled_inventories()")

        if len(self.labelled()) == 0:
            raise ValueError(f"No labelled pages in inventory: {self}")
        elif len(self.labelled()) == len(self):
            logging.debug(f"All pages are labelled, no need to split up: {self}")
            yield self
        else:
            # yield per labeled segment (documents)
            pages = None
            for page in self.pages:
                if pages is None:
                    # outside of any document
                    if page.label != Label.UNK:
                        # new document starting
                        pages = [page]
                    else:
                        pass
                else:
                    # inside a document
                    if page.label != Label.UNK:
                        # document continues
                        pages.append(page)
                    else:
                        # document ending
                        yield Inventory(
                            inv_nr=self.inv_nr,
                            inventory_part=self.inventory_part,
                            pages=pages,
                        )
                        pages = None
            if pages:
                # final segment
                yield Inventory(
                    inv_nr=self.inv_nr, inventory_part=self.inventory_part, pages=pages
                )

    def label_tensor(self) -> torch.Tensor:
        """Get a tensor over all labels in this dataset.

        Returns:
            tensor[int]: a Tensor of shape (len(self), len(Label)).
        """

        return torch.Tensor([label.to_list() for label in self.labels()])

    def link(self, page: Page) -> str:
        """Get the link to the page on the Nationaal Archief website.

        Example: https://www.nationaalarchief.nl/onderzoeken/archief/1.04.02/invnr/1557/file/NL-HaNA_1.04.02_1557_0026

        Args:
            page (Page): The page (scan) to link to
        Returns:
            str: The link to the page on the Nationaal Archief website.

        """

        inv_nr = self.full_inv_nr().rjust(4, "0")
        doc_id = (page.doc_id or page.guess_doc_id(inv_nr)).removesuffix(".jpg")

        return f"https://www.nationaalarchief.nl/onderzoeken/archief/1.04.02/invnr/{inv_nr}/file/{doc_id}"

    def empty_unlabelled(self) -> "Inventory":
        """Post-process labelled inventories.

        1. Empty all pages with label UNK.
        2. Label empty pages with OUT.
        """
        if not self.has_labels():
            logging.warning(f"No labels in inventory {self.inv_nr}")
        for page in self.pages:
            if page.label == Label.UNK:
                page.empty()
        return self

    def remove_scan(self, scan_nr: int) -> "Inventory":
        """Remove the page with the given scan number."""
        i, _ = self.get_scan(scan_nr)
        self.pages.pop(i)

    def remove_empty_pages(
        self, *, max_length: int = MAX_EMPTY_SEQUENCE, label: Label = Label.OUT
    ) -> "Inventory":
        """Remove all unlabelled pages without text.

        Args:
            max_length (int): The maximum number of blank pages in a sequence. Defaults to 10.
            label (Label): Only remove empty pages with this label. Defaults to Label.OUT.
        Returns:
            Inventory: The inventory with blank pages removed.
        """

        empty_seq: list[Page] = []
        """Indices of empty pages in current sequence"""
        to_delete: list[Page] = []

        for page in self.pages:
            if page.is_shorter_than() and page.label == label:
                # Hit empty and unlabelled page
                empty_seq.append(page)
            elif empty_seq:
                # End of sequence
                if len(empty_seq) > max_length:
                    to_delete.extend(empty_seq[max_length:])
                empty_seq = []

        # Final sequence
        if len(empty_seq) > max_length:
            # End of sequence
            # Mark pages for deletion
            to_delete.extend(empty_seq[max_length:])

        # Delete marked pages
        for page in to_delete:
            self.pages.remove(page)

        return self

    def remove_short_regions(
        self, min_chars: int = MIN_REGION_TEXT_LENGTH
    ) -> "Inventory":
        """Remove all page regions with fewer than `min_chars` characters are removed in-place.

        Args:
            min_chars (int, optional): The minimum number of characters in a region.
                Defaults to settings.MIN_REGION_TEXT_LENGTH.
        Returns:
            Inventory: The inventory with short regions removed.
        """
        self.pages = [page.filter_short_regions(min_chars) for page in self.pages]
        return self

    def split(self, max_size) -> Iterable["Inventory"]:
        """Split the inventory into segments of at most `max_size` pages."""
        if len(self) <= max_size:
            yield self
        else:
            for i in range(0, len(self), max_size):
                yield Inventory(
                    inv_nr=self.inv_nr,
                    inventory_part=self.inventory_part,
                    pages=self.pages[i : i + max_size],
                )

    def write(self, target_file: Optional[Path] = None, mode="xt") -> Path:
        """Write the a Json representation of the inventory to a file.

        Args:
            target_file (Path, optional): The target file path.
            mode (str, optional): The mode to open the file in. Defaults to "xt".

        """
        target_file: Path = target_file or self.__class__.local_file(
            self.inv_nr, self.inventory_part, INVENTORY_DIR
        )

        target_file.parent.mkdir(parents=True, exist_ok=True)

        with target_file.open(mode) as f:
            f.write(self.model_dump_json())
        return target_file

    @staticmethod
    def total_class_weights(inventories: Iterable["Inventory"]) -> list[float]:
        """Get the inverse frequency of each label in this dataset.

        Applies add-one smoothing to avoid division by zero.

        Returns:
            list[float]: List of frequency of each label in dataset divided by dataset length.
        """
        counts: Counter[Label] = sum(
            (inventory.class_counts() for inventory in inventories), start=Counter()
        )
        try:
            total = counts.total()
        except AttributeError as e:
            logging.warning(
                f"Python version: '{sys.version}': {str(e)}. Using sum(counts.values()) instead."
            )
            total = sum(counts.values())

        inverse_freq: list[float] = [total / (counts[label] + 1) for label in Label]
        inverse_freq[Label.UNK] = 0.0

        return inverse_freq

    @staticmethod
    def local_file(inv_nr: int, inventory_part: str, directory: Path) -> Path:
        """Return the path of the inventory Json file.

        Args:
            inv_nr (int): The inventory number.
            inventory_part (str, optional): The inventory part. Defaults to None.
            directory (Path): The path to the inventory directory. Defaults to INVENTORY_DIR.

        Returns:
            Path: The path to the inventory Json file.
        """
        inv_nr: str = f"{inv_nr:>04}"

        if inventory_part := Inventory.validate_inv_part(inventory_part):
            inv_nr = "_".join((inv_nr, inventory_part))

        return (directory / inv_nr).with_suffix(".json")

    @classmethod
    def from_file(cls, file: Path) -> "Inventory":
        if file.suffix in {".gz", ".gzip"}:
            f = gzip.open(file, "rt")
        else:
            f = file.open("rt")
        try:
            return cls.model_validate_json(f.read())
        except ValidationError as e:
            raise ValidationError(
                f"Error loading inventory from file {file}: {e}"
            ) from e

    @classmethod
    def load(
        cls, inv_nr: int, inventory_part: str, inventory_dir: Path = INVENTORY_DIR
    ):
        local_file: Path = cls.local_file(inv_nr, inventory_part, inventory_dir)

        fallback_suffixes: list[str] = {
            local_file.suffix + ".gz",
            local_file.suffix + ".gzip",
        }
        local_files: list[Path] = [local_file] + [
            local_file.with_suffix(suffix) for suffix in fallback_suffixes
        ]

        if existing_file := next((f for f in local_files if f.exists()), None):
            inventory = cls.from_file(existing_file)
        else:
            raise FileNotFoundError(f"None of {local_files} found.")

        return inventory

    @classmethod
    def load_or_download(
        cls, inv_nr: int, inventory_part: str, inventory_dir: Path = INVENTORY_DIR
    ):
        try:
            inventory = Inventory.load(inv_nr, inventory_part, inventory_dir)
        except FileNotFoundError:
            logging.info(f"Downloading inventory {inv_nr}...")
            inventory = cls.download(
                inv_nr, inventory_part, target_directory=inventory_dir
            )
        return inventory

    @classmethod
    def download(
        cls,
        inv_nr: int,
        inventory_part: str = "",
        *,
        username: str = SERVER_USERNAME,
        password: str = SERVER_PASSWORD,
        session: Optional[requests.Session] = None,
        target_directory: Optional[Path] = INVENTORY_DIR,
        server_url=DEFAULT_SERVER,
        base_path=DEFAULT_BASE_PATH,
    ) -> "Inventory":
        """Download an inventory from the repository.

        Args:
            inv_nr (int): The inventory number.
            inventory_part (str, optional): The inventory part. Defaults to None.

        Returns:
            Inventory: The downloaded Inventory object.
        """
        local_file = cls.local_file(inv_nr, inventory_part, target_directory)
        if local_file.exists():
            raise FileExistsError(f"Inventory file {local_file} already exists")

        if session is None:
            session = requests.Session()
        session.auth = (username, password)

        remote_filename = (
            f"{inv_nr:>04}{Inventory.validate_inv_part(inventory_part)}.zip"
        )
        url = "/".join((server_url.rstrip("/"), base_path.rstrip("/"), remote_filename))

        r = session.get(url, stream=True)
        r.raise_for_status()

        pages: list[Page] = []
        with TemporaryDirectory() as tmp_dir:
            local_zip_file = Path(tmp_dir) / remote_filename

            with open(local_zip_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            with zipfile.ZipFile(local_zip_file, "r") as zip_file:
                zip_file.extractall(tmp_dir)
                for filename in zip_file.namelist():
                    tmp_file: Path = Path(tmp_dir) / filename
                    try:
                        pagexml = parse_pagexml_file(tmp_file)
                        scan_nr: int = int(tmp_file.stem.split("_")[-1])
                        pages.append(Page.from_pagexml(Label.UNK, scan_nr, pagexml))
                    except IsADirectoryError:
                        continue
        inventory = cls(
            inv_nr=inv_nr,
            inventory_part=inventory_part,
            pages=sorted(pages, key=lambda page: page.scan_nr),
        )
        inventory.write(local_file)

        return inventory


class ThumbnailDownloader:
    """Retrieve thumbnails for scans in an inventory."""

    def __init__(
        self, mapping: dict[str, UUID], *, thumbnails_dir: Path = THUMBNAILS_DIR
    ) -> None:
        """Initialize the thumbnail downloader.

        Args:
            mapping (dict[str, UUID]): A mapping from inventory numbers to UUIDs.
            thumbnails_dir (Path, optional): The directory where the thumbnails are stored. Defaults to THUMBNAILS_DIR.
        """
        self._base_url: str = "https://service.archief.nl/iipsrv?IIIF="

        self._mapping = mapping
        self._thumbnails_dir = thumbnails_dir

        self._session = requests.Session()

    def get_uuid(self, inventory: Inventory) -> UUID:
        return self._mapping[inventory.full_inv_nr()]

    def _local_file(self, inventory: Inventory, page: Page, size: str) -> Path:
        return (
            self._thumbnails_dir / f"{inventory.full_inv_nr()}_{page.scan_nr}_{size}"
        ).with_suffix(".jpg")

    def thumbnail(
        self, inventory: Inventory, page: Page, *, size: str = DEFAULT_THUMBNAIL_SIZE
    ) -> Path:
        """Get the thumbnail for the given page.

        If the file already exists, it will be read from disk. Otherwise, it will be downloaded.

        Args:
            inventory (Inventory): The inventory.
            page (Page): The page.
            size (str): The size of the thumbnail.
        """
        thumbnail_file = self._local_file(inventory, page, size)

        if not thumbnail_file.exists():
            self.download(inventory, page, size=size)
        return thumbnail_file

    def thumbnail_url(
        self, inventory: Inventory, page: Page, *, size: str = DEFAULT_THUMBNAIL_SIZE
    ) -> str:
        """Get the URL of the thumbnail for the given page.

        Example URL: "https://service.archief.nl/iipsrv?IIIF=/db/77/6f/a8/9d/77/45/ca/8e/85/e1/c4/84/06/a5/55/aa84f770-f5d7-40ac-bfda-db3d06f204c9.jp2/full/100,/0/default.jpg"

        Args:
            inventory (Inventory): The inventory.
            page (Page): The page.
            size (str): The size of the thumbnail.
        Returns:
            str: The URL of the thumbnail.
        """
        # TODO: add other parameters for size (and format)

        _uuid: str = self.get_uuid(inventory).hex
        uuid_part: str = "/".join([_uuid[i : i + 2] for i in range(0, len(_uuid), 2)])
        return f"{self._base_url}/{uuid_part}/{page.external_ref}.jp2/full/{size}/0/default.jpg"

    def download(self, inventory: Inventory, page: Page, *, size: str) -> Path:
        """Download the thumbnail for the given page.

        Args:
            inventory (Inventory): The inventory.
            page (Page): The page.
            size (str): The size of the thumbnail.
        Returns:
            Path: The path to the downloaded thumbnail.
        """
        response: requests.Response = self._session.get(
            self.thumbnail_url(inventory, page, size=size), stream=True
        )
        response.raise_for_status()

        image = PIL.Image.open(BytesIO(response.content))

        target_file = self._local_file(inventory, page, size)

        self._thumbnails_dir.mkdir(parents=True, exist_ok=True)
        image.save(target_file)

        return target_file

    @classmethod
    def from_file(
        cls, file: Path = INV_NR_UUID_MAPPING_FILE, **kwargs
    ) -> "ThumbnailDownloader":
        with file.open("rt") as f:
            mapping = {inv_nr: UUID(_uuid) for inv_nr, _uuid in json.load(f).items()}
        return cls(mapping, **kwargs)

    @classmethod
    def from_url(
        cls,
        url="https://raw.githubusercontent.com/globalise-huygens/knowledge-graph/main/htr/inventory2uuid.json",
    ):
        raise NotImplementedError()
        mapping = requests.get(url).json()
        return cls(mapping)
