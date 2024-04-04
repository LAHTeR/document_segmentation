import json
import logging
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
    INV_NR_UUID_MAPPING_FILE,
    INVENTORY_DIR,
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
        page = None
        try:
            page: Page = self.get_scan(scan_nr)
        except IndexError as e:
            raise ValueError(f"Scan {scan_nr} not in inventory ({str(self)})") from e

        if label != page.label:
            if page.label == Label.UNK:
                page.label = label
            elif {page.label, label} == {Label.BEGIN, Label.END}:
                new_label = Label.END_BEGIN
                logging.info(
                    f"Scan {scan_nr} already has label: {page.label.name}. Changing to {new_label.name}. Inventory: {str(self)}"
                )
                page.label = new_label
            else:
                logging.warning(
                    f"Scan {scan_nr} already has label: {page.label}. Ignoring new label: '{label}'. Inventory: {str(self)}"
                )
        else:
            logging.info(
                f"Scan {scan_nr} already has label: {page.label}. Ignoring new label: {label}. Inventory: {str(self)}"
            )

    def get_scan(self, scan_nr: int) -> Page:
        """Get the page with the given scan number."""
        return self.pages[scan_nr - 1]

    def full_inv_nr(self, *, delimiter: str = "") -> str:
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
        """Split the inventory into segments.

        Each segment will have a continuous sequence of labelled pages.
        Non-labelled pages are cut out.

        Returns:
            Iterable[Inventory]: An iterable of Inventories with labelled pages; singleton if all pages are labelled.
        """

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

    def preprocess(self) -> "Inventory":
        """Preprocess the inventory in-place.

        Returns:
            Inventory: The preprocessed inventory.
        """
        self.remove_short_regions()
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
    def local_file(inv_nr: int, inventory_part: str, directory: Path) -> Path:
        """Return the path of the inventory Json file.

        Args:
            inv_nr (int): The inventory number.
            inventory_part (str, optional): The inventory part. Defaults to None.
            directory (Path): The path to the inventory directory. Defaults to INVENTORY_DIR.

        Returns:
            Path: The path to the inventory Json file.
        """
        inventory_part = Inventory.validate_inv_part(inventory_part)
        filename: str = (
            "_".join((f"{inv_nr:04d}", inventory_part)).rstrip("_") + ".json"
        )
        return directory / filename

    @classmethod
    def load_or_download(
        cls, inv_nr: int, inventory_part: str, inventory_dir: Path = INVENTORY_DIR
    ):
        local_file = cls.local_file(inv_nr, inventory_part, inventory_dir)

        try:
            inventory = Inventory.model_validate_json(local_file.read_text())
        except FileNotFoundError:
            logging.info(f"Downloading inventory {inv_nr}...")
            inventory = cls.download(
                inv_nr, inventory_part, target_directory=inventory_dir
            )
        except ValidationError as e:
            raise ValidationError(
                f"Error loading inventory {inv_nr}_{inventory_part} from file {local_file}: {e}"
            ) from e
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
            f"{inv_nr:04d}{Inventory.validate_inv_part(inventory_part)}.zip"
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
        self._thumbnails_dir.mkdir(parents=True, exist_ok=True)

        self._session = requests.Session()

    def get_uuid(self, inventory: Inventory) -> UUID:
        return self._mapping[inventory.full_inv_nr()]

    def _local_file(self, inventory: Inventory, page: Page, size: str) -> Path:
        return (
            self._thumbnails_dir / f"{inventory.full_inv_nr()}_{page.scan_nr}_{size}"
        ).with_suffix(".jpg")

    def thumbnail(
        self, inventory: Inventory, page: Page, *, size: str = "200,"
    ) -> Path:
        """Get the thumbnail for the given page.

        If the file already exists, it will be read from disk. Otherwise, it will be downloaded.

        Args:
            inventory (Inventory): The inventory.
            page (Page): The page.
            size (str): The size of the thumbnail. Defaults to "200,".
                Valid modes are documented here: https://iiif.io/api/image/2.1/#size
        """
        thumbnail_file = self._local_file(inventory, page, size)

        if not thumbnail_file.exists():
            self.download(inventory, page, size=size)
        return thumbnail_file

    def download(self, inventory: Inventory, page: Page, *, size: str) -> Path:
        # TODO: add other parameters for size (and format)
        # Example URL: "https://service.archief.nl/iipsrv?IIIF=/db/77/6f/a8/9d/77/45/ca/8e/85/e1/c4/84/06/a5/55/aa84f770-f5d7-40ac-bfda-db3d06f204c9.jp2/full/100,/0/default.jpg"

        _uuid: str = self.get_uuid(inventory).hex
        uuid_part: str = "/".join([_uuid[i : i + 2] for i in range(0, len(_uuid), 2)])
        url = f"{self._base_url}/{uuid_part}/{page.external_ref}.jp2/full/{size}/0/default.jpg"

        response: requests.Response = self._session.get(url, stream=True)
        response.raise_for_status()

        image = PIL.Image.open(BytesIO(response.content))

        target_file = self._local_file(inventory, page, size)
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
