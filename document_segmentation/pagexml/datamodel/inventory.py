import logging
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Counter, Iterable, Optional

import requests
import torch
from pagexml.parser import parse_pagexml_file
from pydantic import BaseModel, PositiveInt, ValidationError, field_validator
from torch.utils.data import Dataset

from ...settings import (
    DEFAULT_BASE_PATH,
    DEFAULT_SERVER,
    INVENTORY_DIR,
    MIN_REGION_TEXT_LENGTH,
    SERVER_PASSWORD,
    SERVER_USERNAME,
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

        if page.label == Label.UNK:
            page.label = label
        elif {page.label, label} == {Label.BEGIN, Label.END}:
            new_label = Label.END_BEGIN
            # TODO: log to info level
            logging.warning(
                f"Scan {scan_nr} already has label {page.label.name}. Changing to {new_label.name}. Inventory: {str(self)}"
            )
            page.label = new_label
        else:
            logging.warning(
                f"Scan {scan_nr} already has label {page.label}. Ignoring new label ({label}). Inventory: {str(self)}"
            )

    def get_scan(self, scan_nr: int) -> Page:
        """Get the page with the given scan number."""
        return self.pages[scan_nr - 1]

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
