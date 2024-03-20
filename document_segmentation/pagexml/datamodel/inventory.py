import logging
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import requests
from pagexml.parser import parse_pagexml_file
from pydantic import BaseModel, PositiveInt

from ...settings import (
    DEFAULT_BASE_PATH,
    DEFAULT_SERVER,
    INVENTORY_DIR,
    SERVER_PASSWORD,
    SERVER_USERNAME,
)
from .label import Label
from .page import Page


class Inventory(BaseModel):
    """Represents an inventory of a collection of scans."""

    inv_nr: PositiveInt
    inventory_part: str = ""
    pages: list[Page]

    def __len__(self) -> int:
        """Return the number of pages in the inventory."""
        return len(self.pages)

    def __getitem__(self, idx: int) -> Page:
        """Return the page at the given index."""
        return self.pages[idx]

    def labelled(self) -> list[Page]:
        """Return the labelled pages in the inventory."""
        return [page for page in self.pages if page.label != Label.UNK]

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
        return (directory / f"{inv_nr:04d}{inventory_part}").with_suffix(".json")

    @classmethod
    def load_or_download(
        cls, inv_nr: int, inventory_part: str = "", inventory_dir: Path = INVENTORY_DIR
    ):
        local_file = cls.local_file(inv_nr, inventory_part, inventory_dir)

        try:
            inventory = Inventory.model_validate_json(local_file.read_text())
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

        filename = f"{inv_nr:04d}{inventory_part}.zip"
        remote_url = "/".join((server_url.rstrip("/"), base_path.rstrip("/"), filename))

        r = session.get(remote_url, stream=True)
        r.raise_for_status()

        pages: list[Page] = []
        with TemporaryDirectory() as cache_directory:
            local_zip_file = Path(cache_directory) / filename

            with open(local_zip_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            with zipfile.ZipFile(local_zip_file, "r") as zip_file:
                for file in zip_file.filelist:
                    pagexml_path: str = zip_file.extract(file, path=cache_directory)
                    try:
                        pagexml = parse_pagexml_file(pagexml_path)
                        scan_nr: int = int(Path(pagexml_path).stem.split("_")[-1])
                        pages.append(Page.from_pagexml(Label.UNK, scan_nr, pagexml))
                    except IsADirectoryError:
                        continue
        inventory = cls(inv_nr=inv_nr, inventory_part=inventory_part, pages=pages)
        inventory.write(local_file)

        return inventory
