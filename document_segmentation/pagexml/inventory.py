import logging
import shutil
from tempfile import TemporaryDirectory
from typing import Optional
import zipfile
from functools import lru_cache
from pathlib import Path

import requests
from pagexml.model.physical_document_model import PageXMLScan
from pagexml.parser import parse_pagexml_file

from ..settings import (
    DEFAULT_BASE_PATH,
    DEFAULT_SERVER,
    SERVER_PASSWORD,
    SERVER_USERNAME,
)


class InventoryReader:
    def __init__(
        self,
        inv_nr: str,
        *,
        inventory_part: str = None,
        cache_directory=None,
        pagexml_directory=None,
        server_url=DEFAULT_SERVER,
        base_path=DEFAULT_BASE_PATH,
        username: Optional[str] = SERVER_USERNAME,
        password: Optional[str] = SERVER_PASSWORD,
        session: Optional[requests.Session] = None,
        remove_cache_on_exit=True,
    ) -> None:
        """Create an Inventory object.

        Args:
            inv_nr (str): The inventory number of the inventory.
            inventory_part (str, optional): The part of the inventory. Defaults to None.
            cache_directory (Path, optional): The directory in which to cache the inventory. Defaults to PAGEXML_CACHE_DIRECTORY.
            pagexml_directory (Path, optional): The directory in which to cache the PageXML files.
                Defaults to a temporary directory.
            server_url (str, optional): The URL of the HUC server. Defaults to DEFAULT_SERVER.
            base_path (str, optional): The base path on the HUC server. Defaults to DEFAULT_BASE_PATH.
            username (str, optional): The username for accessing the HUC server. Defaults to SERVER_USERNAME.
            password (str, optional): The password for accessing the HUC server. Defaults to SERVER_PASSWORD.
            remove_cache_on_exit (bool, optional): Whether to remove the cache directory when the object is deleted. Defaults to True.
        """
        self._inv_nr: str = str(inv_nr)
        self._inventory_part: Optional[str] = inventory_part

        self._cache_directory: Path = cache_directory or Path(TemporaryDirectory().name)
        self._pagexml_directory: Path = (
            pagexml_directory or self._cache_directory / self.inv_nr
        )
        self._remove_cache_on_exit: bool = remove_cache_on_exit

        self._server: str = server_url
        if not self._server.endswith("/"):
            raise ValueError("Server URL must end with a slash")
        self._base_path: str = base_path
        if not self._base_path.endswith("/"):
            raise ValueError("Base path must end with a slash")

        if session is None:
            self._session = requests.Session()
            self._session.auth = (username, password)
        else:
            self._session = session

    def __enter__(self) -> "InventoryReader":
        self._download()
        self._extract()

        return self

    def __exit__(self, *exc_details):
        self.__delete_cache_dir()

    def __del__(self) -> None:
        self.__delete_cache_dir()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.inv_nr!r})"

    def __delete_cache_dir(self):
        if self._remove_cache_on_exit:
            if self._pagexml_directory.exists():
                if not all(
                    file.name.endswith(".xml") and file.name.startswith("NL-HaNA")
                    for file in self._pagexml_directory.rglob("*")
                    if file.is_file()
                ):
                    raise RuntimeError(
                        f"Only files matching the pattern 'NL-HaNA*.xml' should be in the cache directory '{self._pagexml_directory}'."
                    )
                shutil.rmtree(self._pagexml_directory)

            self.local_zip_file.unlink(missing_ok=True)
            logging.info(f"Deleted cache directory '{self._pagexml_directory}'")
        else:
            logging.info(f"Cache directory '{self._pagexml_directory}' not deleted.")

    @property
    def inv_nr(self) -> str:
        if self._inventory_part is None:
            return self._inv_nr
        else:
            return self._inv_nr + str(self._inventory_part)

    @property
    def local_zip_file(self) -> Path:
        return (self._cache_directory / self.inv_nr).with_suffix(".zip")

    def _page_filename(self, page: int) -> str:
        return f"NL-HaNA_1.04.02_{self.inv_nr}_{page:04}.xml"

    def _download(self) -> Path:
        self._cache_directory.mkdir(parents=True, exist_ok=True)

        if self.local_zip_file.exists():
            raise RuntimeError(
                f"File {self.local_zip_file} already exists, skipping download"
            )

        remote_url = f"{self._server}{self._base_path}{self.local_zip_file.name}"

        logging.info(f"Downloading '{remote_url}' to '{self.local_zip_file}'")
        r = self._session.get(remote_url)
        r.raise_for_status()

        with self.local_zip_file.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        if not zipfile.is_zipfile(self.local_zip_file):
            raise RuntimeError(f"'{self.local_zip_file}' is not a Zip file.")

        return self.local_zip_file

    def _extract(self):
        if self._pagexml_directory.exists():
            raise RuntimeError(f"'{self._pagexml_directory}' already exists.")

        with zipfile.ZipFile(self.local_zip_file, "r") as zip:
            zip.extractall(path=self._pagexml_directory)

    def _find_page_xml(self, page_nr: int) -> Path:
        try:
            page_xml_path: Path = next(
                self._pagexml_directory.rglob(self._page_filename(page_nr))
            )
        except StopIteration as e:
            raise ValueError(
                f"File '{self._page_filename(page_nr)}' not found in '{self._pagexml_directory}'"
            ) from e

        return page_xml_path

    @lru_cache(maxsize=512)
    def pagexml(self, page_nr: int) -> PageXMLScan:
        if not self.local_zip_file.exists():
            self._download()
        if not self._pagexml_directory.exists():
            self._extract()
        page_xml_path: Path = self._find_page_xml(page_nr)

        return parse_pagexml_file(str(page_xml_path))
