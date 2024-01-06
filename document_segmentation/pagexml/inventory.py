import logging
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path

import requests
from pagexml.model.physical_document_model import PageXMLScan
from pagexml.parser import parse_pagexml_file

from ..settings import (
    DEFAULT_BASE_PATH,
    DEFAULT_SERVER,
    PAGEXML_CACHE_DIRECTORY,
    SERVER_PASSWORD,
    SERVER_USERNAME,
)


class Inventory:
    def __init__(
        self,
        inv_nr: str,
        *,
        cache_directory=PAGEXML_CACHE_DIRECTORY,
        server_url=DEFAULT_SERVER,
        base_path=DEFAULT_BASE_PATH,
        username=SERVER_USERNAME,
        password=SERVER_PASSWORD,
    ) -> None:
        self._inv_nr: str = inv_nr
        self._cache_directory: Path = cache_directory

        self._server: str = server_url
        assert self._server.endswith("/"), "Server URL must end with a slash"
        self._base_path: str = base_path
        assert self._base_path.endswith("/"), "Base path must end with a slash"

        self._username = username
        if not self._username:
            logging.warning("No username set for accessing the HUC server.")
        self._password = password
        if not self._password:
            logging.warning("No password set for accessing the HUC server.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.inv_nr!r})"

    @property
    def inv_nr(self) -> str:
        return self._inv_nr

    @property
    def local_file(self) -> Path:
        return (self._cache_directory / self._inv_nr).with_suffix(".zip")

    def _page_filename(self, page: int) -> str:
        return f"NL-HaNA_1.04.02_{self._inv_nr}_{page:04}.xml"

    def download(self) -> Path:
        self._cache_directory.mkdir(parents=True, exist_ok=True)

        if self.local_file.exists():
            logging.info(f"File {self.local_file} already exists, skipping download")
        else:
            remote_url = f"{self._server}{self._base_path}{self.local_file.name}"

            logging.info(f"Downloading '{remote_url}' to '{self.local_file}'")
            r = requests.get(remote_url, auth=(self._username, self._password))
            r.raise_for_status()

            with self.local_file.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        if not zipfile.is_zipfile(self.local_file):
            raise RuntimeError(
                f"{self.local_file} for invetory number '{self._inv_nr}' is not a Zip file. Remove file '{self.local_file}' manually to re-download."
            )
        return self.local_file

    def pagexml(self, page_nr: int) -> PageXMLScan:
        self.download()
        return extract_page_xml(self.local_file, self._page_filename(page_nr))


@lru_cache(maxsize=2**16)
def extract_page_xml(zip_file: Path, page_xml_file: str) -> PageXMLScan:
    """Get a PageXMLScan object for the given inventory number and page number.

    This function caches all PageXMLScan objects for all inventories in memory.

    Args:
        zip_file (Path): the Zip file containing the compressed PageXML files.
        page_xml_file (str): the name of the PageXML file to extract (see Inventory._page_filename()

    Returns:
        PageXMLScan: PageXMLScan object.
    """
    with zipfile.ZipFile(zip_file, "r") as zip:
        try:
            # Scan for file name because sub-directory names vary (e.g. "pagexml", "pagexml-2")
            zipped: str = next(
                name for name in zip.namelist() if name.endswith(page_xml_file)
            )

            with zip.open(zipped) as pagexml, tempfile.NamedTemporaryFile() as tmp:
                tmp.write(pagexml.read())
                tmp.flush()
                return parse_pagexml_file(tmp.name)

        except StopIteration:
            raise ValueError(f"File '{page_xml_file}' not found in '{zip_file}'")
