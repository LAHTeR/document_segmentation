import logging
import tempfile
import zipfile
from pathlib import Path

import requests
from pagexml.parser import parse_pagexml_file

from ..settings import DEFAULT_BASE_PATH, DEFAULT_SERVER
from .pagexml import PageXML


class Inventory:
    def __init__(self, inv_nr: str, *, cache_directory=None) -> None:
        self._inv_nr: str = inv_nr
        self._cache_directory: Path = Path(cache_directory or tempfile.gettempdir())

        self._server: str = DEFAULT_SERVER
        assert self._server.endswith("/"), "Server URL must end with a slash"
        self._base_path: str = DEFAULT_BASE_PATH
        assert self._base_path.endswith("/"), "Base path must end with a slash"

    @property
    def local_file(self) -> Path:
        filename = f"{self._inv_nr}.zip"
        return self._cache_directory / filename

    def _xml_file_name(self, page: int) -> str:
        return f"NL-HaNA_1.04.02_{self._inv_nr}_{page:04}.xml"

    def download(self, username, password, *, overwrite: bool = False) -> Path:
        if not self.local_file.exists() or overwrite:
            remote_url = f"{self._server}{self._base_path}{self.filename.stem}"
            logging.info(f"Downloading '{remote_url}' to '{self.local_file}'")
            r = requests.get(remote_url, auth=(username, password))
            with self.local_file.open("wb") as f:
                f.write(r.content)
        else:
            logging.warning(f"File {self.local_file} already exists, skipping download")

        assert self.local_file.exists(), f"Downloading {self.local_file} failed."
        return self.local_file

    def pagexml(self, page_nr: int) -> PageXML:
        with zipfile.ZipFile(self.local_file, "r") as zip_ref:
            try:
                xml_file_path = next(
                    name
                    for name in zip_ref.namelist()
                    if self._xml_file_name(page_nr) in name
                )

                with zip_ref.open(
                    xml_file_path
                ) as xml_file, tempfile.NamedTemporaryFile() as tmp:
                    tmp.write(xml_file.read())
                    tmp.flush()
                    return PageXML(parse_pagexml_file(tmp.name))

            except StopIteration:
                raise ValueError(f"Page {page_nr} not found in {self.local_file}")
