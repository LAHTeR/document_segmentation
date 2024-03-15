from pathlib import Path
from typing import Optional

from pydantic import BaseModel, PositiveInt

from .page import Page


class Document(BaseModel):
    """Class for representing a document in a PageXML file."""

    id: str
    inventory_nr: PositiveInt
    inventory_part: Optional[str] = None
    pages: list[Page]

    @classmethod
    def from_json_file(cls, file: Path):
        """Create a dataset from a list of JSON files.

        The Json files are supposed to contain a list of documents as defined in the `Document` class.

        Args:
            files (Iterable[Path]): A collection of JSON files.
        """
        return Document.model_validate_json(file.open("rt").read())
