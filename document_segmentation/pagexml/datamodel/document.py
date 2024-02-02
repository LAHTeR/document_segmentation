from typing import Optional

from pydantic import BaseModel, PositiveInt

from .page import Page


class Document(BaseModel):
    """Class for representing a document in a PageXML file."""

    id: str
    inventory_nr: PositiveInt
    inventory_part: Optional[str] = None
    pages: list[Page]
