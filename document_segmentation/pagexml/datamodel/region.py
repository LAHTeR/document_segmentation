from enum import Enum
from typing import Iterable

from pagexml.model.physical_document_model import PageXMLTextRegion
from pydantic import BaseModel


class RegionType(Enum):
    """The types of regions that are used for the page sequence tagger."""

    CATCH_WORD = "catch-word"
    HEADER = "header"
    MARGINALIA = "marginalia"
    PAGE_NUMBER = "page-number"
    PAGEXML_DOC = "pagexml_doc"
    PARAGRAPH = "paragraph"
    PHYSICAL_STRUCTURE_DOC = "physical_structure_doc"
    SIGNATURE_MARK = "signature-mark"
    TEXT_REGION = "text_region"

    def index(self) -> int:
        """Return the index of the region type."""
        return list(RegionType).index(self)

    @staticmethod
    def indices(region_types: Iterable["RegionType"]) -> list[int]:
        """Return the indices of the region types."""
        return [int(region_type in region_types) for region_type in RegionType]


class Region(BaseModel, frozen=True):
    """Class for a region in a PageXML file."""

    id: str
    types: tuple[RegionType, ...]
    coordinates: tuple[tuple[int, int], ...]
    lines: tuple[str, ...]

    @classmethod
    def from_pagexml(cls, region: PageXMLTextRegion) -> "Region":
        """Create a Region object from a PageXML file.

        Args:
            region (PageXMLTextRegion): The PageXMLTextRegion object.

        Returns:
            A Region object.
        """
        return cls(
            id=region.id,
            types=[RegionType(_type) for _type in region.types],
            coordinates=region.coords.points,
            lines=[line.text for line in region.get_lines() if line.text is not None],
        )
