from typing import Optional

from pagexml.model.physical_document_model import PageXMLScan
from pydantic import BaseModel, PositiveInt, ValidationError, field_validator

from .label import Label
from .region import Region


class Page(BaseModel):
    """Class for representing a page in a PageXML file."""

    label: Label
    regions: list[Region]
    scan_nr: PositiveInt
    doc_id: Optional[str] = None

    @field_validator("label")
    def enum_from_int(cls, value):
        if isinstance(value, int):
            try:
                value = Label(value)
            except ValueError as e:
                raise ValidationError from e
        return value

    def text(self, delimiter: str = "\n") -> str:
        """Return the text of the page.

        Args:
            delimiter (str, optional): The delimiter to use between lines. Defaults to "\n".
        """
        return delimiter.join(line for region in self.regions for line in region.lines)

    def filter_short_regions(self, min_chars: int = 1) -> "Page":
        """Remove regions with fewer than `min_chars` characters.

        Args:
            min_chars (int, optional): The minimum number of characters in a region. Defaults to 1.
        """
        return Page(
            label=self.label,
            regions=[region for region in self.regions if len(region) >= min_chars],
            scan_nr=self.scan_nr,
            doc_id=self.doc_id,
        )

    @classmethod
    def from_pagexml(cls, label: Label, scan_nr: int, pagexml: PageXMLScan):
        """Create a Page object from a PageXML object.

        Args:
            label (Label): The label of the page.
            scan_nr (int): The scan number of the page.
            pagexml (PageXMLScan): The PageXMLScan object.
        """
        regions = [
            Region.from_pagexml(region)
            for region in pagexml.get_text_regions_in_reading_order()
        ]

        return cls(label=label, regions=regions, scan_nr=scan_nr, doc_id=pagexml.id)
