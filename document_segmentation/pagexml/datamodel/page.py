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
