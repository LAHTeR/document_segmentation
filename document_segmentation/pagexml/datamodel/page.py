from enum import Enum, auto
from typing import Iterable

from pagexml.model.physical_document_model import PageXMLScan
from pydantic import BaseModel, PositiveInt, ValidationError, field_validator

from .region import Region


class Label(Enum):
    """Labels for pages in a sequence."""

    BEGIN = auto()
    IN = auto()
    END = auto()
    # OUT = auto()

    def to_list(self) -> list[int]:
        """Convert the label to a list of integers.

        Returns:
            list[int]: A list of integers representing the label.
        """
        return [int(self == label) for label in Label]

    @staticmethod
    def map_scores(scores: Iterable[float]) -> dict[str, float]:
        """Map a list of scores to a dictionary of label names and scores.

        Args:
            scores (Iterable[float]): List of scores with the same length as Label.
        Returns:
            dict[str, float]: Dictionary mapping label names to scores.
        """
        if len(scores) != len(Label):
            raise ValueError(
                f"Expected {len(Label)} scores, got {len(scores)}: {scores}"
            )
        return {label.name: score for label, score in zip(Label, scores)}


class Page(BaseModel):
    """Class for representing a page in a PageXML file."""

    label: Label
    regions: list[Region]
    scan_nr: PositiveInt

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

        return cls(label=label, regions=regions, scan_nr=scan_nr)
