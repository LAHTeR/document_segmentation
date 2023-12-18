from typing import Optional

from pagexml.model.physical_document_model import PageXMLScan, PageXMLTextRegion

from ..settings import DOCUMENT_TYPES


class PageXML:
    """A Wrapper around the PageXMLScan class from the pagexml library"""

    def __init__(self, pagexml_scan: PageXMLScan) -> None:
        self._scan = pagexml_scan

    def get_regions_from_top(self) -> list[PageXMLTextRegion]:
        return sorted(
            self._scan.get_inner_text_regions(), key=lambda region: region.coords.top
        )

    def top_paragraph(
        self, region_types=["paragraph", "header"]
    ) -> Optional[PageXMLTextRegion]:
        return next(
            (
                region
                for region in self.get_regions_from_top()
                if any(_type in region.type for _type in region_types)
                and region.stats["words"] > 0
            ),
            None,
        )

    def document_type(self) -> list[str]:
        """Yield one or multiple document types found in the first paragraph/header of the page."""

        first_paragraph = self.top_paragraph()

        if first_paragraph:
            return [
                document_type
                for line in first_paragraph.get_lines()
                if line.text
                for document_type, keywords in DOCUMENT_TYPES.items()
                for keyword in keywords
                if keyword.casefold() in line.text.casefold()
            ]
