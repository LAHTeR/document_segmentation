import logging
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel, PositiveInt

from .label import Label
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

    @classmethod
    def from_pages(
        self, pages: list[Page], inventory_nr: int, inventory_part: Optional[str] = None
    ) -> Iterable["Document"]:
        """Segment a list of pages into documents based on their labels.

        Args:
            pages (list[Page]): The pages of the document.
            inventory_nr (int): The inventory number of the document.
            inventory_part (str, optional): The part of the inventory. Defaults to None.
        """

        id = 0
        document_pages: Optional[list[Page]] = None
        """The pages of the current document. If None, we are not in a document."""

        for page in pages:
            if page.label == Label.BEGIN:
                if document_pages is None:
                    document_pages = [page]
                else:
                    logging.warning("Page with label BEGIN found before END.")
                    yield Document(
                        id=str(id),
                        inventory_nr=inventory_nr,
                        inventory_part=inventory_part,
                        pages=document_pages.copy(),
                    )
                    id += 1
                    document_pages = [page]
            elif page.label == Label.IN:
                if document_pages is None:
                    logging.warning("Page with label IN found before BEGIN.")
                    document_pages = [page]
                else:
                    document_pages.append(page)
            elif page.label == Label.END:
                if document_pages is None:
                    logging.warning("Page with label IN found before BEGIN.")
                    document_pages = [page]
                else:
                    document_pages.append(page)

                yield Document(
                    id=str(id),
                    inventory_nr=inventory_nr,
                    inventory_part=inventory_part,
                    pages=document_pages.copy(),
                )
                id += 1
                document_pages = None
            elif page.label == Label.OUT:
                if document_pages is not None:
                    logging.warning(f"Page with label OUT found: {str(page)}")

                    yield Document(
                        id=str(id),
                        inventory_nr=inventory_nr,
                        inventory_part=inventory_part,
                        pages=document_pages.copy(),
                    )
                    id += 1
                    document_pages = None
