from pydantic import BaseModel

from .label import Tanap
from .page import Page


class Document(BaseModel):
    pages: list[Page]
    label: Tanap

    def __len__(self):
        return len(self.pages)
