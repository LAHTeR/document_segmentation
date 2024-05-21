from collections import Counter

from pydantic import BaseModel

from .label import Tanap
from .page import Page


class Document(BaseModel):
    pages: list[Page]
    label: Tanap

    def __len__(self):
        return len(self.pages)

    def labels(self):
        return [page.label for page in self.pages]

    def class_counts(self):
        return Counter(self.labels())
