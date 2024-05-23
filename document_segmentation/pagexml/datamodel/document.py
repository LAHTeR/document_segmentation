from collections import Counter

import torch
from annotated_types import MinLen
from pydantic import BaseModel
from typing_extensions import Annotated

from .label import Tanap
from .page import Page


# TODO: add common (abstract) class for Document, Inventory with common methods
class Document(BaseModel):
    pages: Annotated[list[Page], MinLen(1)]
    label: Tanap

    def __len__(self):
        return len(self.pages)

    def labels(self):
        return [page.label for page in self.pages]

    def class_counts(self):
        return Counter(self.label)

    def label_tensor(self) -> torch.Tensor:
        label_list: list[int] = [0] * len(Tanap)
        label_list[self.label] = 1
        return torch.Tensor(label_list)
