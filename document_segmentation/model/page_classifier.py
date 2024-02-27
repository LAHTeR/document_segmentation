from typing import Optional

import torch
from torch import nn, optim
from tqdm import tqdm

from ..pagexml.datamodel.label import Label
from .dataset import PageDataset
from .device_module import DeviceModule
from .page_embedding import PageEmbedding


class PageClassifier(nn.Module, DeviceModule):
    # TODO args
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__()

        self._embedding = PageEmbedding()

        self._linear = nn.Linear(self._embedding.output_size, len(Label), device=device)
        self._softmax = nn.Softmax(dim=1)

        self.to_device(device)

    def forward(self, pages: PageDataset):
        page_embeddings = self._embedding(pages.pages)

        return self._softmax(self._linear(page_embeddings))

    def train_(
        self,
        pages: PageDataset,
        epochs: int,
        batch_size: int,
        weights: Optional[torch.Tensor] = None,
    ):
        self.train()

        criterion = nn.CrossEntropyLoss(weight=weights).to(self._device)
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for _ in range(epochs):
            for batch in tqdm(
                pages.batches(batch_size), unit="batch", total=len(pages) / batch_size
            ):
                optimizer.zero_grad()
                outputs = self(batch).to(self._device)
                loss = criterion(outputs, batch.label_tensor().to(self._device)).to(
                    self._device
                )

                loss.backward()
                optimizer.step()
            if self._device == "mps":
                tqdm.write(
                    f"Current allocated memory (MPS): {torch.mps.current_allocated_memory() / 1024 ** 2:.0f} MB"
                )
                tqdm.write(
                    f"Driver allocated memory (MPS): {torch.mps.driver_allocated_memory() / 1024 ** 2:.0f} MB"
                )
            tqdm.write(f"[Loss:\t{loss:.3f}]")
