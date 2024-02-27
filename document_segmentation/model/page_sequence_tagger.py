from typing import Any, Optional

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from ..pagexml.datamodel.label import Label
from ..settings import PAGE_SEQUENCE_TAGGER_RNN_CONFIG
from .dataset import PageDataset
from .device_module import DeviceModule
from .page_embedding import PageEmbedding


class PageSequenceTagger(nn.Module, DeviceModule):
    """A page sequence tagger that uses an RNN over the regions on a page."""

    _DEFAULT_BATCH_SIZE: int = 8

    def __init__(
        self,
        *,
        rnn_config: dict[str, Any] = PAGE_SEQUENCE_TAGGER_RNN_CONFIG,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._page_embedding = PageEmbedding(device=device)

        # LSTM, because GRU does not seem not to work on MPS: https://github.com/pytorch/pytorch/issues/94691
        self._rnn = nn.LSTM(
            input_size=self._page_embedding.output_size, batch_first=True, **rnn_config
        )

        self._linear = nn.Linear(
            self._rnn.hidden_size * (self._rnn.bidirectional + 1), len(Label)
        )
        self._softmax = nn.Softmax(dim=1)

        self._eval_args = {"average": None, "num_classes": len(Label)}

        self.to_device(device)

    def forward(self, pages: PageDataset):
        page_embeddings = self._page_embedding(pages.pages)

        assert page_embeddings.size() == (
            len(pages),
            self._page_embedding.output_size,
        ), "Bad shape: {pages.size()}"

        rnn_out, hidden = self._rnn(page_embeddings)

        output = self._linear(rnn_out)
        assert output.size() == (len(pages), len(Label)), f"Bad shape: {output.size()}"

        softmax = self._softmax(output)
        assert softmax.size() == (len(pages), len(Label)), f"Bad shape: {output.size()}"

        return softmax

    def train_(
        self,
        pages: PageDataset,
        epochs: int = 3,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        weights: Optional[torch.Tensor] = None,
    ):
        self.train()

        if weights is None:
            weights = pages.class_weights()
        if len(weights) != len(Label):
            raise ValueError(f"Expected {len(Label)} weights, got {len(weights)}.")

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
