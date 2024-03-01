import csv
import logging
from typing import Any, Optional, TextIO

import torch
import torch.nn as nn
from torch import optim
from torcheval.metrics import (
    Metric,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from tqdm import tqdm

from ..pagexml.datamodel.label import Label
from ..settings import PAGE_SEQUENCE_TAGGER_RNN_CONFIG
from .dataset import DocumentDataset, PageDataset
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
        dataset: DocumentDataset,
        epochs: int = 3,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        weights: Optional[list[float]] = None,
    ):
        self.train()

        if weights is not None:
            if len(weights) != len(Label):
                raise ValueError(f"Expected {len(Label)} weights, got {len(weights)}.")
            weights = torch.Tensor(weights).to(self._device)

        criterion = nn.CrossEntropyLoss(weight=weights).to(self._device)
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for _ in range(epochs):
            for batch in tqdm(
                dataset.batches(batch_size),
                desc="Training",
                unit="batch",
                total=dataset.n_batches(batch_size),
            ):
                optimizer.zero_grad()
                outputs = self(batch).to(self._device)
                loss = criterion(outputs, batch.label_tensor().to(self._device)).to(
                    self._device
                )

                loss.backward()
                optimizer.step()
            if self._device == "mps":
                logging.debug(
                    f"Current allocated memory (MPS): {torch.mps.current_allocated_memory() / 1024 ** 2:.0f} MB"
                )
                logging.debug(
                    f"Driver allocated memory (MPS): {torch.mps.driver_allocated_memory() / 1024 ** 2:.0f} MB"
                )

            tqdm.write(f"[Loss:\t{loss:.3f}]")

    def eval_(
        self, dataset: DocumentDataset, batch_size: int, results_out: TextIO
    ) -> list[Metric]:
        metrics: list[Metric] = [
            metric(average=None, num_classes=len(Label))
            for metric in (
                MulticlassPrecision,
                MulticlassRecall,
                MulticlassF1Score,
            )
        ] + [MulticlassAccuracy()]

        writer = csv.DictWriter(
            results_out,
            fieldnames=("Predicted", "Actual", "Page ID", "Text", "Scores"),
            delimiter="\t",
        )

        self.eval()

        for batch in tqdm(
            dataset.batches(batch_size),
            desc="Evaluating",
            total=dataset.n_batches(batch_size),
            unit="batch",
        ):
            predicted = self(batch)
            labels = batch.labels()

            _labels = torch.Tensor([label.value for label in labels]).to(int)

            for metric in metrics:
                metric.update(predicted, _labels)

            for page, pred, label in zip(batch.pages, predicted, labels):
                writer.writerow(
                    {
                        "Predicted": Label(pred.argmax().item()).name,
                        "Actual": label.name,
                        "Page ID": page.doc_id,
                        "Text": page.text(delimiter="; ")[:50],
                        "Scores": str(pred.tolist()),
                    }
                )

        return metrics
