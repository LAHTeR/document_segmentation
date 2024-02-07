from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torcheval.metrics.metric import Metric
from tqdm import tqdm

from ..pagexml.datamodel import Label
from ..settings import PAGE_SEQUENCE_TAGGER_RNN_CONFIG
from .dataset import PageDataset
from .device_module import DeviceModule
from .page_embedding import PageEmbedding


class PageSequenceTagger(nn.Module, DeviceModule):
    """A page sequence tagger that uses an RNN over the regions on a page."""

    _DEFAULT_BATCH_SIZE: int = 32

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

        return softmax  # output of last timestep

    def train_(
        self,
        pages: PageDataset,
        epochs: int = 3,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        weights: list[float] = None,
    ):
        """Train the model.

        Args:
            pages (PageXmlDataset): The dataset to train on.
            epochs (int, optional): The number of epochs. Defaults to 3.
            batch_size (int, optional): The batch size. Defaults to 1.
            weights (list[float], optional): The weights for each label; can be of any type which can be converted into a tensor.
                If not given, use the inverse frequency from the labels in the dataset. Defaults to None.
        """
        self.train()

        if weights is None:
            weights = pages.class_weights()
        if len(weights) != len(Label):
            raise ValueError(f"Expected {len(Label)} weights, got {len(weights)}.")

        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights)).to(self._device)
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for epoch in tqdm(range(epochs), unit="epoch"):
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
            tqdm.write(f"[Loss:\t{loss:.3f}]")

    def _evaluate(self, dataset: PageDataset, metric: Metric, batch_size: int):
        self.eval()

        metric_name = metric.__class__.__name__
        for batch in tqdm(
            dataset.batches(batch_size),
            desc=metric_name,
            unit="batch",
            total=len(dataset) / batch_size,
        ):
            outputs = self(batch)

            true_labels = [label.value - 1 for label in batch.labels()]
            metric.update(outputs, torch.tensor(true_labels))
            scores = Label.map_scores(metric.compute().tolist())
            tqdm.write(f"[{metric_name}: {scores}]")

        return metric.compute()

    def precision(
        self, dataset: PageDataset, *, batch_size: int = _DEFAULT_BATCH_SIZE
    ) -> dict[str, float]:
        metric = MulticlassPrecision(**self._eval_args)
        scores = self._evaluate(dataset, metric, batch_size=batch_size)

        return Label.map_scores(scores.tolist())

    def recall(
        self, dataset: PageDataset, *, batch_size: int = _DEFAULT_BATCH_SIZE
    ) -> dict[str, float]:
        metric = MulticlassRecall(**self._eval_args)
        scores = self._evaluate(dataset, metric, batch_size=batch_size)

        return Label.map_scores(scores.tolist())

    def f1_score(
        self, dataset: PageDataset, *, batch_size: int = _DEFAULT_BATCH_SIZE
    ) -> dict[str, float]:
        metric = MulticlassF1Score(**self._eval_args)
        scores = self._evaluate(dataset, metric, batch_size=batch_size)

        return Label.map_scores(scores.tolist())

    def accuracy(
        self, dataset: PageDataset, *, batch_size: int = _DEFAULT_BATCH_SIZE
    ) -> float:
        metric = MulticlassAccuracy(**self._eval_args)
        scores = self._evaluate(dataset, metric, batch_size=batch_size)

        return scores[Label.IN.value - 1]
