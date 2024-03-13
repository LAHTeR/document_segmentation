import csv
import logging
import sys
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

import wandb

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
        shuffle: bool = True,
    ):
        """Train the model on the given dataset.

        Args:
            dataset (DocumentDataset): The dataset to train on.
            epochs (int, optional): The number of epochs to train. Defaults to 3.
            batch_size (int, optional): The batch size. Defaults to 8.
            weights (Optional[list[float]], optional): The weights for the loss function. Defaults to None.
            shuffle (bool, optional): Whether to shuffle the dataset for each epoch. Defaults to True.
        """
        self.train()

        if weights is not None:
            if len(weights) != len(Label):
                raise ValueError(f"Expected {len(Label)} weights, got {len(weights)}.")
            weights = torch.Tensor(weights).to(self._device)

        criterion = nn.CrossEntropyLoss(weight=weights).to(self._device)
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        _wandb = wandb.login()
        if _wandb:
            # TODO: add model architecture (self.parameters()) to config?
            wandb.init(
                project=self.__class__.__name__,
                config={
                    "training size": len(dataset),
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "weights": weights,
                    "shuffle": shuffle,
                },
            )
        else:
            logging.warning(
                "Weights & Biases not available. Set the WANDB_API_KEY environment variable for logging in."
            )

        for epoch in range(epochs):
            if shuffle:
                dataset.shuffle()

            for batch_n, batch in enumerate(
                tqdm(
                    dataset.batches(batch_size),
                    desc="Training",
                    unit="batch",
                    total=dataset.n_batches(batch_size),
                )
            ):
                optimizer.zero_grad()
                outputs = self(batch).to(self._device)

                loss = criterion(outputs, batch.label_tensor().to(self._device)).to(
                    self._device
                )

                loss.backward()
                optimizer.step()

                if _wandb:
                    wandb.log({"loss": loss.item(), "batch_size": len(batch)})

            if _wandb:
                results = {}
                for metric in self.eval_(dataset, batch_size, None):
                    if metric.average is None:
                        results[metric.__class__.__name__] = {
                            label.name: score
                            for label, score in zip(Label, metric.compute().tolist())
                        }
                    else:
                        results[metric.__class__.__name__] = metric.compute().item()
                wandb.log(results)
                self.train()

        if _wandb:
            wandb.finish()
            # TODO: save model to WandB

    def eval_(
        self,
        dataset: DocumentDataset,
        batch_size: int,
        results_out: TextIO = sys.stdout,
    ) -> tuple[Metric, Metric, Metric, Metric]:
        """Evaluate the model on the given dataset.

        Args:
            dataset (DocumentDataset): The dataset to evaluate on.
            batch_size (int): The batch size.
            results_out (TextIO): The file to write the output labels per sample.
                If None (default), does not print the individual test labels.
                Defaults to stdout.
        Returns:
            tuple[Metric, Metric, Metric, Metric]: The precision, recall, F1 score, and accuracy.
        """
        metrics: tuple[Metric] = tuple(
            metric(average=None, num_classes=len(Label))
            for metric in (
                MulticlassPrecision,
                MulticlassRecall,
                MulticlassF1Score,
            )
        ) + (MulticlassAccuracy(),)

        if results_out is not None:
            writer = csv.DictWriter(
                results_out,
                fieldnames=("Predicted", "Actual", "Page ID", "Text", "Scores"),
                delimiter="\t",
            )
            writer.writeheader()

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

            if results_out is not None:
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
