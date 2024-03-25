import csv
import logging
import math
import random
import sys
from typing import Any, Optional, TextIO

import torch
import torch.nn as nn
import wandb
from torch import optim
from torcheval.metrics import (
    Metric,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from tqdm import tqdm

from document_segmentation.pagexml.datamodel.inventory import Inventory
from document_segmentation.pagexml.datamodel.page import Page

from ..pagexml.datamodel.label import Label
from ..settings import PAGE_SEQUENCE_TAGGER_RNN_CONFIG
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

    def forward(self, pages: list[Page]):
        page_embeddings = self._page_embedding(pages)

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
        training_inventories: list[Inventory],
        validation_inventories: Optional[list[Inventory]] = None,
        *,
        epochs: int = 3,
        weights: Optional[list[float]] = None,
        shuffle: bool = True,
        log_wandb: bool = True,
    ):
        """Train the model on the given dataset.

        Args:
            training_inventories (list[Inventory]): The dataset to train on.
            validation_inventories (Optional[list[Inventory]], optional): The validation dataset.
                Defaults to None. If given, will evaluate after each epoch.
            epochs (int, optional): The number of epochs to train. Defaults to 3.
            weights (Optional[list[float]], optional): The weights for the loss function. Defaults to None.
            shuffle (bool, optional): Whether to shuffle the dataset for each epoch. Defaults to True.
            log_wandb (bool, optional): Whether to log the training to Weights & Biases. Defaults to True.
        """
        self.train()

        if weights is None:
            # TODO: weighed average?
            weights = (
                torch.Tensor(
                    [inventory.class_weights() for inventory in training_inventories]
                )
                .to(self._device)
                .mean(dim=0)
            )
        if not len(weights) == len(Label):
            raise ValueError(
                f"Length of weights ({len(weights)}) does not match number of labels ({len(Label)})"
            )

        criterion = nn.CrossEntropyLoss(weight=weights).to(self._device)
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        _wandb = log_wandb and wandb.login()
        if _wandb:
            wandb.init(
                project=self.__class__.__name__,
                config={
                    "training size": len(training_inventories),
                    "epochs": epochs,
                    "weights": weights,
                    "shuffle": shuffle,
                    "optimizer": optimizer.__class__.__name__,
                    "criterion": criterion.__class__.__name__,
                    # TODO: convert to nested dict
                    "modules": self.__dict__["_modules"],
                },
            )
        elif log_wandb:
            logging.warning(
                "Weights & Biases not available. Set the WANDB_API_KEY environment variable for logging in."
            )

        for epoch in range(epochs):
            if shuffle:
                random.shuffle(training_inventories)

            for inventory in tqdm(
                training_inventories,
                desc="Training",
                unit="inventory",
                total=len(training_inventories),
            ):
                if len(inventory) < 2:
                    logging.warning(f"Skipping inventory: {inventory}")
                    continue

                optimizer.zero_grad()
                outputs = self(inventory.pages).to(self._device)

                loss = criterion(
                    outputs, inventory.label_tensor().to(self._device)
                ) * math.log(len(inventory))

                loss.backward()
                optimizer.step()

                if _wandb:
                    wandb.log(
                        {
                            "loss": loss.item(),
                            "inventory length": len(inventory),
                            "cache": self._page_embedding._region_model._text_embeddings.cache_info()._asdict(),
                        }
                    )

            if validation_inventories:
                _eval = {"epoch": epoch}
                for metric in self.eval_(validation_inventories, None):
                    if metric.average is None:
                        _eval[metric.__class__.__name__] = {
                            label.name: score
                            for label, score in zip(Label, metric.compute().tolist())
                        }
                    else:
                        _eval[metric.__class__.__name__] = metric.compute().item()

                if _wandb:
                    wandb.log(_eval)
                else:
                    tqdm.write(str(_eval))

                self.train()

        if _wandb:
            wandb.finish()
            # TODO: save model to WandB or HuggingFace

    def eval_(
        self, inventories: list[Inventory], results_out: TextIO = sys.stdout
    ) -> tuple[Metric, Metric, Metric, Metric]:
        """Evaluate the model on the given dataset.

        Args:
            inventories (list[Inventory]): The inventories to evaluate on.
            results_out (TextIO): The file to write the output labels per sample.
                If None (default), does not print the individual test labels.
                Defaults to stdout.
        Returns:
            tuple[Metric, Metric, Metric, Metric]: The precision, recall, F1 score, and accuracy.
        """
        metrics: tuple[Metric] = (
            MulticlassPrecision(average=None, num_classes=len(Label)),
            MulticlassRecall(average=None, num_classes=len(Label)),
            MulticlassF1Score(average=None, num_classes=len(Label)),
            MulticlassAccuracy(),
        )

        if results_out is not None:
            writer = csv.DictWriter(
                results_out,
                fieldnames=("Predicted", "Actual", "Page ID", "Text", "Scores"),
                delimiter="\t",
            )
            writer.writeheader()

        self.eval()

        for inventory in tqdm(
            inventories, desc="Evaluating", total=len(inventories), unit="batch"
        ):
            predicted = self(inventory.pages)
            labels = inventory.labels()

            _labels = torch.Tensor([label.value for label in labels]).to(int)

            for metric in metrics:
                metric.update(predicted, _labels)

            if results_out is not None:
                for page, pred, label in zip(
                    inventory.pages, predicted, labels, strict=True
                ):
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
