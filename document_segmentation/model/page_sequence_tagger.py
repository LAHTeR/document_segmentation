import logging
import math
import random
from typing import Any, Optional

import pandas as pd
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

from document_segmentation.pagexml.datamodel.inventory import (
    Inventory,
    ThumbnailDownloader,
)
from document_segmentation.pagexml.datamodel.page import Page

from ..pagexml.datamodel.label import Label
from ..settings import PAGE_SEQUENCE_TAGGER_RNN_CONFIG
from .device_module import DeviceModule
from .page_embedding import PageEmbedding


class PageSequenceTagger(nn.Module, DeviceModule):
    """A page sequence tagger that uses an RNN over the regions on a page."""

    _LARGE_INVENTORY_SIZE: int = 1751
    """Issue a warning for inventories larger than this size."""

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
        validation_inventories: Optional[dict[str, list[Inventory]]] = None,
        *,
        epochs: int = 3,
        weights: Optional[list[float]] = None,
        shuffle: bool = True,
        log_wandb: bool = True,
        thumbnail_downloader: Optional[ThumbnailDownloader] = None,
    ):
        """Train the model on the given dataset.

        Args:
            training_inventories (list[Inventory]): The dataset to train on.
            validation_inventories (Optional[dict[str[list[Inventory]]], optional): Validation datasets with name.
                Defaults to None. If given, will evaluate each dataset separately after each epoch.
            epochs (int, optional): The number of epochs to train. Defaults to 3.
            weights (Optional[list[float]], optional): The weights for the loss function. Defaults to None.
            shuffle (bool, optional): Whether to shuffle the dataset for each epoch. Defaults to True.
            log_wandb (bool, optional): Whether to log the training to Weights & Biases. Defaults to True.
            thumbnail_downloader (Optional[ThumbnailDownloader], optional): The thumbnail downloader to use.
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

        if log_wandb and wandb.login():
            wandb_run = wandb.init(
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
        else:
            wandb_run = None
            logging.warning(
                "Not logging to Weights & Biases. Set the WANDB_API_KEY environment variable for logging in."
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
                elif len(inventory) > PageSequenceTagger._LARGE_INVENTORY_SIZE:
                    # FIXME: introduce max_size parameter and split large inventories if necessary
                    logging.warning(f"Large inventory: {inventory}")

                optimizer.zero_grad()
                outputs = self(inventory.pages).to(self._device)

                loss = criterion(
                    outputs, inventory.label_tensor().to(self._device)
                ) * math.log(len(inventory))

                loss.backward()
                optimizer.step()

                if wandb_run:
                    wandb_run.log(
                        {
                            "loss": loss.item(),
                            "inventory length": len(inventory),
                            "cache": self._page_embedding._region_model._text_embeddings.cache_info()._asdict(),
                        }
                    )

            if validation_inventories and wandb_run:
                _eval = {"epoch": epoch}

                for name, inventories in validation_inventories.items():
                    dataset_eval = {}

                    results = self.eval_(inventories, thumbnail_downloader)
                    metrics = results[:4]
                    table = results[4]

                    wandb_run.log({name + "_results": wandb.Table(dataframe=table)})

                    for metric in metrics:
                        if metric.average is None:
                            dataset_eval[metric.__class__.__name__] = {
                                label.name: score
                                for label, score in zip(
                                    Label, metric.compute().tolist()
                                )
                            }
                        else:
                            dataset_eval[metric.__class__.__name__] = (
                                metric.compute().item()
                            )

                    _eval[name] = dataset_eval

                wandb_run.log(_eval)
                self.train()

        if wandb_run:
            wandb.finish()
            # TODO: save model to WandB or HuggingFace

    def eval_(
        self,
        inventories: list[Inventory],
        thumbnail_downloader: Optional[ThumbnailDownloader] = None,
    ) -> tuple[Metric, Metric, Metric, Metric, pd.DataFrame]:
        """Evaluate the model on the given dataset.

        Args:
            inventories (list[Inventory]): The inventories to evaluate on.
            thumbnail_downloader (Optional[ThumbnailDownloader], optional): The thumbnail downloader to use.
        Returns:
            tuple[Metric, Metric, Metric, Metric, pd.DataFrame]: The precision, recall, F1 score and accuracy metrics,
                and a DataFrame containing the results per row.
        """
        metrics: tuple[Metric] = (
            MulticlassPrecision(average=None, num_classes=len(Label)),
            MulticlassRecall(average=None, num_classes=len(Label)),
            MulticlassF1Score(average=None, num_classes=len(Label)),
            MulticlassAccuracy(),
        )

        self.eval()

        results: list[list[Any]] = []

        for inventory in tqdm(
            inventories, desc="Evaluating", total=len(inventories), unit="inventory"
        ):
            predicted = self(inventory.pages)
            labels = inventory.labels()

            _labels = torch.Tensor([label.value for label in labels]).to(int)

            for metric in metrics:
                metric.update(predicted, _labels)

            for page, pred, label in zip(
                inventory.pages, predicted, labels, strict=True
            ):
                row = [
                    Label(pred.argmax().item()).name,
                    label.name,
                    page.doc_id,
                    page.text(delimiter="; ")[:50],
                    str(pred.tolist()),
                ]
                if thumbnail_downloader:
                    thumbnail_url = thumbnail_downloader.thumbnail_url(inventory, page)
                    link: str = inventory.link(page)
                    row.append(
                        wandb.Html(
                            f"<a href='{link}'><img src='{thumbnail_url}' alt='{thumbnail_url}'/></a>"
                        )
                    )

                results.append(row)

        columns = ["Predicted", "Actual", "Page ID", "Text", "Scores"]
        if thumbnail_downloader:
            columns.append("Image")

        return metrics + (pd.DataFrame(results, columns=columns),)
