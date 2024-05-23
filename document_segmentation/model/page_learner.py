import abc
import logging
import random
import sys
from collections import Counter
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable, Optional

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

from document_segmentation.pagexml.datamodel.document import Document

from .. import settings
from ..pagexml.datamodel.inventory import (
    Inventory,
    ThumbnailDownloader,
)
from ..pagexml.datamodel.page import Page
from ..settings import (
    LEARNING_RATE,
    MAX_INVENTORY_SIZE,
    PAGE_SEQUENCE_TAGGER_RNN_CONFIG,
    WEIGHT_DECAY,
)
from .device_module import DeviceModule
from .page_embedding import PageEmbedding


class AbstractPageLearner(nn.Module, DeviceModule, abc.ABC):
    """A page sequence tagger that uses an RNN over the regions on a page."""

    _LOSS_REDUCTION = "mean"
    """The reduction method for the loss function.
    See https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    Can be overriden by subclasses."""

    _MULTI_LABEL: bool
    """If True, the model outputs a label per page, otherwise one per document.
    Must be set by subclasses.
    """

    _LABEL_TYPE: type[IntEnum]
    """The type of label used by the model.
    Must be set by the sub-classes"""

    def __init__(
        self,
        *,
        rnn_config: dict[str, Any] = PAGE_SEQUENCE_TAGGER_RNN_CONFIG,
        device: Optional[str] = None,
        thumbnail_downloader: ThumbnailDownloader = ThumbnailDownloader.from_file(),
    ) -> None:
        super().__init__()

        self._thumbnail_downloader = thumbnail_downloader

        self._page_embedding = PageEmbedding(device=device)

        # LSTM, because GRU does not seem not to work on MPS: https://github.com/pytorch/pytorch/issues/94691
        self._rnn = nn.LSTM(
            input_size=self._page_embedding.output_size, batch_first=True, **rnn_config
        )

        self._linear = nn.Linear(
            self._rnn.hidden_size * (self._rnn.bidirectional + 1), len(self._LABEL_TYPE)
        )
        self._softmax = nn.Softmax(dim=int(self._MULTI_LABEL))

        self._eval_args = {"average": None, "num_classes": len(self._LABEL_TYPE)}

        self.to_device(device)

        self._wandb_run: Optional[wandb.run] = None

    @property
    def wandb_run(self) -> Optional[wandb.run]:
        return self._wandb_run

    @wandb_run.setter
    def wandb_run(self, wandb_run: Optional[wandb.run]) -> None:
        self._wandb_run = wandb_run

    def _log_wandb(
        self,
        name: str,
        epoch: int,
        *,
        metrics: tuple[Metric] = None,
        output: pd.DataFrame = None,
        commit: bool = False,
    ) -> dict[str, Any]:
        """Log metrics and/or output to Weights & Biases.

        Metrics and output can be None, meaning they are not logged.

        Args:
            name (str): The name of the dataset.
            epoch (int): The epoch number.
            metrics (tuple[Metric], optional): The metrics to log. Defaults to None.
            output (pd.DataFrame, optional): The output to log. Defaults to None.
            commit (bool, optional): Whether to commit the log, incrementing the global step. Defaults to False.
        """
        if self.wandb_run is None:
            logging.warning("No Weights & Biases run found. Skipping logging.")
        else:
            prefix = name + "/"
            wandb_metrics: dict[str, Any] = {"epoch": epoch}

            for metric in metrics or []:
                if metric.average is not None:
                    score: float = metric.compute().item()
                else:
                    score: dict[str, float] = {
                        label.name: score
                        for label, score in zip(
                            self._LABEL_TYPE, metric.compute().tolist()
                        )
                    }
                wandb_metrics[prefix + metric.__class__.__name__] = score

            if output is not None:
                try:
                    output["Image"] = output["ThumbnailHtml"].dropna().apply(wandb.Html)
                    table = wandb.Table(
                        dataframe=output.drop(
                            columns=["ThumbnailUrl", "ThumbnailHtml", "Link"]
                        )
                    )
                except KeyError:
                    logging.error("Error transforming output to WandB table: {e}")
                    table = wandb.Table(dataframe=output)

                wandb_metrics[prefix + "pages"] = table

        return self.wandb_run.log(wandb_metrics, commit=commit)

    def forward(self, pages: list[Page]) -> torch.Tensor:
        page_embeddings = self._page_embedding(pages)

        assert page_embeddings.size() == (
            len(pages),
            self._page_embedding.output_size,
        ), "Bad shape: {pages.size()}"

        rnn_out, hidden = self._rnn(page_embeddings)
        states = rnn_out if self._MULTI_LABEL else rnn_out[-1]
        output = self._linear(states)
        softmax = self._softmax(output)
        return softmax

    def _wandb_config(self, training_data, validation_data) -> dict[str, Any]:
        """Class-specific configuration for Weights & Biases.

        Can be overriden by sub-classes.

        Args:
            training_data: The training data.
            validation_data: The validation data.
        Returns:
            dict[str, Any]: The configuration dictionary for merging with general WandB configuration.
        """
        return {}

    def train_(
        self,
        training_data: list[Inventory | Document],
        validation_data=None,
        *,
        epochs: int = 3,
        weights: list[float] = None,
        shuffle: bool = True,
        log_wandb: bool = True,
    ) -> dict[str, Any]:
        """Train the model on the given dataset.

        Args:
            training_inventories (list[Inventory | Document]): The dataset to train on.
            validation_inventories (Optional[Any], optional): Validation datasets as expected by the specific class.
                Defaults to None. If given, will evaluate each dataset separately after each epoch.
            epochs (int, optional): The number of epochs to train. Defaults to 3.
            weights (list[float]): The weights for the loss function.
                If None (default), the class weights are computed from the training dataset.
            shuffle (bool, optional): Whether to shuffle the dataset for each epoch. Defaults to True.
            log_wandb (bool, optional): Whether to log the training to Weights & Biases. Defaults to True.

        Returns:
            dict[str, Any]: The state dictionary of the model; the best model if validation data is given, otherwise the last state.
        """
        self.train()

        if weights is None:
            weights = torch.Tensor(self._total_class_weights(training_data)).to(
                self._device
            )
        if not len(weights) == len(self._LABEL_TYPE):
            raise ValueError(
                f"Length of weights ({len(weights)}) does not match number of labels ({len(self._LABEL_TYPE)})"
            )

        criterion = nn.CrossEntropyLoss(
            reduction=self._LOSS_REDUCTION,
            weight=torch.Tensor(weights).to(self._device),
        ).to(self._device)
        optimizer = optim.Adam(
            self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        if log_wandb:
            if self.wandb_run is None:
                self.wandb_run = wandb.init(
                    project=self.__class__.__name__,
                    config=self._wandb_config(training_data, validation_data)
                    | {
                        "epochs": epochs,
                        "weights": {
                            label.name: weight
                            for label, weight in zip(self._LABEL_TYPE, weights)
                        },
                        "shuffle": shuffle,
                        "optimizer": optimizer.__class__.__name__,
                        "criterion": criterion.__class__.__name__,
                        "criteron_config": {
                            key: value
                            for key, value in criterion.__dict__.items()
                            if not key.startswith("_")
                        },
                        # TODO: convert to nested dict
                        "modules": self.__dict__["_modules"],
                        "settings": settings.as_dict(),
                    },
                )
                if self.wandb_run is None:
                    logging.warning(
                        "Failed to initialize Weights & Biases. Set the WANDB_API_KEY environment variable for logging in."
                    )
            else:
                logging.info(
                    f"Resuming logging on Weights & Biases run: {self.wandb_run}."
                )
        else:
            self.wandb_run = None

        best_score = 0
        best_model = None

        for epoch in range(1, epochs + 1):
            self.train()

            if shuffle:
                random.shuffle(training_data)

            for doc in tqdm(
                training_data, desc="Training", unit="docs", total=len(training_data)
            ):
                if isinstance(doc, Inventory):
                    # TODO: move to class-specific filter method
                    if len(doc) < 2:
                        logging.warning(f"Skipping single page inventory: {doc}")
                        continue
                    elif MAX_INVENTORY_SIZE and (len(doc) > MAX_INVENTORY_SIZE):
                        logging.error(
                            f"Inventory '{doc}' larger than {MAX_INVENTORY_SIZE} pages."
                        )

                optimizer.zero_grad()
                outputs = self(doc.pages).to(self._device)

                loss = criterion(outputs, doc.label_tensor().to(self._device))
                if torch.isnan(loss):
                    logging.error(f"Loss is NaN for inventory: '{doc}'")

                loss.backward()
                optimizer.step()

                if self.wandb_run:
                    if not settings.UPDATE_LM_WEIGHTS:
                        self.wandb_run.log(
                            {
                                "cache": self._page_embedding._region_model._text_embeddings_cached.cache_info()._asdict()
                            },
                            commit=False,
                        )
                    if isinstance(doc, Inventory):
                        # TODO: move logging to class-specific method
                        self.wandb_run.log({"inventory": doc.inv_nr}, commit=False)

                    self.wandb_run.log(
                        {
                            "loss": loss.item(),
                            "document length": len(doc),
                        }
                    )

            if validation_data:
                score, validation_results = self._validate(validation_data, epoch=epoch)

                if score > best_score:
                    logging.info(
                        f"New best model found with average F1 score: {score:.4f} (previous: {best_score:.4f})."
                    )
                    best_score = score
                    best_model = self.state_dict()

                    # log result tables per sheet for new best model
                    for sheet_name, output in validation_results.items():
                        self._log_wandb(sheet_name, epoch, output=output)
            else:
                best_model = self.state_dict()

        return best_model or self.state_dict()

    def eval_(
        self, docs: list[Any], name: str
    ) -> tuple[Metric, Metric, Metric, Metric, pd.DataFrame]:
        """Evaluate the model on the given dataset.

        Args:
            docs (list[Any]): The data to evaluate.
            name (str): The name to use for the evaluation logs.
        Returns:
            tuple[Metric, Metric, Metric, Metric, pd.DataFrame]: The precision, recall, F1 score and accuracy metrics,
                and a DataFrame containing the results per row.
        """
        metrics: tuple[Metric] = (
            MulticlassPrecision(average=None, num_classes=len(self._LABEL_TYPE)),
            MulticlassRecall(average=None, num_classes=len(self._LABEL_TYPE)),
            MulticlassF1Score(average=None, num_classes=len(self._LABEL_TYPE)),
            MulticlassAccuracy(),
        )
        self.eval()

        output: pd.DataFrame = pd.concat(
            (
                self.predict(doc, *metrics)
                for doc in tqdm(
                    docs, desc=f"Evaluating '{name}'", total=len(docs), unit="doc"
                )
            )
        ).reset_index()

        return metrics + (output,)

    @abc.abstractmethod
    @torch.inference_mode()
    def _validate(self, validation_data, *, epoch: int) -> tuple[float, pd.DataFrame]:
        """Get model predictions for all pages in the given inventory.
        Metrics are updated in-place.

        Args:
            validation_data: The data to infer label(s) for.
            metrics (Metric): The metrics to update.
        Returns:
            pd.DataFrame: A DataFrame containing the results per row.
        """
        return NotImplemented

    @abc.abstractmethod
    @torch.inference_mode()
    def predict(self, data, *metrics: Metric) -> pd.DataFrame:
        return NotImplemented

    def load(self, path: Path) -> None:
        """Load the model from the given path.

        Args:
            path (str): The path to load the model from.
        """
        self.load_state_dict(torch.load(path, map_location=torch.device(self._device)))

    def _prediction_heuristics(self, model_output: torch.Tensor) -> list[IntEnum]:
        """Convert model output tensor to the predicted labels, using argmax and heuristics.

        Args:
            labels (torch.Tensor): The predicted labels.
        Returns:
            list[IntEnum]: The labels
        """
        return [self._LABEL_TYPE(arg) for arg in model_output.argmax(dim=1).tolist()]

    @abc.abstractmethod
    def _class_counts(self, docs: Iterable[Inventory | Document]) -> Counter:
        """Get the frequency of each label in this dataset.

        To be implemented by subclasses.

        Returns:
            Counter: Counter of frequency of each label in dataset.
        """
        return NotImplemented

    def _total_class_weights(self, docs: Iterable[Inventory | Document]) -> list[float]:
        """Get the inverse frequency of each label in this dataset.

        Applies add-one smoothing to avoid division by zero.

        Returns:
            list[float]: List of frequency of each label in dataset divided by dataset length.
        """
        counts: Counter[self._LABEL_TYPE] = self._class_counts(docs)

        try:
            total = counts.total()
        except AttributeError as e:
            logging.warning(
                f"Python version: '{sys.version}': {str(e)}. Using sum(counts.values()) instead."
            )
            total = sum(counts.values())

        inverse_frequencies: list[float] = [
            total / (counts[label] + 1) for label in self._LABEL_TYPE
        ]
        inverse_frequencies[self._LABEL_TYPE.UNK] = 0.0

        return inverse_frequencies
