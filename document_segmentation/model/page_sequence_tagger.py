import logging
import random
from pathlib import Path
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

from .. import settings
from ..pagexml.datamodel.inventory import (
    Inventory,
    ThumbnailDownloader,
)
from ..pagexml.datamodel.label import Label
from ..pagexml.datamodel.page import Page
from ..settings import (
    LEARNING_RATE,
    MAX_INVENTORY_SIZE,
    PAGE_SEQUENCE_TAGGER_RNN_CONFIG,
    WEIGHT_DECAY,
)
from .device_module import DeviceModule
from .page_embedding import PageEmbedding


class PageSequenceTagger(nn.Module, DeviceModule):
    """A page sequence tagger that uses an RNN over the regions on a page."""

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
            self._rnn.hidden_size * (self._rnn.bidirectional + 1), len(Label)
        )
        self._softmax = nn.Softmax(dim=1)

        self._eval_args = {"average": None, "num_classes": len(Label)}

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
                        for label, score in zip(Label, metric.compute().tolist())
                    }
                wandb_metrics[prefix + metric.__class__.__name__] = score

            if output is not None:
                output["Image"] = output["ThumbnailHtml"].dropna().apply(wandb.Html)
                table = wandb.Table(
                    dataframe=output.drop(
                        columns=["ThumbnailUrl", "ThumbnailHtml", "Link"]
                    )
                )
                wandb_metrics[prefix + "pages"] = table

        return self.wandb_run.log(wandb_metrics, commit=commit)

    def forward(self, pages: list[Page]) -> torch.Tensor:
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
        weights: list[float] = None,
        shuffle: bool = True,
        log_wandb: bool = True,
    ) -> dict[str, Any]:
        """Train the model on the given dataset.

        Args:
            training_inventories (list[Inventory]): The dataset to train on.
            validation_inventories (Optional[dict[str[list[Inventory]]], optional): Validation datasets with name.
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
            weights = torch.Tensor(
                Inventory.total_class_weights(training_inventories)
            ).to(self._device)
        if not len(weights) == len(Label):
            raise ValueError(
                f"Length of weights ({len(weights)}) does not match number of labels ({len(Label)})"
            )

        criterion = nn.CrossEntropyLoss(
            reduction="sum", weight=torch.Tensor(weights).to(self._device)
        ).to(self._device)
        optimizer = optim.Adam(
            self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        if log_wandb:
            if self.wandb_run is None:
                self.wandb_run = wandb.init(
                    project=self.__class__.__name__,
                    config={
                        "training size": {
                            "inventories": len(training_inventories),
                            "pages": sum(len(inv) for inv in training_inventories),
                        },
                        "validation sets": {
                            key: len(invs)
                            for key, invs in (validation_inventories or {}).items()
                        },
                        "epochs": epochs,
                        "weights": {
                            label.name: weight for label, weight in zip(Label, weights)
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
                random.shuffle(training_inventories)

            for inventory in tqdm(
                training_inventories,
                desc="Training",
                unit="inventory",
                total=len(training_inventories),
            ):
                if len(inventory) < 2:
                    logging.warning(f"Skipping single page inventory: {inventory}")
                    continue
                elif MAX_INVENTORY_SIZE and (len(inventory) > MAX_INVENTORY_SIZE):
                    logging.error(
                        f"Inventory '{inventory}' larger than {MAX_INVENTORY_SIZE} pages."
                    )

                optimizer.zero_grad()
                outputs = self(inventory.pages).to(self._device)

                loss = criterion(outputs, inventory.label_tensor().to(self._device))
                if torch.isnan(loss):
                    logging.error(f"Loss is NaN for inventory: '{inventory}'")

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
                    self.wandb_run.log(
                        {
                            "loss": loss.item(),
                            "inventory length": len(inventory),
                            "inventory": inventory.inv_nr,
                        }
                    )

            if validation_inventories:
                validation_results: dict[str, pd.DataFrame] = {}
                for sheet_name, _validation in validation_inventories.items():
                    if _validation:
                        precision, recall, f1, accuracy, output = self.eval_(
                            _validation, sheet_name
                        )
                        # log metrics per sheet for each epoch
                        self._log_wandb(
                            sheet_name, epoch, metrics=(precision, recall, f1, accuracy)
                        )
                        validation_results[sheet_name] = output
                    else:
                        logging.warning(f"Empty validation set for '{sheet_name}'.")

                # FIXME: re-running the evaluation on all inventories is redundant
                all_validation_inventories = [
                    inventory
                    for values in validation_inventories.values()
                    for inventory in values
                ]
                if all_validation_inventories:
                    sheet_name = "total"
                    precision, recall, f1, accuracy, output = self.eval_(
                        all_validation_inventories, sheet_name
                    )
                    # Log total metrics
                    self._log_wandb(
                        sheet_name, epoch, metrics=(precision, recall, f1, accuracy)
                    )

                    score = f1.compute().mean().item()
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
                    logging.warning("All validation sets are empty.")
        return best_model or self.state_dict()

    def eval_(
        self, inventories: list[Inventory], sheet_name: str
    ) -> tuple[Metric, Metric, Metric, Metric, pd.DataFrame]:
        """Evaluate the model on the given dataset.

        Args:
            inventories (list[Inventory]): The inventories to evaluate.
            sheet_name (str): The name to use for the evaluation logs.
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

        output: pd.DataFrame = pd.concat(
            (
                self.predict(inventory, *metrics)
                for inventory in tqdm(
                    inventories,
                    desc=f"Evaluating '{sheet_name}'",
                    total=len(inventories),
                    unit="inventory",
                )
            )
        ).reset_index()

        return metrics + (output,)

    @torch.inference_mode()
    def predict(self, inventory: Inventory, *metrics: Metric) -> pd.DataFrame:
        """Get model predictions for all pages in the given inventory.
        Metrics are updated in-place.

        Args:
            inventory (Inventory): The inventory to infer on.
            metrics (Metric): The metrics to update.
        Returns:
            pd.DataFrame: A DataFrame containing the results per row.
        """
        model_output: torch.Tensor = self(inventory.pages)
        predicted: list[Label] = PageSequenceTagger._prediction_heuristics(model_output)

        actual: list[Label] = inventory.labels()

        for metric in metrics:
            metric.update(Label.to_tensor(predicted), Label.to_tensor(actual))

        rows: list[dict[str, str]] = []

        output_chars: int = 100
        for page, _prediction, output_row, _actual in zip(
            inventory.pages, predicted, model_output, actual, strict=True
        ):
            row = {
                "Inventory": inventory.full_inv_nr(),
                "Predicted": _prediction.name,
                "Actual": "" if _actual == Label.UNK else _actual.name,
                "Page ID": page.doc_id,
                f"Text (first {output_chars} characters)": page.text(delimiter="; ")[
                    :output_chars
                ],
                "Scores": str(output_row.tolist()),
            }
            if self._thumbnail_downloader and page.doc_id:
                thumbnail_url = self._thumbnail_downloader.thumbnail_url(
                    inventory, page
                )
                link: str = inventory.link(page)

                row["ThumbnailHtml"] = (
                    f"<a href='{link}'><img src='{thumbnail_url}' alt='{thumbnail_url}'/></a>"
                )
                row["ThumbnailUrl"] = thumbnail_url
                row["Link"] = link

            rows.append(row)

        assert len(rows) == len(
            inventory.pages
        ), f"Expected {len(inventory.pages)} rows, got {len(rows)}."

        return pd.DataFrame(rows)

    @staticmethod
    def _prediction_heuristics(model_output: torch.Tensor) -> list[Label]:
        """Convert model output tensor to the predicted labels, using argmax and heuristics.

        Args:
            labels (torch.Tensor): The predicted labels.
        Returns:
            list[Label]: The labels.
        """
        labels: list[Label] = [Label(model_output[0].argmax().item())]
        # TODO: fix first and last

        for i in range(1, len(model_output) - 1):
            prev: Label = labels[-1]
            curr = Label(model_output[i].argmax().item())
            next = Label(model_output[i + 1].argmax().item())

            correction = None
            if prev == Label.OUT and next == Label.IN:
                correction: Label = Label.BOUNDARY

            if correction is not None and curr != correction:
                logging.info(
                    f"Correcting label from '{curr.name}' to '{correction.name}'."
                )
                curr = correction

            labels.append(curr)
        labels.append(Label(model_output[-1, :].argmax().item()))

        return labels

    def save(self, path: Path) -> None:
        """Save the model to the given path.

        Args:
            path (str): The path to save the model to.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: Path) -> None:
        """Load the model from the given path.

        Args:
            path (str): The path to load the model from.
        """
        self.load_state_dict(torch.load(path, map_location=torch.device(self._device)))
