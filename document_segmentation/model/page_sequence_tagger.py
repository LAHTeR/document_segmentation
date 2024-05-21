import logging
from typing import Any, Optional

import pandas as pd
import torch
import wandb
from torcheval.metrics import Metric

from ..pagexml.datamodel.inventory import (
    Inventory,
    ThumbnailDownloader,
)
from ..pagexml.datamodel.label import Label
from ..pagexml.datamodel.page import Page
from ..settings import PAGE_SEQUENCE_TAGGER_RNN_CONFIG
from .page_learner import AbstractPageLearner


class PageSequenceTagger(AbstractPageLearner):
    """A page sequence tagger that uses an RNN over the regions on a page."""

    def __init__(
        self,
        *,
        rnn_config: dict[str, Any] = PAGE_SEQUENCE_TAGGER_RNN_CONFIG,
        device: Optional[str] = None,
        thumbnail_downloader: ThumbnailDownloader = ThumbnailDownloader.from_file(),
    ) -> None:
        super().__init__(
            label_type=Label,
            rnn_config=rnn_config,
            device=device,
            thumbnail_downloader=thumbnail_downloader,
        )

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

    def _wand_config(
        self,
        training_inventories: list[Inventory],
        validation_inventories: dict[str, list[Inventory]],
    ) -> dict[str, Any]:
        return {
            "training size": {
                "inventories": len(training_inventories),
                "pages": sum(len(inv) for inv in training_inventories),
            },
            "validation sets": {
                key: len(invs) for key, invs in (validation_inventories or {}).items()
            },
        }

    def _validate(
        self, validation_inventories: dict[str, list[Inventory]], *, epoch: int
    ) -> float:
        results: dict[str, pd.DataFrame] = {}
        for sheet_name, _validation in validation_inventories.items():
            if _validation:
                precision, recall, f1, accuracy, output = self.eval_(
                    _validation, sheet_name
                )
                # log metrics per sheet for each epoch
                self._log_wandb(
                    sheet_name, epoch, metrics=(precision, recall, f1, accuracy)
                )
                results[sheet_name] = output
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
        else:
            logging.warning("All validation sets are empty.")
            score = 0
        return score, results

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

    def predict_documents(self, inventory: Inventory) -> list[list[Page]]:
        """Predict page labels and extract the documents from the inventory.

        Args:
            inventory (Inventory): The inventory to get the documents from.
        Returns:
            list[list[Page]]: A list of documents, each represented as a list of pages.
        """
        predictions: pd.DataFrame = self.predict(inventory)
        for page, label in zip(inventory.pages, predictions["Predicted"]):
            page.annotate(Label[label])
        return inventory.get_documents()

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
                logging.warning(
                    f"Correcting label from '{curr.name}' to '{correction.name}'."
                )
                curr = correction

            labels.append(curr)
        labels.append(Label(model_output[-1, :].argmax().item()))

        return labels
