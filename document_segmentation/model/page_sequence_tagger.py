import logging
from collections import Counter
from typing import Any, Iterable

import pandas as pd
import torch
from torcheval.metrics import Metric

from ..pagexml.datamodel.inventory import Inventory
from ..pagexml.datamodel.label import Label, SequenceLabel
from ..pagexml.datamodel.page import Page
from .page_learner import AbstractPageLearner


class PageSequenceTagger(AbstractPageLearner):
    """A page sequence tagger that uses an RNN over the regions on a page."""

    _LOSS_REDUCTION = "sum"
    _LABEL_TYPE = SequenceLabel

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
    ) -> tuple[float, pd.DataFrame]:
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
        predicted: list[SequenceLabel] = self._prediction_heuristics(model_output)

        actual: list[SequenceLabel] = inventory.labels()

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
                "Actual": "" if _actual == SequenceLabel.UNK else _actual.name,
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
            page.annotate(SequenceLabel[label])
        return inventory.get_documents()

    @staticmethod
    def _prediction_heuristics(model_output: torch.Tensor) -> list[SequenceLabel]:
        """Convert model output tensor to the predicted labels, using argmax and heuristics.

        Args:
            labels (torch.Tensor): The predicted labels.
        Returns:
            list[Label]: The labels.
        """
        labels: list[SequenceLabel] = [SequenceLabel(model_output[0].argmax().item())]
        # TODO: fix first and last

        for i in range(1, len(model_output) - 1):
            prev: SequenceLabel = labels[-1]
            curr = SequenceLabel(model_output[i].argmax().item())
            next = SequenceLabel(model_output[i + 1].argmax().item())

            correction = None
            if prev == SequenceLabel.OUT and next == SequenceLabel.IN:
                correction: SequenceLabel = SequenceLabel.BOUNDARY

            if correction is not None and curr != correction:
                logging.warning(
                    f"Correcting label from '{curr.name}' to '{correction.name}'."
                )
                curr = correction

            labels.append(curr)
        labels.append(SequenceLabel(model_output[-1, :].argmax().item()))

        return labels

    def _class_counts(self, inventories: Iterable[Inventory]) -> Counter[SequenceLabel]:
        """
        Count the number of labels in the given inventories.

        Returns:
            Counter[Label]: Frequency of each label in dataset.
        """
        return sum((doc.class_counts() for doc in inventories), start=Counter())
