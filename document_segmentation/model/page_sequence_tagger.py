import logging
from collections import Counter
from typing import Iterable

import pandas as pd
import torch
from torcheval.metrics import Metric

from ..pagexml.datamodel.inventory import Inventory
from ..pagexml.datamodel.label import Label, SequenceLabel
from ..pagexml.datamodel.page import Page
from ..settings import MAX_INVENTORY_SIZE
from .page_learner import AbstractPageLearner


class PageSequenceTagger(AbstractPageLearner):
    """A page sequence tagger that uses an RNN over the regions on a page."""

    _LOSS_REDUCTION = "sum"
    _LABEL_TYPE = SequenceLabel
    _MULTI_LABEL = True

    def _filter_training_data(self, inventory: Inventory) -> bool:
        """Filter out training data samples.

        Filters out short inventories. Issues error on large inventories.

        Args:
            inventory (Inventory): The inventory to filter.
        Returns:
            bool: True if inventory should be skipped.
        """
        if MAX_INVENTORY_SIZE and (len(inventory) > MAX_INVENTORY_SIZE):
            logging.error(
                f"Inventory '{inventory}' larger than {MAX_INVENTORY_SIZE} pages."
            )
        return len(inventory) > 1

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

        return pd.DataFrame(
            [
                inventory.output_row(
                    _prediction, _actual, page, output_row, self._thumbnail_downloader
                )
                for page, _prediction, output_row, _actual in zip(
                    inventory.pages, predicted, model_output, actual, strict=True
                )
            ]
        )

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

    @staticmethod
    def _class_counts(inventories: Iterable[Inventory]) -> Counter[SequenceLabel]:
        """
        Count the number of labels in the given inventories.

        Returns:
            Counter[Label]: Frequency of each label in dataset.
        """
        return sum((doc.class_counts() for doc in inventories), start=Counter())
