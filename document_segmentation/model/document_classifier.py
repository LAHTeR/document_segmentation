import logging
from collections import Counter
from typing import Iterable

import pandas as pd
import torch
from torcheval.metrics import Metric

from ..pagexml.datamodel.document import Document
from ..pagexml.datamodel.label import Tanap
from .page_learner import AbstractPageLearner


class DocumentClassifier(AbstractPageLearner):
    _MULTI_LABEL: bool = False
    _LABEL_TYPE = Tanap

    def _validate(
        self, validation_docs: list[Document], *, epoch: int
    ) -> tuple[float, pd.DataFrame]:
        results: dict[str, pd.DataFrame] = {}

        if validation_docs:
            sheet_name = "total"
            precision, recall, f1, accuracy, output = self.eval_(
                validation_docs, sheet_name
            )
            results[sheet_name] = output

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
    def predict(self, document: Document, *metrics: Metric) -> pd.DataFrame:
        """Get model predictions for all pages in the given document.
        Metrics are updated in-place.

        Args:
            document (Document): The document to infer on.
            metrics (Metric): Metrics to calculate.

        Returns:
            pd.DataFrame: The model predictions.
        """
        model_output: torch.Tensor = self(document.pages)
        predicted: Tanap = Tanap(model_output.argmax().item())
        actual: Tanap = document.label

        # TODO: add thumbnails etc
        for metric in metrics:
            metric.update(
                torch.tensor([predicted]).to(torch.int64).to(self._device),
                torch.tensor([actual]).to(torch.int64).to(self._device),
            )

        return pd.DataFrame(
            data=[
                (
                    predicted.name,
                    document.label.name,
                    document.pages[0].external_ref,
                    document.pages[-1].external_ref,
                )
            ],
            columns=["Predicted", "Actual", "First Page", "Last Page"],
        )

    def _class_counts(self, docs: Iterable[Document]) -> Counter[Tanap]:
        """Count the number of labels in the given documents.

        Returns:
            Counter[Tanap]: Frequency of each label in the dataset.
        """
        return Counter(doc.label for doc in docs)
