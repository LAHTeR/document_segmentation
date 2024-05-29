from typing import List, Optional

import torch
from torch import nn, optim
from tqdm import tqdm

from ..pagexml.datamodel.label import SequenceLabel
from ..pagexml.datamodel.region import Region
from ..settings import LANGUAGE_MODEL, REGION_TYPE_EMBEDDING_SIZE
from .dataset import RegionDataset
from .device_module import DeviceModule
from .region_embedding import RegionEmbeddingSentenceTransformer


class RegionClassifier(nn.Module, DeviceModule):
    def __init__(
        self,
        *,
        transformer_model_name: str = LANGUAGE_MODEL,
        region_type_embedding_size: int = REGION_TYPE_EMBEDDING_SIZE,
        line_separator: str = "\n",
        device: Optional[str] = None,
    ):
        super().__init__()

        # TODO this can be a RegionEmbedding or RegionEmbeddingSentenceTransformer
        self._region_embedding = RegionEmbeddingSentenceTransformer(
            transformer_model_name=transformer_model_name,
            region_type_embedding_size=region_type_embedding_size,
            line_separator=line_separator,
            device=device,
        )

        self._linear = nn.Linear(
            in_features=self.text_embedding_size + region_type_embedding_size,
            out_features=len(SequenceLabel),
        )

        self._softmax = nn.Softmax(dim=1)
        self.to_device(device)

    def forward(self, regions: List[Region]) -> torch.Tensor:
        region_embeddings = self._region_embedding(regions)
        return self._softmax(self._linear(region_embeddings))

    def train_(
        self,
        dataset: RegionDataset,
        epochs: int = 3,
        batch_size: int = 32,
        weights: list[float] = None,
    ):
        """Train the model on a dataset.

        Args:
            dataset (RegionDataset): The dataset to train on.
            epochs (int): The number of epochs to train for.
            batch_size (int): The batch size -- refers to the number of pages in a batch, and uses all of their regions.
            weights (list[float]): The weights for each class.
        """
        self.train()

        if weights is None:
            weights = dataset.class_weights()
        if len(weights) != len(SequenceLabel):
            raise ValueError(
                f"Expected {len(SequenceLabel)} weights, got {len(weights)}."
            )

        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights)).to(self._device)
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for _ in range(epochs):
            for batch in tqdm(
                dataset.batches(batch_size),
                unit="batch",
                total=len(dataset) / batch_size,
            ):
                optimizer.zero_grad()
                outputs = self(batch.regions()).to(self._device)
                labels = batch.label_tensor()

                loss = criterion(outputs, labels.to(self._device)).to(self._device)

                loss.backward()
                optimizer.step()
                # TODO: stop training when loss does not decrease anymore
            tqdm.write(f"[Loss:\t{loss:.3f}]")
