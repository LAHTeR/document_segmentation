import logging
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer

from ..pagexml.datamodel import Region, RegionType
from ..settings import LANGUAGE_MODEL
from .device_module import DeviceModule


class RegionEmbedding(nn.Module, DeviceModule):
    def __init__(
        self,
        hidden_size: int = 64,
        transformer_model_name: str = LANGUAGE_MODEL,
        line_separator: str = "\n",
        device: Optional[str] = None,
    ):
        super().__init__()

        self._line_separator = line_separator

        # Text embeddings from a Transformers model
        self.transformer_model: AutoModel = AutoModel.from_pretrained(
            transformer_model_name
        )
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            transformer_model_name
        )

        self.embedding_size = self.transformer_model.config.hidden_size
        self._max_length = self.transformer_model.config.max_position_embeddings
        self._linear = nn.Linear(hidden_size * 2, 128)

        self.to_device(device)

    @lru_cache(maxsize=256)
    @torch.no_grad()
    def _text_embeddings(self, region_batch: tuple[Region, ...]) -> torch.Tensor:
        """Embed the text of a page using a Transformers model.

        Args:
            region_batch: The regions to embed.
                If empty, return a zero tensor of embeddings dimensionality
        Returns:
            A tensor of shape (1, embedding_size).
        """
        # TODO: process input batches

        if region_batch:
            region_texts: list[str] = [
                self._line_separator.join(region.lines) for region in region_batch
            ]

            text_inputs = self.tokenizer(
                region_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self._max_length,
            )

            out = self.transformer_model(
                **text_inputs.to(self._device)
            ).last_hidden_state
            cls_tokens = out[:, 0, :]  # CLS token is first token of sequence
        else:
            logging.debug("Empty region batch.")
            cls_tokens = torch.zeros(0, self.embedding_size)

        expected_size = (len(region_batch), self.embedding_size)
        assert (
            cls_tokens.size() == expected_size
        ), f"Output shape was {cls_tokens.size()}, but should be {expected_size}."

        return cls_tokens

    def _region_types_tensor(self, region_batch: list[Region]) -> torch.Tensor:
        # FIXME: populate the types tensor more efficiently
        types = torch.zeros(len(region_batch), len(RegionType)).to(self._device)
        for i, region in enumerate(region_batch):
            for region_type in region.types:
                types[i, region_type.index()] = 1

        assert types.size() == (
            len(region_batch),
            len(RegionType),
        ), f"Output shape: {types.size()}."

        return types

    # TODO
    def _coordinates_tensor(self, page) -> torch.Tensor:
        """Convert coordinates to a PyTorch tensor.

        Args:
            coordinates: The coordinates to convert.
        """
        coordinates: np.ndarray = np.zeros(
            (len(page.get_text_regions_in_reading_order()), self._max_length, 2)
        )

        for i, region in enumerate(page.get_text_regions_in_reading_order()):
            points = np.array(region.coords.points)  # array of shape (n_points, 2)
            coordinates[i, : points.shape[0]] = points

        scaler = MinMaxScaler()  # TODO: use sklearn.preprocessing.minmax_scale
        return torch.from_numpy(
            scaler.fit_transform(
                coordinates.reshape(-1, coordinates.shape[-1])
            ).reshape(coordinates.shape)
        )

    def forward(self, regions: list[Region]):
        """Embed a sequence of regions.

        Args:
            regions (list[Region]): The regions to embed. This must be a tuple for caching.
        """
        if not isinstance(regions, list):
            logging.warning(
                f"Converting regions into a tuple (was: '{type(regions)}')."
            )
            regions = list(regions)

        text_embeddings = self._text_embeddings(tuple(regions)).float()
        region_types = self._region_types_tensor(regions).float()
        # TODO region_coordinates = self._coordinates_tensor(region).float()

        # Combine input features
        region_inputs = torch.cat(
            [
                text_embeddings,
                region_types,
                # region_coordinates
            ],
            dim=-1,
        )

        assert region_inputs.size() == (
            len(regions),
            self.embedding_size + len(RegionType),
        ), f"Bad output shape: {region_inputs.size()}."

        return region_inputs
