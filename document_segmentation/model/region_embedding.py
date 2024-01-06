import logging
from functools import lru_cache

import numpy as np
import torch
from pagexml.model.physical_document_model import PageXMLTextRegion
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer

from ..settings import LANGUAGE_MODEL, REGION_TYPES


class RegionEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int = 64,
        transformer_model_name: str = LANGUAGE_MODEL,
        line_separator: str = "\n",
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

    @lru_cache(maxsize=256)
    @torch.no_grad()
    def _text_embeddings(self, region_batch: tuple[PageXMLTextRegion]) -> torch.Tensor:
        """Embed the text of a page using a Transformers model.

        Returns:
            A tensor of shape (1, embedding_size).
        """
        # TODO: process input batches

        region_texts: list[str] = [
            self._line_separator.join(
                [line.text or str() for line in region.get_lines()]
            )
            for region in region_batch
        ]

        text_inputs = self.tokenizer(
            region_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self._max_length,
        )

        out = self.transformer_model(**text_inputs).last_hidden_state
        cls_tokens = out[:, 0, :]  # CLS token is first token of sequence
        assert cls_tokens.size() == (
            len(region_batch),
            self.embedding_size,
        ), f"Output shape: {cls_tokens.size()}."

        return cls_tokens

    def _region_types_tensor(
        self, region_batch: list[PageXMLTextRegion]
    ) -> torch.Tensor:
        types = torch.zeros(len(region_batch), len(REGION_TYPES))

        for i, region in enumerate(region_batch):
            for region_type in region.type:
                types[i, REGION_TYPES.index(region_type)] = 1

        assert types.size() == (
            len(region_batch),
            len(REGION_TYPES),
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

    def forward(self, regions: list[PageXMLTextRegion]):
        """Embed a sequence of regions.

        Args:
            regions (list[PageXMLTextRegion]): The regions to embed. This must be a tuple for caching.
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
            self.embedding_size + len(REGION_TYPES),
        ), f"Bad output shape: {region_inputs.size()}."

        return region_inputs
