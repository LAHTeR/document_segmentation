import logging
from functools import cache
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer

from .. import settings
from ..pagexml.datamodel.region import Region, RegionType
from ..settings import (
    LANGUAGE_MODEL,
    REGION_EMBEDDING_OUTPUT_SIZE,
    REGION_TYPE_EMBEDDING_SIZE,
)
from .device_module import DeviceModule


class RegionEmbedding(nn.Module, DeviceModule):
    """A module to create a representation of a region using a Transformer model."""

    def __init__(
        self,
        *,
        transformer_model_name: str,
        output_size: int = REGION_EMBEDDING_OUTPUT_SIZE,
        region_type_embedding_size: int = REGION_TYPE_EMBEDDING_SIZE,
        line_separator: str = "\n",
        device: Optional[str] = None,
    ):
        super().__init__()

        self._line_separator = line_separator

        self._init_transformer(transformer_model_name)

        self._region_type = nn.Embedding(len(RegionType), region_type_embedding_size)

        # TODO self._position = nn.Linear(1, 1)

        self._linear = nn.Linear(
            in_features=self.text_embedding_size + region_type_embedding_size,
            out_features=output_size,
        )

        self.output_size = output_size

        self.to_device(device)

    def _init_transformer(self, transformer_model_name):
        self._transformer_model: AutoModel = AutoModel.from_pretrained(
            transformer_model_name
        )
        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            transformer_model_name
        )
        self._max_length = self._transformer_model.config.max_position_embeddings

    @property
    def text_embedding_size(self) -> int:
        """Return the size of the embeddings."""
        return self._transformer_model.config.hidden_size

    def _text_embeddings(self, region_batch: tuple[Region, ...]) -> torch.Tensor:
        """Embed the text of a page using a Transformers model."""
        if settings.UPDATE_LM_WEIGHTS:
            return self._compute_text_embeddings(region_batch)
        else:
            return self._text_embeddings_cached(region_batch)

    @cache
    def _text_embeddings_cached(self, region_batch: tuple[Region, ...]) -> torch.Tensor:
        """Compute the text embeddings for the regions and cache them.
        Weights of the language model are not updated.
        """
        # TODO: caching only works on the batch level, individual regions are not cached
        with torch.no_grad():
            return self._compute_text_embeddings(region_batch)

    def _compute_text_embeddings(
        self, region_batch: tuple[Region, ...]
    ) -> torch.Tensor:
        """Embed the text of a page using a Transformers model.

        Args:
            region_batch: The regions to embed. Must not be mutable for caching (ie. not a list, but a tuple)
                If empty, return a zero tensor of embeddings dimensionality
        Returns:
            A tensor of shape (1, embedding_size).
        """

        if region_batch:
            region_texts: list[str] = [
                self._line_separator.join(region.lines) for region in region_batch
            ]

            text_inputs = self._tokenizer(
                region_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self._max_length,
            )

            out = self._transformer_model(
                **text_inputs.to(self._device)
            ).last_hidden_state
            cls_tokens = out[:, 0, :]  # CLS token is first token of sequence
        else:
            logging.debug("Empty region batch.")
            cls_tokens = torch.zeros(0, self.text_embedding_size).to(self._device)

        return cls_tokens

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
            regions (list[Region]): The regions to embed.
        """
        if not isinstance(regions, tuple):
            logging.debug(f"Converting regions into a tuple (was: '{type(regions)}').")
            regions = tuple(regions)

        text_embeddings = self._text_embeddings(regions).float()
        expected_size = (len(regions), self.text_embedding_size)
        assert (
            text_embeddings.size() == expected_size
        ), f"Output shape was {text_embeddings.size()}, but should be {expected_size}."

        region_types = (
            self._region_type(
                torch.IntTensor(
                    [RegionType.indices(set(region.types)) for region in regions]
                ).to(self._device)
            ).mean(dim=1)
            if regions
            else torch.zeros(0, self._region_type.embedding_dim).to(self._device)
        )

        # region_coordinates = self._coordinates_tensor(region).float() TODO

        region_inputs = torch.cat(
            [
                text_embeddings,
                region_types,
                # region_coordinates
            ],
            dim=-1,
        )
        return self._linear(region_inputs)

    @classmethod
    def from_model_name(cls, model_name: str = LANGUAGE_MODEL, **kwargs):
        """Try to autodetect the model type from the model name.

        Can be either a standard BERT model or a SentenceBERT model,
        resulting in RegionEmbedding or RegionEmbeddingSentenceTransformer respectively.

        Args:
            model_name: str the name of the model (using HuggingFace identifiers)
            **kwargs: passed to the selected class constructor.
        Returns:
            a (sub-class of a) RegionEmbedding model
        """

        if "sentence".casefold() in model_name.casefold():
            cls = RegionEmbeddingSentenceTransformer

        logging.info(f"Using a '{cls.__name__}' model.")
        return cls(transformer_model_name=model_name, **kwargs)


class RegionEmbeddingSentenceTransformer(RegionEmbedding):
    @property
    def text_embedding_size(self) -> int:
        """Return the size of the embeddings."""
        return self._transformer_model.get_sentence_embedding_dimension()

    def _init_transformer(self, transformer_model_name):
        self._transformer_model = SentenceTransformer(transformer_model_name)

    def _compute_text_embeddings(
        self, region_batch: tuple[Region, ...]
    ) -> torch.Tensor:
        if region_batch:
            region_texts: list[str] = [
                self._line_separator.join(region.lines) for region in region_batch
            ]

            embedding = self._transformer_model.encode(
                region_texts, convert_to_tensor=True
            )
        else:
            embedding = torch.zeros(0, self.text_embedding_size).to(self._device)
        return embedding
