import logging
from typing import Any, Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from ..pagexml.datamodel.page import Page
from ..pagexml.datamodel.region import Region
from ..settings import PAGE_EMBEDDING_OUTPUT_SIZE, PAGE_EMBEDDING_RNN_CONFIG
from .device_module import DeviceModule
from .region_embedding import RegionEmbeddingSentenceTransformer


class PageEmbedding(nn.Module, DeviceModule):
    """Embeds a page using a Transformer model and a GRU over the regions on a page."""

    def __init__(
        self,
        *,
        rnn_config: dict[str, Any] = PAGE_EMBEDDING_RNN_CONFIG,
        output_size: int = PAGE_EMBEDDING_OUTPUT_SIZE,
        device: Optional[str] = None,
    ):
        super().__init__()

        self._region_model = RegionEmbeddingSentenceTransformer(device=device)

        self._transformer_dim = self._region_model.text_embedding_size
        self.output_size = output_size

        # LSTM, because GRU does not seem not to work on MPS: https://github.com/pytorch/pytorch/issues/94691
        self._rnn = nn.LSTM(
            input_size=self._region_model.output_size, batch_first=True, **rnn_config
        )
        self._linear = nn.Linear(
            in_features=self._rnn.hidden_size * (self._rnn.bidirectional + 1),
            out_features=output_size,
        )

        self.output_size = output_size

        self.to_device(device)

    @property
    def rnn(self) -> nn.LSTM:
        """Return the RNN used for the page embedding."""
        return self._rnn

    def forward(self, pages: list[Page]):
        """Embed the pages using a Transformer model and a GRU over the regions on a page.

        Args:
            pages (list[PageXMLScan]): The pages to embed; each is embedded separately in this batch.
        """
        if not isinstance(pages, list):
            logging.warning(f"Expected a list of pages, got {pages}")
            pages = [pages]

        regions_batch: list[list[Region]] = [page.regions for page in pages]

        if any(regions for regions in regions_batch):
            region_inputs = pad_sequence(
                [self._region_model(regions) for regions in regions_batch],
                batch_first=True,
                padding_value=0.0,
            )
        else:
            logging.warning("No regions found in the batch, using zero tensor")
            region_inputs = torch.zeros(
                1, len(pages), self._region_model.output_size
            ).to(self._device)

        rnn_out, hidden = self._rnn(region_inputs)

        out = self._linear(rnn_out)

        final_step_output_batch = out[:, -1, :]
        _expected_size = (len(pages), self.output_size)
        assert (
            final_step_output_batch.size() == _expected_size
        ), f"Bad output shape: {final_step_output_batch.size()}. Expected: {_expected_size}"

        return final_step_output_batch
