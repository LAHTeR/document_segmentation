import logging

from torch import nn
from torch.nn.utils.rnn import pad_sequence

from ..pagexml.datamodel import Page, Region, RegionType
from .region_embedding import RegionEmbedding


class PageEmbedding(nn.Module):
    """Embeds a page using a Transformer model and a GRU over the regions on a page."""

    def __init__(
        self,
        hidden_size: int = 64,
        dropout: float = 0.1,
        num_layers: int = 1,
        output_size: int = 128,
    ):
        super().__init__()

        self._region_model = RegionEmbedding()

        self._transformer_dim = self._region_model.embedding_size
        self.output_size = output_size

        # GRU over the regions on a page
        self.gru = nn.GRU(
            # text embeddings + region types + region coordinates
            input_size=self._transformer_dim + len(RegionType),  # + 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )

        self._linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, pages: list[Page]):
        """Embed the pages using a Transformer model and a GRU over the regions on a page.

        Args:
            pages (list[PageXMLScan]): The pages to embed; each is embedded separately in this batch.
        """
        if not isinstance(pages, list):
            logging.warning(f"Expected a list of pages, got {pages}")
            pages = [pages]

        regions_batch: list[list[Region]] = [page.regions for page in pages]

        region_inputs = pad_sequence(
            [self._region_model(regions) for regions in regions_batch],
            batch_first=True,
            padding_value=0.0,
        )

        gru_out, hidden = self.gru(region_inputs)

        out = self._linear(gru_out)

        final_step_output_batch = out[:, -1, :]
        _expected_size = (len(pages), self.output_size)
        assert (
            final_step_output_batch.size() == _expected_size
        ), f"Bad output shape: {final_step_output_batch.size()}. Expected: {_expected_size}"

        return final_step_output_batch
