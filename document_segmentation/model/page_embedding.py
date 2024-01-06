import logging

from pagexml.model.physical_document_model import PageXMLScan, PageXMLTextRegion
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from ..settings import REGION_TYPES
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
            input_size=self._transformer_dim + len(REGION_TYPES),  # + 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )

        self._linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, pages: list[PageXMLScan]):
        """Embed the pages using a Transformer model and a GRU over the regions on a page.

        Args:
            pages (list[PageXMLScan]): The pages to embed; each is embedded separately in this batch.
        """
        if not isinstance(pages, list):
            logging.warning(f"Expected a list of pages, got {pages}")
            pages = [pages]

        regions_batch: list[list[PageXMLTextRegion]] = [
            page.get_text_regions_in_reading_order() for page in pages
        ]
        max_regions: int = max(len(regions) for regions in regions_batch)

        region_inputs = pad_sequence(
            [self._region_model(regions) for regions in regions_batch],
            batch_first=True,
            padding_value=0.0,
        )
        _expected_size = (
            len(pages),
            max_regions,
            self._region_model.embedding_size + len(REGION_TYPES),  # + 2,
        )
        assert (
            region_inputs.size() == _expected_size
        ), f"Bad region input shape: {region_inputs.size()}. Expected: {_expected_size}"

        gru_out, hidden = self.gru(region_inputs)

        out = self._linear(gru_out)
        _expected_size = (len(pages), max_regions, self.output_size)
        assert (
            out.size() == _expected_size
        ), f"Bad output shape: {out.size()}. Expected: {_expected_size}"

        final_step_output_batch = out[:, -1, :]
        _expected_size = (len(pages), self.output_size)
        assert (
            final_step_output_batch.size() == _expected_size
        ), f"Bad output shape: {final_step_output_batch.size()}. Expected: {_expected_size}"

        return final_step_output_batch
