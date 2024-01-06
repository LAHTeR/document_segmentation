import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class RegionEmbedding(nn.Module):
    def __init__(self, num_region_types, transformer_model_name, hidden_size):
        super(RegionEmbedding, self).__init__()

        # Embedding layer for region types
        self.type_embedding = nn.Embedding(num_region_types, hidden_size)

        # Transformer-based text embedding
        self.transformer_model = AutoModel.from_pretrained(transformer_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
        self.text_embedding_size = self.transformer_model.config.hidden_size

        # Linear layer for coordinates
        self.coordinates_embedding = nn.Linear(2, hidden_size)

    def forward(self, region_types, text_input, coordinates):
        # Embed region types
        type_embedded = self.type_embedding(region_types)

        # Tokenize and obtain text embeddings
        # FIXME: text embeddings are not padded
        text_encodings = self.tokenizer(
            text_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.transformer_model.config.max_position_embeddings,
        )
        text_embeddings = self.transformer_model(**text_encodings).last_hidden_state

        # Embed coordinates
        coordinates_embedded = self.coordinates_embedding(coordinates)

        # Pad embeddings to the same length
        # TODO: fix size to global maximum (from Transformer model sequence length)
        # TODO: turn into a function
        max_length = max(
            type_embedded.size(0),
            text_embeddings.size(1),
            coordinates_embedded.size(0),
        )
        if type_embedded.size(0) < max_length:
            type_embedded = torch.cat(
                (
                    type_embedded,
                    torch.zeros(
                        (max_length - type_embedded.size(0), type_embedded.size(1))
                    ),
                ),
            )
        if text_embeddings.size(1) < max_length:
            text_embeddings = torch.cat(
                (
                    text_embeddings,
                    torch.zeros(
                        (
                            text_embeddings.size(0),
                            max_length - text_embeddings.size(1),
                            text_embeddings.size(2),
                        )
                    ),
                )
            )
        if coordinates_embedded.size(0) < max_length:
            coordinates_embedded = torch.cat(
                (
                    coordinates_embedded,
                    torch.zeros(
                        (
                            max_length - coordinates_embedded.size(0),
                            coordinates_embedded.size(1),
                        )
                    ),
                )
            )

        # Combine embeddings
        region_embedding = type_embedded + text_embeddings + coordinates_embedded

        return region_embedding


class PageTagger(nn.Module):
    def __init__(
        self,
        num_region_types,
        transformer_model_name,
        hidden_size,
        rnn_hidden_size,
        num_classes,
    ):
        super(PageTagger, self).__init__()

        # Region embedding module
        self.region_embedding = RegionEmbedding(
            num_region_types, transformer_model_name, hidden_size
        )

        # GRU layer to process the sequence of regions
        self.gru_regions = nn.LSTM(
            input_size=hidden_size,
            hidden_size=rnn_hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        # GRU layer to process the sequence of pages
        self.gru_pages = nn.LSTM(
            input_size=rnn_hidden_size * 2,
            hidden_size=rnn_hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        # Output layer for sequence tagging
        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)

    def forward(self, pages):
        # pages is a list of dictionaries, each representing a page with keys: 'region_types', 'text_inputs', 'coordinates'

        # Embed each region in a page
        region_embeddings = [
            self.region_embedding(
                page["region_types"], page["text_inputs"], page["coordinates"]
            )
            for page in pages
        ]

        # Pad region embeddings to the same length
        # TODO: fix size to global maximum
        max_length = max(
            [region_embedding.size(1) for region_embedding in region_embeddings]
        )
        region_embeddings_padded = [
            torch.cat(
                (
                    region_embedding,
                    torch.zeros(
                        (
                            region_embedding.size(0),
                            max_length - region_embedding.size(1),
                            region_embedding.size(2),
                        )
                    ),
                ),
                dim=1,
            )
            for region_embedding in region_embeddings
        ]
        region_embeddings_stacked = torch.cat(region_embeddings_padded, dim=1)

        # Pass through the GRU layer for regions
        gru_regions_out, _ = self.gru_regions(region_embeddings_stacked)

        # Pass through the GRU layer for pages
        gru_pages_out, _ = self.gru_pages(gru_regions_out)

        # Apply fully connected layer for sequence tagging
        output = self.fc(gru_pages_out)

        return output
