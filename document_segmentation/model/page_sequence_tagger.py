import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..pagexml.dataset import Label, PageXmlDataset
from .page_embedding import PageEmbedding

# TODO: move to device


class PageSequenceTagger(nn.Module):
    """A page sequence tagger that uses a GRU over the regions on a page."""

    def __init__(self) -> None:
        super().__init__()

        # TODO pass arguments to PageEmbedding
        self._page_embedding = PageEmbedding()
        self._gru = nn.GRU(
            input_size=self._page_embedding.gru.hidden_size * 2,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self._linear = nn.Linear(64 * 2, len(Label))
        self._softmax = nn.Softmax(dim=1)

    def forward(self, pages: PageXmlDataset):
        page_embeddings = self._page_embedding(list(pages.page_xmls()))

        assert page_embeddings.size() == (
            len(pages),
            self._page_embedding.output_size,
        ), "Bad shape: {pages.size()}"

        gru_out, hidden = self._gru(page_embeddings)
        assert gru_out.size() == (
            len(pages),
            self._page_embedding.output_size,
        ), f"Bad shape: {gru_out.size()}"

        output = self._linear(gru_out)
        assert output.size() == (len(pages), len(Label)), f"Bad shape: {output.size()}"

        softmax = self._softmax(output)
        assert softmax.size() == (len(pages), len(Label)), f"Bad shape: {output.size()}"

        return softmax  # output of last timestep

    def train_(self, pages: PageXmlDataset, epochs: int = 3, batch_size: int = 32):
        """Train the model.

        Args:
            pages (PageXmlDataset): The dataset to train on.
            epochs (int, optional): The number of epochs. Defaults to 3.
            batch_size (int, optional): The batch size. Defaults to 1.
        """
        self.train()

        counts = pages.label_counts()
        weights = [len(pages) / counts[label] for label in Label]
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights))
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for epoch in tqdm(range(epochs), unit="epoch"):
            for batch_start in tqdm(range(0, len(pages), batch_size), unit="batch"):
                batch = pages[batch_start : batch_start + batch_size]

                _labels = batch.label_tensor()

                optimizer.zero_grad()
                outputs = self(batch)
                loss = criterion(outputs, _labels)
                loss.backward()
                optimizer.step()
            tqdm.write(f"[Loss:\t{loss:.3f}]")

    def accuracy(self, pages, batch_size: int = 32) -> float:
        """Evaluate the model.

        Args:
            pages (PageXmlDataset): The dataset to evaluate on.
            batch_size (int, optional): The batch size. Defaults to 1.
        """
        correct = 0
        self.eval()

        for batch_start in tqdm(range(0, len(pages), batch_size), unit="batch"):
            batch_end = min(batch_start + batch_size, len(pages))
            batch = pages[batch_start:batch_end]

            outputs = self(batch)
            pred_labels = torch.argmax(outputs, dim=1)
            true_labels = torch.argmax(batch.label_tensor(), dim=1)

            _correct = (pred_labels == true_labels).sum().item()
            tqdm.write(f"[Batch Accuracy:\t{_correct / len(batch):.3f}]")

            correct += _correct
            tqdm.write(f"[Total Accuracy:\t{correct / batch_end:.3f}]")

        return correct / len(pages)
