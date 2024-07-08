import pytest
import sentence_transformers

from document_segmentation.model.region_embedding import (
    RegionEmbedding,
    RegionEmbeddingSentenceTransformer,
)


class TestRegionEmbedding:
    @pytest.mark.parametrize(
        "model_name,expected",
        [
            ("my_model/bert", RegionEmbedding),
            ("my_model/sentence_bert", RegionEmbeddingSentenceTransformer),
            ("sentence_bert", RegionEmbeddingSentenceTransformer),
            ("SENTENCEBERT", RegionEmbeddingSentenceTransformer),
        ],
    )
    def test_from_model_name(self, mocker, model_name, expected):
        if expected == RegionEmbedding:
            _mocker = mocker.patch(
                "transformers.AutoModel.from_pretrained", autospec=True
            )
            mocker.patch("transformers.AutoTokenizer.from_pretrained", autospec=True)
        elif expected == RegionEmbeddingSentenceTransformer:
            _mocker = mocker.patch.object(
                sentence_transformers.SentenceTransformer,
                "_load_auto_model",
                autospec=True,
            )
            mocker.patch.object(
                sentence_transformers.SentenceTransformer,
                "get_sentence_embedding_dimension",
                autospec=True,
            )

        assert RegionEmbedding.from_model_name(model_name).__class__ == expected
        assert _mocker.call_count == 1
