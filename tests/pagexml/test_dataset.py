from contextlib import nullcontext as does_not_raise

import pytest

from document_segmentation.model.dataset import PageDataset
from document_segmentation.pagexml.datamodel import Label, Page


class TestLabel:
    @pytest.mark.parametrize(
        "scores,expected,expected_exception",
        [
            ([0.9, 0.1, 0], {"BEGIN": 0.9, "IN": 0.1, "END": 0}, does_not_raise()),
            ([0.1] * 5, None, pytest.raises(ValueError)),
            ([0.1] * 2, None, pytest.raises(ValueError)),
        ],
    )
    def test_map_scores(self, scores, expected, expected_exception):
        with expected_exception:
            assert Label.map_scores(scores) == expected

    @pytest.mark.parametrize(
        "pages, expected",
        [
            ([], [0.0, 0.0, 0.0]),
            ([Page(label=Label.BEGIN, regions=[], scan_nr=1)], [0.5, 1.0, 1.0]),
            (
                [
                    Page(label=Label.BEGIN, regions=[], scan_nr=1),
                    Page(label=Label.IN, regions=[], scan_nr=1),
                    Page(label=Label.IN, regions=[], scan_nr=1),
                    Page(label=Label.END, regions=[], scan_nr=1),
                ],
                [2.0, 1.3333333333333333, 2.0],
            ),
        ],
    )
    def test_class_weights(self, pages, expected):
        assert PageDataset(pages).class_weights() == expected
