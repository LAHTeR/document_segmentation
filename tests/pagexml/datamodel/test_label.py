import pytest

from document_segmentation.pagexml.datamodel.label import SequenceLabel, Tanap


class TestLabel:
    @pytest.mark.parametrize(
        "label, expected",
        [
            (Tanap.UNK, [1] + [0] * 14),
            (Tanap.DAGREGISTERS, [0, 1] + [0] * 13),
            (Tanap.STUKKEN_OVERIG, [0] * 14 + [1]),
            (SequenceLabel.UNK, [1, 0, 0, 0]),
            (SequenceLabel.BOUNDARY, [0, 1, 0, 0]),
            (SequenceLabel.IN, [0, 0, 1, 0]),
            (SequenceLabel.OUT, [0, 0, 0, 1]),
        ],
    )
    def test_to_list(self, label, expected):
        assert label.to_list() == expected
