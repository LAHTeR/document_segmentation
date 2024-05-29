import pytest

from document_segmentation.model.page_sequence_tagger import PageSequenceTagger
from document_segmentation.pagexml.datamodel.inventory import Inventory
from document_segmentation.pagexml.datamodel.label import SequenceLabel
from document_segmentation.pagexml.datamodel.page import Page


class TestPageLearner:
    @pytest.mark.parametrize(
        "inventories, expected",
        [
            ([], [0.0] * 4),
            (
                [
                    Inventory(
                        inv_nr=1,
                        inventory_part="",
                        pages=[
                            Page(
                                label=SequenceLabel.OUT,
                                scan_nr=1,
                                external_ref="test_ref",
                                regions=[],
                            )
                        ],
                    )
                ],
                [0.0, 1.0, 1.0, 0.5],
            ),
            (
                [
                    Inventory(
                        inv_nr=1,
                        inventory_part="",
                        pages=[
                            Page(
                                label=SequenceLabel.OUT,
                                scan_nr=1,
                                external_ref="test_ref",
                                regions=[],
                            )
                        ],
                    ),
                    Inventory(
                        inv_nr=1,
                        inventory_part="",
                        pages=[
                            Page(
                                label=SequenceLabel.OUT,
                                scan_nr=1,
                                external_ref="test_ref",
                                regions=[],
                            ),
                            Page(
                                label=SequenceLabel.BOUNDARY,
                                scan_nr=1,
                                external_ref="test_ref",
                                regions=[],
                            ),
                        ],
                    ),
                ],
                [0.0, 1.5, 3.0, 1.0],
            ),
        ],
    )
    def test_total_class_weights(self, inventories, expected):
        assert PageSequenceTagger().total_class_weights(inventories) == expected
