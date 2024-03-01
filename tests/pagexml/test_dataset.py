from contextlib import nullcontext as does_not_raise

import pytest

from document_segmentation.model.dataset import (
    DocumentDataset,
    PageDataset,
    RegionDataset,
)
from document_segmentation.pagexml.datamodel.label import Label
from document_segmentation.pagexml.datamodel.page import Page
from document_segmentation.pagexml.datamodel.region import Region

REGION1 = Region.model_validate(
    {
        "id": "region_8c74aff4-3cf0-4ce5-925e-4ff424fb564b_10",
        "types": [
            "physical_structure_doc",
            "text_region",
            "pagexml_doc",
            "header",
        ],
        "coordinates": [],
        "lines": ["Drentfeste. wijse, voorsienige heer discrete Heeren."],
    }
)
REGION2 = Region.model_validate(
    {
        "id": "test_region_2",
        "types": [
            "physical_structure_doc",
            "text_region",
            "pagexml_doc",
            "header",
        ],
        "coordinates": [],
        "lines": [],
    }
)
REGION3 = Region.model_validate(
    {
        "id": "test_region_2",
        "types": [
            "physical_structure_doc",
            "text_region",
            "pagexml_doc",
            "header",
        ],
        "coordinates": [],
        "lines": ["abc"],
    }
)


def page(scan_nr: int = 1, doc_id: str = "test doc"):
    """Create a page with the given document ID."""
    return Page(label=Label.BEGIN, regions=[], scan_nr=scan_nr, doc_id=doc_id)


class TestLabel:
    @pytest.mark.parametrize(
        "scores,expected,expected_exception",
        [
            (
                [0.9, 0.1, 0, 0],
                {"BEGIN": 0.9, "IN": 0.1, "END": 0, "OUT": 0},
                does_not_raise(),
            ),
            ([0.1] * 5, None, pytest.raises(ValueError)),
            ([0.1] * 2, None, pytest.raises(ValueError)),
        ],
    )
    def test_map_scores(self, scores, expected, expected_exception):
        with expected_exception:
            assert Label.map_scores(scores) == expected


class TestDocumentDataset:
    @pytest.mark.parametrize(
        "dataset,batch_size,expected",
        [
            (DocumentDataset([]), 2, 0),
            (DocumentDataset([PageDataset([])]), 2, 0),
            (DocumentDataset([PageDataset([page()])]), 2, 1),
            (DocumentDataset([PageDataset([page()] * 5)]), 2, 3),
            (DocumentDataset([PageDataset([page()] * 5)] * 2), 2, 6),
        ],
    )
    def test_n_batches(self, dataset, batch_size, expected):
        assert dataset.n_batches(batch_size) == expected

    @pytest.mark.parametrize(
        "dataset,portion,expected_training,expected_test",
        [
            (DocumentDataset([]), 0.8, DocumentDataset([]), DocumentDataset([])),
            (
                DocumentDataset([page()] * 10),
                0.8,
                DocumentDataset([page()] * 8),
                DocumentDataset([page()] * 2),
            ),
            (
                DocumentDataset([page()] * 9),
                0.8,
                DocumentDataset([page()] * 7),
                DocumentDataset([page()] * 2),
            ),
        ],
    )
    def test_split(self, dataset, portion, expected_training, expected_test):
        training, test = dataset.split(portion)

        assert training == expected_training, "Training dataset is not as expected"
        assert test == expected_test, "Test dataset is not as expected"


class TestPageDataset:
    @pytest.mark.parametrize(
        "dataset,batch_size,expected",
        [
            (
                PageDataset([page(doc_id="test doc 1")] * 2),
                2,
                [PageDataset([page(doc_id="test doc 1")] * 2)],
            ),
            (
                PageDataset([page(doc_id="test doc 1")] * 4),
                2,
                [
                    PageDataset([page(doc_id="test doc 1")] * 2),
                    PageDataset([page(doc_id="test doc 1")] * 2),
                ],
            ),
            (
                PageDataset([page(doc_id="test doc 1"), page(doc_id="test doc 2")]),
                1,
                [
                    PageDataset([page(doc_id="test doc 1")]),
                    PageDataset([page(doc_id="test doc 2")]),
                ],
            ),
            (
                PageDataset(
                    [
                        page(doc_id="test doc 1"),
                        page(doc_id="test doc 1"),
                        page(doc_id="test doc 2"),
                    ]
                ),
                4,
                [
                    PageDataset(
                        [
                            page(doc_id="test doc 1"),
                            page(doc_id="test doc 1"),
                            page(doc_id="test doc 2"),
                        ]
                    )
                ],
            ),
        ],
    )
    def test_batches(self, dataset, batch_size, expected):
        assert list(dataset.batches(batch_size)) == expected

    @pytest.mark.parametrize(
        "pages, expected",
        [
            ([], [0.0, 0.0, 0.0, 0.0]),
            ([Page(label=Label.BEGIN, regions=[], scan_nr=1)], [0.5, 1.0, 1.0, 1.0]),
            (
                [
                    Page(label=Label.BEGIN, regions=[], scan_nr=1),
                    Page(label=Label.IN, regions=[], scan_nr=1),
                    Page(label=Label.IN, regions=[], scan_nr=1),
                    Page(label=Label.END, regions=[], scan_nr=1),
                ],
                [2.0, 1.3333333333333333, 2.0, 4.0],
            ),
        ],
    )
    def test_class_weights(self, pages, expected):
        assert PageDataset(pages).class_weights() == expected

    @pytest.mark.parametrize(
        "pages,expected",
        [
            ([], []),
            ([Page(label=Label.BEGIN, scan_nr=1, regions=[])], []),
            ([Page(label=Label.BEGIN, scan_nr=1, regions=[REGION1])], [Label.BEGIN]),
            (
                [Page(label=Label.IN, scan_nr=1, regions=[REGION1] * 5)],
                [Label.IN] * 5,
            ),
        ],
    )
    def test_region_labels(self, pages, expected):
        assert list(PageDataset(pages).region_labels()) == expected

    def test_regions(self):
        dataset = PageDataset([Page(label=Label.BEGIN, scan_nr=1, regions=[REGION1])])

        assert list(dataset.regions()) == [REGION1]


class TestRegionDataset:
    @pytest.mark.parametrize(
        "pages,expected_regions",
        [
            ([], []),
            ([Page(label=Label.BEGIN, scan_nr=1, regions=[])], []),
            ([Page(label=Label.BEGIN, scan_nr=1, regions=[REGION1])], [REGION1]),
            (
                [Page(label=Label.BEGIN, scan_nr=1, regions=[REGION1, REGION1])],
                [REGION1, REGION1],
            ),
        ],
    )
    def test_from_page_dataset(self, pages, expected_regions):
        page_dataset = PageDataset(pages)
        dataset = RegionDataset.from_page_dataset(page_dataset)

        assert dataset.regions() == expected_regions

    @pytest.mark.parametrize(
        "dataset, expected",
        [
            (RegionDataset([], []), [0.0, 0.0, 0.0, 0.0]),
            (RegionDataset([REGION1], [Label.BEGIN]), [0.5, 1.0, 1.0, 1.0]),
            (
                RegionDataset([REGION1, REGION1], [Label.BEGIN, Label.IN]),
                [1.0, 1.0, 2.0, 2.0],
            ),
        ],
    )
    def test_class_weights(self, dataset, expected):
        assert dataset.class_weights() == expected

    @pytest.mark.parametrize("sample_size", [-1, 0, 1, 2, 3, 4, 10])
    def test_sample(self, sample_size: int):
        dataset = RegionDataset(
            [REGION1, REGION1, REGION1], [Label.BEGIN, Label.IN, Label.END]
        )
        expected_exception = (
            does_not_raise()
            if sample_size in range(0, len(dataset) + 1)
            else pytest.raises(ValueError)
        )
        with expected_exception:
            sample = dataset.sample(sample_size)

            assert len(sample) == sample_size
            assert all(region in dataset.regions() for region in sample.regions())
            assert all(label in dataset.labels() for label in sample.labels())

    @pytest.mark.parametrize(
        "dataset,min_chars,expected",
        [
            (RegionDataset([], []), 0, RegionDataset([], [])),
            (
                RegionDataset([REGION1], [Label.BEGIN]),
                1,
                RegionDataset([REGION1], [Label.BEGIN]),
            ),
            (RegionDataset([REGION1], [Label.BEGIN]), 100, RegionDataset([], [])),
            (
                RegionDataset([REGION1, REGION2], [Label.BEGIN, Label.IN]),
                1,
                RegionDataset([REGION1], [Label.BEGIN]),
            ),
            (
                RegionDataset(
                    [REGION1, REGION2, REGION3], [Label.BEGIN, Label.IN, Label.END]
                ),
                3,
                RegionDataset([REGION1, REGION3], [Label.BEGIN, Label.END]),
            ),
            (
                RegionDataset(
                    [REGION1, REGION2, REGION3], [Label.BEGIN, Label.IN, Label.END]
                ),
                4,
                RegionDataset([REGION1], [Label.BEGIN]),
            ),
            (
                RegionDataset(
                    [REGION1, REGION2, REGION3], [Label.BEGIN, Label.IN, Label.END]
                ),
                100,
                RegionDataset([], []),
            ),
        ],
    )
    def test_remove_empty(self, dataset, min_chars: int, expected):
        assert dataset.remove_empty(min_chars) == expected

    @pytest.mark.parametrize(
        "dataset,sample_size,expected",
        [
            (RegionDataset([], []), None, RegionDataset([], [])),
            (
                RegionDataset([REGION1], [Label.BEGIN]),
                1,
                RegionDataset([REGION1], [Label.BEGIN]),
            ),
            (
                RegionDataset([REGION1], [Label.BEGIN]),
                None,
                RegionDataset([REGION1], [Label.BEGIN]),
            ),
            (
                RegionDataset([REGION1] * 3, [Label.BEGIN] * 2 + [Label.IN]),
                None,
                RegionDataset([REGION1] * 2, [Label.BEGIN, Label.IN]),
            ),
            (
                RegionDataset([REGION1] * 4, [Label.BEGIN] * 3 + [Label.IN]),
                2,
                RegionDataset([REGION1] * 3, [Label.BEGIN] * 2 + [Label.IN]),
            ),
        ],
    )
    def test_balance(self, dataset, sample_size, expected):
        assert dataset.balance(sample_size) == expected
