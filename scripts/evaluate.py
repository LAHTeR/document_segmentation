#!/usr/bin/env python3

from document_segmentation.evaluation.dataset import TestSet


if __name__ == "__main__":
    test_set = TestSet()
    test_set.recall()
