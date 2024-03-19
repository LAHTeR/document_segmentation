from enum import IntEnum
from typing import Iterable


class Label(IntEnum):
    """Labels for pages in a sequence."""

    UNK = 0
    BEGIN = 1
    IN = 2
    END = 3
    OUT = 4

    def to_list(self) -> list[int]:
        """Convert the label to a list of integers.

        Returns:
            list[int]: A list of integers representing the label.
        """
        return [int(self == label) for label in Label]

    def successor(self) -> "Label":
        """Return the successor of the label.

        Returns:
            Label: The successor of the label.
        """
        if self == Label.BEGIN:
            return Label.IN
        elif self == Label.IN:
            return Label.IN
        elif self == Label.END or self == Label.OUT:
            return Label.OUT
        return Label(self.value + 1)

    @staticmethod
    def map_scores(scores: Iterable[float]) -> dict[str, float]:
        """Map a list of scores to a dictionary of label names and scores.

        Args:
            scores (Iterable[float]): List of scores with the same length as Label.
        Returns:
            dict[str, float]: Dictionary mapping label names to scores.
        """
        if len(scores) != len(Label):
            raise ValueError(
                f"Expected {len(Label)} scores, got {len(scores)}: {scores}"
            )
        return {label.name: score for label, score in zip(Label, scores)}
