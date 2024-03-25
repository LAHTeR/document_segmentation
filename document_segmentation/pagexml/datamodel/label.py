from enum import IntEnum


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
