from enum import IntEnum


class Label(IntEnum):
    """Labels for pages in a sequence."""

    UNK = 0
    """No annotation available."""
    BEGIN = 1
    """The beginning of a document."""
    IN = 2
    """Part of a document."""
    END = 3
    """The end of a document."""
    OUT = 4
    """Not part of any document."""
    END_BEGIN = 5
    """Page on which a document ends, and another one begins"""

    def to_list(self) -> list[int]:
        """Convert the label to a list of integers.

        Returns:
            list[int]: A list of integers representing the label.
        """
        return [int(self == label) for label in Label]
