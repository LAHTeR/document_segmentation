from typing import Mapping


DEFAULT_SERVER = "https://hucdrive.huc.knaw.nl/"
DEFAULT_BASE_PATH = "HTR/obp-v2-pagexml-leon-metadata-trimmed-2023-11/"

# TODO: list all document types and their spelling variants
DOCUMENT_TYPES: Mapping[str, list[str]] = {
    "Journaal": ["Journaal", "Journael"],
    "Resolutie": ["Resolutie"],
}
"""A mapping from document types to all spelling variants of that type."""
