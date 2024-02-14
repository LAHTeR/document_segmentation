import logging
import os
from pathlib import Path
from typing import Any


CWD: Path = Path(__file__).parent.absolute()

DATA_DIR: Path = CWD / "data"
TEST_SHEET: Path = DATA_DIR / "Spreadsheet Renate Revised.xlsx"

GENERALE_MISSIVEN_SHEET: Path = (
    DATA_DIR / "Overzicht van Generale Missiven in 1.04.02 v.3.csv"
)
GENERALE_MISSIVEN_DOCUMENT_DIR: Path = DATA_DIR / "generale_missiven"

DEFAULT_SERVER: str = "https://hucdrive.huc.knaw.nl/"
"""The default server URL."""
DEFAULT_BASE_PATH: str = "HTR/obp-v2-pagexml-leon-metadata-trimmed-2023-11/"
"""The default base path for the server directory that contains the inventory files."""

SERVER_USERNAME: str = os.getenv("HUC_USERNAME", "")
"""The username for accessing the HUC server."""
if not SERVER_USERNAME:
    logging.warning("No username set for accessing the HUC server.")

SERVER_PASSWORD: str = os.getenv("HUC_PASSWORD", "")
"""The password for accessing the HUC server."""
if not SERVER_PASSWORD:
    logging.warning("No password set for accessing the HUC server.")

LANGUAGE_MODEL: str = os.getenv("LANGUAGE_MODEL", "emanjavacas/GysBERT-v2")

# TODO: list all document types and their spelling variants
DOCUMENT_TYPES: dict[str, set[str]] = {
    "Journaal": {"Journaal", "Journael"},
    "Resolutie": {"Resolutie", "resolutien"},
    "Dagregister": {"Daghregister", "Dagregister", "Dag Register", "dag-register"},
    "Notulen": {"Notulen"},
    # "Register": {"Register"},
    "Monsterrol": {"Monsterrol", "Monsterrolle", "Monster Rolle"},
}
"""A mapping from document types to all spelling variants of that type."""


### Settings for the document segmentation model
PAGE_SEQUENCE_TAGGER_RNN_CONFIG: dict[str, Any] = {
    "hidden_size": 128,
    "num_layers": 1,
    "dropout": 0.1,
    "bidirectional": True,
}
"""Default configuration for the RNN module in the PageSequenceTagger"""

PAGE_EMBEDDING_RNN_CONFIG: dict[str, Any] = {
    "hidden_size": 128,
    "num_layers": 1,
    "dropout": 0.1,
    "bidirectional": True,
}
"""Default configuration for the RNN module in the PageEmbedding"""

PAGE_EMBEDDING_OUTPUT_SIZE: int = 64
"""Default output size for the PageEmbedding output layer"""

MAX_REGIONS_PER_PAGE: int = 16
"""
The maximum number of regions per page.
If a page has more regions, only the first and last regions are used.
"""

REGION_EMBEDDING_OUTPUT_SIZE: int = 128
"""Default output size for the RegionEmbedding output layer"""

REGION_TYPE_EMBEDDING_SIZE: int = 16
"""Output size for the RegionType embedding layer"""
