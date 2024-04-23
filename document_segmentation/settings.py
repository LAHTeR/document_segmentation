import logging
import os
import sys
from pathlib import Path
from typing import Any

CWD: Path = Path(__file__).parent.absolute()

DATA_DIR: Path = CWD / "data"

RENATE_TANAP_CATEGORISATION_SHEET: Path = (
    DATA_DIR
    / "Appendix F - Spreadsheet concerning the TANAP document categorisation, Renate Smit, January 2024.xlsx"
)

RENATE_ANALYSIS_SHEETS: tuple[Path] = tuple(DATA_DIR.glob("Analysis Renate ????.csv"))

GENERALE_MISSIVEN_SHEET: Path = (
    DATA_DIR / "Overzicht van Generale Missiven in 1.04.02 v.3.csv"
)

INVENTORY_DIR: Path = DATA_DIR / "inventories"
"""Path to store the inventory files locally."""

DEFAULT_SERVER: str = "https://hucdrive.huc.knaw.nl/"
"""The default server URL."""
DEFAULT_BASE_PATH: str = "HTR/obp-v2-pagexml-leon-metadata-trimmed-2024-03/"
"""The default base path for the server directory that contains the inventory files."""

SERVER_USERNAME: str = os.getenv("HUC_USERNAME", "")
"""The username for accessing the HUC server."""
if not SERVER_USERNAME:
    logging.warning("No username set for accessing the HUC server.")

SERVER_PASSWORD: str = os.getenv("HUC_PASSWORD", "")
"""The password for accessing the HUC server."""
if not SERVER_PASSWORD:
    logging.warning("No password set for accessing the HUC server.")

INV_NR_UUID_MAPPING_FILE: Path = DATA_DIR / "1.04.02_inventory2uuid.json"
if not INV_NR_UUID_MAPPING_FILE.exists():
    raise ValueError(
        f"Inventory number to UUID mapping file not found: {INV_NR_UUID_MAPPING_FILE}"
    )
THUMBNAILS_DIR: Path = DATA_DIR / "thumbnails"
DEFAULT_THUMBNAIL_SIZE: str = ",200"
"""Valid size formats are documented here: https://iiif.io/api/image/2.1/#size"""

LANGUAGE_MODEL: str = os.getenv(
    "LANGUAGE_MODEL",
    "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers",
)
"""The name of the language model to use for the region classifier.

The type of model needs to be with the RegionClassifier class:
- use RegionClassifier for standard transformer models
- use RegionClassifierSentenceTransformer for SentenceTransformer models
"""

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


MIN_REGION_TEXT_LENGTH: int = 20
"""The minimum number of characters of the text(s) in a region.
Shorter regions are filtered out during training and inference."""

REGION_TYPE_EMBEDDING_SIZE: int = 16
"""Size for the RegionType embedding layer"""

REGION_EMBEDDING_OUTPUT_SIZE: int = 2**9
"""Output size for the (linear) RegionEmbedding output layer"""

PAGE_EMBEDDING_RNN_CONFIG: dict[str, Any] = {
    "hidden_size": 2**8,
    "num_layers": 2,
    "dropout": 0.1,
    "bidirectional": True,
}
"""Default configuration for the RNN module in the PageEmbedding"""

PAGE_EMBEDDING_OUTPUT_SIZE: int = 2**8
"""Default output size for the PageEmbedding linear output layer"""

PAGE_SEQUENCE_TAGGER_RNN_CONFIG: dict[str, Any] = {
    "hidden_size": 2**8,
    "num_layers": 2,
    "dropout": 0.1,
    "bidirectional": True,
}
"""Default configuration for the RNN module in the PageSequenceTagger"""

MAX_INVENTORY_SIZE: int = int(os.getenv("MAX_INVENTORY_SIZE", "0"))
"""The maximum number of pages per inventory. Larger inventories are chunked. Set to 0 to disable chunking.
This is used to limit (GPU) memory usage."""
MIN_INVENTORY_SIZE: int = int(os.getenv("MAX_INVENTORY_SIZE", "0"))
"""The minimum number of pages per inventory. Smaller inventories are skipped.
This is used because small inventories can be problematic for training and evaluation."""

MAX_EMPTY_SEQUENCE: int = 1
"""If an annotated inventory has more than this number of subsequent empty OUT pages, they are replaced with a single OUT page."""

UPDATE_LM_WEIGHTS: bool = bool(os.getenv("UPDATE_LM_WEIGHTS", "True"))
"""If True, the weights of the language model are updated during training.
If False, the text embeddings are cached and never updated.

Set to empty string to deactivate updating and caching.
"""


def as_dict():
    excluded_keys = {"SERVER_USERNAME", "SERVER_PASSWORD"}
    return {
        key: "*" * 8 if key in excluded_keys else value
        for key, value in sys.modules[__name__].__dict__.items()
        if key.isupper()
    }
