import logging
import os
from pathlib import Path
from typing import Mapping

CWD: Path = Path(__file__).parent.absolute()

DATA_DIR: Path = CWD / "data"
TEST_SHEET: Path = DATA_DIR / "Spreadsheet Renate Revised.xlsx"
GENERALE_MISSIVEN_SHEET: Path = (
    DATA_DIR / "Overzicht van Generale Missiven in 1.04.02 v.3.csv"
)

PAGEXML_CACHE_DIRECTORY: Path = DATA_DIR / "pagexml_cache"
"""Directory that contains the downloaded PageXML files."""

DEFAULT_SERVER: str = "https://hucdrive.huc.knaw.nl/"
DEFAULT_BASE_PATH: str = "HTR/obp-v2-pagexml-leon-metadata-trimmed-2023-11/"

SERVER_USERNAME: str = os.getenv("HUC_USERNAME", "")
if not SERVER_USERNAME:
    logging.error("No username set for accessing the HUC server.")

SERVER_PASSWORD: str = os.getenv("HUC_PASSWORD", "")
if not SERVER_PASSWORD:
    logging.error("No password set for accessing the HUC server.")


# TODO: list all document types and their spelling variants
DOCUMENT_TYPES: Mapping[str, set[str]] = {
    "Journaal": {"Journaal", "Journael"},
    "Resolutie": {"Resolutie", "resolutien"},
    "Dagregister": {"Daghregister", "Dagregister", "Dag Register", "dag-register"},
    "Notulen": {"Notulen"},
    # "Register": {"Register"},
    "Monsterrol": {"Monsterrol", "Monsterrolle", "Monster Rolle"},
}
"""A mapping from document types to all spelling variants of that type."""
