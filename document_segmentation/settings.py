import logging
import os
from pathlib import Path
from typing import Mapping

CWD = Path(__file__).parent.absolute()

DATA_DIR = CWD / "data"
TEST_SHEET = DATA_DIR / "Spreadsheet Renate Revised.xlsx"

DEFAULT_SERVER = "https://hucdrive.huc.knaw.nl/"
DEFAULT_BASE_PATH = "HTR/obp-v2-pagexml-leon-metadata-trimmed-2023-11/"

SERVER_USERNAME = os.getenv("HUC_USERNAME", "")
if not SERVER_USERNAME:
    logging.error("No username set for accessing the HUC server.")

SERVER_PASSWORD = os.getenv("HUC_PASSWORD", "")
if not SERVER_PASSWORD:
    logging.error("No password set for accessing the HUC server.")


# TODO: list all document types and their spelling variants
DOCUMENT_TYPES: Mapping[str, list[str]] = {
    "Journaal": ["Journaal", "Journael"],
    "Resolutie": ["Resolutie"],
}
"""A mapping from document types to all spelling variants of that type."""
