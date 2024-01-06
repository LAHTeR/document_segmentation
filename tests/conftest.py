from pathlib import Path

import pytest

from document_segmentation.pagexml.inventory import Inventory


@pytest.fixture
def inventory():
    return Inventory("1201", cache_directory=DATA_DIR)


CWD = Path(__file__).parent.absolute()
DATA_DIR = CWD / "data"
GENERALE_MISSIVEN_CSV = DATA_DIR / "Overzicht van Generale Missiven in 1.04.02 v.3.csv"
