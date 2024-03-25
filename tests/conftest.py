from pathlib import Path

import pytest
import requests_mock

TEST_INV_NR = 1201
TEST_SHEET_SIZE = 192681


@pytest.fixture
def mock_request(inventory_nr: int = TEST_INV_NR):
    test_url = f"https://hucdrive.huc.knaw.nl/HTR/obp-v2-pagexml-leon-metadata-trimmed-2023-11/{inventory_nr}.zip"

    test_file: Path = (DATA_DIR / str(inventory_nr)).with_suffix(".zip")
    with requests_mock.Mocker() as mocker:
        mocker.get(test_url, content=test_file.open("rb").read())

        yield mocker

        assert mocker.call_count == 1, f"Request was made {mocker.call_count} times."
        assert mocker.request_history[0].url == test_url, "Request URL does not match."
        assert (
            mocker.request_history[0].method == "GET"
        ), "Request was not a GET request."


CWD = Path(__file__).parent.absolute()
DATA_DIR = CWD / "data"
GENERALE_MISSIVEN_CSV = DATA_DIR / "Overzicht van Generale Missiven in 1.04.02 v.3.csv"
TEST_FILE = (DATA_DIR / str(TEST_INV_NR)).with_suffix(".zip")
