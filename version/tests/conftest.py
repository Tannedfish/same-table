import os
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SAME_TABLE_OFFLINE", "1")  # never download models in tests
os.environ["LIBRARY_PATH"] = str(ROOT / "data" / "seed_library.json")  # fixed fixture, not the built library

from same_table.config import Settings  # noqa: E402
from same_table.library import load_library  # noqa: E402

TODAY = date(2026, 10, 1)


@pytest.fixture(scope="session")
def library():
    return load_library(ROOT / "data" / "seed_library.json")


@pytest.fixture
def settings():
    return Settings.load()
