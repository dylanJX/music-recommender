"""
test_data_loader.py
===================
Unit tests for src/data_loader.py.

Test strategy:
  - All tests use small in-memory fixtures (StringIO or tmp_path) so that
    the real data files in data/ are never required to run the test suite.
  - Each loader function is tested for:
      * happy-path parsing (correct column names, dtypes, row count)
      * variable-length genre columns (tracks and albums)
      * empty files (should return an empty DataFrame, not raise)
      * missing fields / malformed lines (should raise a descriptive error)
"""

import io
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRACK_DATA_SAMPLE = """\
1|10|100|200|201
2|11|101|202
3|12|102|203|204|205
"""

ALBUM_DATA_SAMPLE = """\
10|100|200|201
11|101
"""

ARTIST_DATA_SAMPLE = """\
100
101
102
"""

GENRE_DATA_SAMPLE = """\
200
201
202
"""

INTERACTION_SAMPLE = """\
1\t1\t1
1\t2\t1
2\t3\t1
"""


# ---------------------------------------------------------------------------
# Placeholder tests — implement alongside data_loader functions
# ---------------------------------------------------------------------------


def test_load_track_data_columns():
    """Parsed track DataFrame must contain the expected columns."""
    pytest.skip("Not implemented yet")


def test_load_track_data_variable_genres():
    """Tracks with different numbers of genre columns must all parse cleanly."""
    pytest.skip("Not implemented yet")


def test_load_album_data_columns():
    """Parsed album DataFrame must contain the expected columns."""
    pytest.skip("Not implemented yet")


def test_load_artist_data_columns():
    """Parsed artist DataFrame must contain exactly ['artist_id']."""
    pytest.skip("Not implemented yet")


def test_load_genre_data_columns():
    """Parsed genre DataFrame must contain exactly ['genre_id']."""
    pytest.skip("Not implemented yet")


def test_load_interactions_columns():
    """Interaction DataFrame must contain at least user_id and track_id."""
    pytest.skip("Not implemented yet")


def test_load_interactions_empty_file(tmp_path: Path):
    """An empty interaction file should return an empty DataFrame, not raise."""
    pytest.skip("Not implemented yet")


def test_load_all_returns_all_keys():
    """load_all() dict must contain all expected keys."""
    pytest.skip("Not implemented yet")
