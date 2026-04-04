"""
data_loader.py
==============
Responsible for parsing all raw data files from the data/ directory and
exposing clean Python objects (DataFrames or dicts) to the rest of the pipeline.

Files handled:
  - trainItem2.txt   : user-item interaction records for training
  - testItem2.txt    : user-item interaction records for evaluation
  - trackData2.txt   : TrackId|AlbumId|ArtistId|GenreId_1|...|GenreId_k  (pipe-delimited, variable columns)
  - albumData2.txt   : AlbumId|ArtistId|GenreId_1|...|GenreId_k
  - artistData2.txt  : ArtistId
  - genreData2.txt   : GenreId

All paths are resolved relative to a configurable data directory so that
callers never hard-code file locations.

Interaction file block format
-----------------------------
Both train and test files use the same block structure::

    user_id|n
    track_id[\tplay_count]
    track_id[\tplay_count]
    ...   (n lines)

The optional ``\tplay_count`` column is present in the training file only.
Missing play_count values default to 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


DATA_DIR = Path(__file__).parent.parent / "data"


def load_interactions(path: Path) -> pd.DataFrame:
    """Parse a train or test interaction file.

    The file uses a block format: each user's section starts with a header
    line ``user_id|n``, followed by *n* lines.  Training lines have the form
    ``track_id\\tplay_count``; test lines contain only ``track_id``.

    Parameters
    ----------
    path : Path
        Absolute or relative path to the interaction .txt file.

    Returns
    -------
    pd.DataFrame
        Columns: ['user_id', 'track_id', 'play_count'].
        ``play_count`` is 1 for every test record (count is not available).
    """
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        if not raw:
            i += 1
            continue
        if "|" in raw:
            uid_str, n_str = raw.split("|", 1)
            uid = int(uid_str)
            n = int(n_str)
            for j in range(n):
                track_line = lines[i + 1 + j].strip()
                if "\t" in track_line:
                    tid_str, cnt_str = track_line.split("\t", 1)
                    rows.append({"user_id": uid, "track_id": int(tid_str),
                                 "play_count": int(cnt_str)})
                else:
                    rows.append({"user_id": uid, "track_id": int(track_line),
                                 "play_count": 1})
            i += n + 1
        else:
            i += 1  # skip malformed lines

    return pd.DataFrame(rows, columns=["user_id", "track_id", "play_count"])


def load_track_data(path: Path | None = None) -> pd.DataFrame:
    """Parse trackData2.txt.

    Format: ``TrackId|AlbumId|ArtistId|GenreId_1|...|GenreId_k``
    Genre columns are variable-length; they are normalised into a list column
    'genre_ids'.  Literal ``"None"`` values in any column are treated as NaN
    and replaced with -1 for integer columns.

    Parameters
    ----------
    path : Path or None
        Path to the file.  Defaults to ``DATA_DIR/trackData2.txt``.

    Returns
    -------
    pd.DataFrame
        Columns: ['track_id', 'album_id', 'artist_id', 'genre_ids']
    """
    if path is None:
        path = DATA_DIR / "trackData2.txt"

    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            track_id = int(parts[0])
            album_id = int(parts[1]) if parts[1] not in ("None", "") else -1
            artist_id = int(parts[2]) if parts[2] not in ("None", "") else -1
            genre_ids = [
                int(p) for p in parts[3:]
                if p not in ("None", "") and p.isdigit()
            ]
            rows.append({
                "track_id": track_id,
                "album_id": album_id,
                "artist_id": artist_id,
                "genre_ids": genre_ids,
            })

    return pd.DataFrame(rows, columns=["track_id", "album_id", "artist_id", "genre_ids"])


def load_album_data(path: Path | None = None) -> pd.DataFrame:
    """Parse albumData2.txt.

    Format: ``AlbumId|ArtistId|GenreId_1|...|GenreId_k``
    Literal ``"None"`` values are treated as -1 for integer columns.

    Parameters
    ----------
    path : Path or None
        Path to the file.  Defaults to ``DATA_DIR/albumData2.txt``.

    Returns
    -------
    pd.DataFrame
        Columns: ['album_id', 'artist_id', 'genre_ids']
    """
    if path is None:
        path = DATA_DIR / "albumData2.txt"

    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            album_id = int(parts[0])
            artist_id = int(parts[1]) if parts[1] not in ("None", "") else -1
            genre_ids = [
                int(p) for p in parts[2:]
                if p not in ("None", "") and p.isdigit()
            ]
            rows.append({
                "album_id": album_id,
                "artist_id": artist_id,
                "genre_ids": genre_ids,
            })

    return pd.DataFrame(rows, columns=["album_id", "artist_id", "genre_ids"])


def load_artist_data(path: Path | None = None) -> pd.DataFrame:
    """Parse artistData2.txt.

    Format: one ArtistId per line.

    Parameters
    ----------
    path : Path or None
        Path to the file.  Defaults to ``DATA_DIR/artistData2.txt``.

    Returns
    -------
    pd.DataFrame
        Columns: ['artist_id']
    """
    if path is None:
        path = DATA_DIR / "artistData2.txt"

    artist_ids: list[int] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and line.isdigit():
                artist_ids.append(int(line))

    return pd.DataFrame({"artist_id": artist_ids})


def load_genre_data(path: Path | None = None) -> pd.DataFrame:
    """Parse genreData2.txt.

    Format: one GenreId per line.

    Parameters
    ----------
    path : Path or None
        Path to the file.  Defaults to ``DATA_DIR/genreData2.txt``.

    Returns
    -------
    pd.DataFrame
        Columns: ['genre_id']
    """
    if path is None:
        path = DATA_DIR / "genreData2.txt"

    genre_ids: list[int] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and line.isdigit():
                genre_ids.append(int(line))

    return pd.DataFrame({"genre_id": genre_ids})


def load_all(data_dir: Path = DATA_DIR, config: dict | None = None) -> dict[str, Any]:
    """Convenience wrapper: load every data file at once.

    Parameters
    ----------
    data_dir : Path
        Directory containing all raw data files.
    config : dict or None
        Parsed config.yaml dict.  When provided, file names are read from
        ``config['data']``; otherwise the default file names are used.

    Returns
    -------
    dict with keys:
        'train'   -> pd.DataFrame  (training interactions)
        'test'    -> pd.DataFrame  (test interactions)
        'tracks'  -> pd.DataFrame
        'albums'  -> pd.DataFrame
        'artists' -> pd.DataFrame
        'genres'  -> pd.DataFrame
    """
    data_dir = Path(data_dir)
    cfg_data = (config or {}).get("data", {})

    train_file = cfg_data.get("train_file", "trainItem2.txt")
    test_file  = cfg_data.get("test_file",  "testItem2.txt")
    track_file = cfg_data.get("track_file", "trackData2.txt")
    album_file = cfg_data.get("album_file", "albumData2.txt")
    artist_file = cfg_data.get("artist_file", "artistData2.txt")
    genre_file  = cfg_data.get("genre_file",  "genreData2.txt")

    return {
        "train":   load_interactions(data_dir / train_file),
        "test":    load_interactions(data_dir / test_file),
        "tracks":  load_track_data(data_dir / track_file),
        "albums":  load_album_data(data_dir / album_file),
        "artists": load_artist_data(data_dir / artist_file),
        "genres":  load_genre_data(data_dir / genre_file),
    }
