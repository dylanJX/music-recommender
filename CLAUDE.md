# Music Recommender — Architecture Reference

This document is the primary reference for Claude Code when working in this repository.
Read it before making changes to understand intent, constraints, and conventions.

---

## Project goal

Build a rule-based music recommender that takes user listening histories and track/album/artist/genre metadata and produces ranked recommendation lists, formatted as a submission CSV.

---

## Directory layout

```
Cproject/
├── data/                    ← raw data files (not committed; place manually)
│   ├── trainIterm2.txt
│   ├── testIterm2.txt
│   ├── trackData2.txt
│   ├── albumData2.txt
│   ├── artistData2.txt
│   └── genreData2.txt
├── src/
│   ├── __init__.py
│   ├── data_loader.py       ← Step 0:  parse all 6 raw files
│   ├── feature_engineering.py  ← Step 1a: derive user/track features
│   ├── scorer.py            ← Step 1b: rule-based weighted ranking
│   ├── cold_start.py        ← Steps 2a/2b: new-user and new-track fallbacks
│   └── pipeline.py          ← orchestrates everything; CLI entry point
├── submissions/             ← generated CSVs written here
├── tests/
│   ├── test_data_loader.py
│   └── test_feature_engineering.py
├── config.yaml              ← all tunable parameters
├── requirements.txt
└── CLAUDE.md                ← this file
```

---

## Data file formats

| File | Format | Notes |
|------|--------|-------|
| `trainIterm2.txt` | tab-separated user-item records | exact schema TBD |
| `testIterm2.txt` | same as train | |
| `trackData2.txt` | `TrackId\|AlbumId\|ArtistId\|GenreId_1\|…\|GenreId_k` | variable number of genre columns |
| `albumData2.txt` | `AlbumId\|ArtistId\|GenreId_1\|…\|GenreId_k` | variable genres |
| `artistData2.txt` | one `ArtistId` per line | |
| `genreData2.txt` | one `GenreId` per line | |

All IDs are integers. Variable-length genre columns must be normalised to a Python list (`genre_ids`) in the loader.

---

## Module responsibilities (one module, one concern)

### `data_loader.py`
- Parse raw files into clean DataFrames.
- Expose `load_all(data_dir)` → dict of DataFrames.
- No feature computation here — pure I/O.

### `feature_engineering.py`  _(Step 1a)_
- Consume raw DataFrames; produce derived feature artifacts.
- Key outputs: `user_profiles`, `track_features`, `genre_affinity` matrix, `artist_affinity` matrix.
- Entry point: `run(data: dict) → dict`.
- Results may be cached to `.cache/features/` (controlled by `config.yaml`).

### `scorer.py`  _(Step 1b)_
- Weighted linear scoring of (user, track) pairs.
- All weights live in `config.yaml` under `scorer.weights`; never hard-code them.
- Each signal (genre match, artist match, album match, popularity, novelty) has its own function so it can be tested and tuned independently.
- Entry point: `rank_candidates(user_id, candidate_track_ids, features, config) → list[int]`.

### `cold_start.py`  _(Steps 2a & 2b)_
- **2a — New user**: no training history → fall back to `global_popularity_fallback()` or genre-filtered popularity.
- **2b — New track**: track unseen in training → proxy score from artist/album/genre signals, then blend with warm candidates.
- Strategy selection is config-driven (`cold_start.new_user_strategy`, `cold_start.new_track_strategy`).
- Ultimate fallback for any case: `global_popularity_fallback()`.

### `pipeline.py`
- Load config → load data → build features → iterate test users → route to scorer or cold-start → write submission CSV.
- Runnable as `python -m src.pipeline [--config config.yaml] [--run-name <name>]`.
- Output goes to `submissions/<run_name>.csv`.

---

## Configuration

All tunable parameters are in `config.yaml`.  **Never hard-code a weight, threshold, path, or strategy name in source code.**  Read them from the config dict passed down from `pipeline.py`.

Key sections:

| Section | Controls |
|---------|----------|
| `data` | file names and data directory path |
| `pipeline` | `top_n`, `run_name`, `candidate_strategy` |
| `scorer.weights` | per-signal weights for the linear combiner |
| `scorer.popularity_log_base` | dampening of raw popularity counts |
| `cold_start` | fallback strategies and cold-track slot fraction |
| `features.cache_dir` | where to persist computed feature artifacts |

---

## Testing conventions

- Tests live in `tests/`; run with `pytest` from the project root.
- Tests must **not** depend on files in `data/`; use `tmp_path` fixtures or in-memory `StringIO`.
- Each loader function and each scorer signal function should have at least one happy-path test and one edge-case test.
- Placeholder tests use `pytest.skip("Not implemented yet")` until the corresponding source function is implemented.

---

## Implementation sequence

When implementing (not yet done), follow this order:

1. `data_loader.py` — needed by everything else.
2. `feature_engineering.py` — depends on data_loader output.
3. `scorer.py` — depends on feature_engineering output.
4. `cold_start.py` — depends on feature_engineering and scorer signals.
5. `pipeline.py` — wires all of the above together.
6. Fill in test stubs after each module is implemented.

---

## Conventions

- Python 3.10+; use `from __future__ import annotations` for forward-ref compatibility.
- All public functions must have NumPy-style docstrings (Parameters / Returns sections).
- IDs are integers throughout; do not cast to strings unless writing the final CSV.
- DataFrames are preferred over dicts-of-lists for tabular data; plain dicts are fine for mappings.
- Do not add logging statements during stub phase; add them when implementing.
