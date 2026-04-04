"""
test_feature_engineering.py
============================
Unit tests for src/feature_engineering.py.

Test strategy:
  - Construct minimal synthetic DataFrames that exercise the feature
    computation logic without requiring real data files.
  - Verify output shapes, column names, value ranges ([0, 1] for normalised
    scores), and edge cases (user with single interaction, track with no genre).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    build_user_profiles,
    build_track_features,
    compute_artist_affinity,
    compute_genre_affinity,
    compute_user_track_features,
    run,
)


# ---------------------------------------------------------------------------
# Shared fixtures
#
# Training interactions:
#   user 1 → track 10  (album 100, artist 200, genres [1, 2])
#   user 1 → track 11  (album 101, artist 201, genres [2, 3])
#   user 2 → track 10  (album 100, artist 200, genres [1, 2])
#
# Track 12 is in the catalogue but has no training interactions.
# ---------------------------------------------------------------------------

TRAIN = pd.DataFrame([
    {"user_id": 1, "track_id": 10},
    {"user_id": 1, "track_id": 11},
    {"user_id": 2, "track_id": 10},
])

TRACKS = pd.DataFrame([
    {"track_id": 10, "album_id": 100, "artist_id": 200, "genre_ids": [1, 2]},
    {"track_id": 11, "album_id": 101, "artist_id": 201, "genre_ids": [2, 3]},
    {"track_id": 12, "album_id": 100, "artist_id": 200, "genre_ids": [3]},
])

ALBUMS = pd.DataFrame([
    {"album_id": 100, "artist_id": 200, "genre_ids": [1, 2]},
    {"album_id": 101, "artist_id": 201, "genre_ids": [2, 3]},
])


# ---------------------------------------------------------------------------
# build_user_profiles
# ---------------------------------------------------------------------------

def test_build_user_profiles_shape():
    """Output must have one row per unique user in training data."""
    profiles = build_user_profiles(TRAIN, TRACKS, ALBUMS)
    assert len(profiles) == 2


def test_build_user_profiles_top_genres_type():
    """top_genres column must contain list objects, not scalars."""
    profiles = build_user_profiles(TRAIN, TRACKS, ALBUMS)
    for val in profiles["top_genres"]:
        assert isinstance(val, list)


def test_build_user_profiles_top_genres_order():
    """top_genres must be sorted by descending play count.

    User 1 interacts with genre 2 twice (tracks 10 and 11),
    so genre 2 should be first.
    """
    profiles = build_user_profiles(TRAIN, TRACKS, ALBUMS)
    u1_genres = profiles.loc[profiles["user_id"] == 1, "top_genres"].iloc[0]
    assert u1_genres[0] == 2


def test_build_user_profiles_n_interactions():
    """n_interactions must equal the user's total row count in training data."""
    profiles = build_user_profiles(TRAIN, TRACKS, ALBUMS)
    n1 = profiles.loc[profiles["user_id"] == 1, "n_interactions"].iloc[0]
    assert n1 == 2


def test_build_user_profiles_genre_counts_type():
    """genre_counts must contain dict objects."""
    profiles = build_user_profiles(TRAIN, TRACKS, ALBUMS)
    for val in profiles["genre_counts"]:
        assert isinstance(val, dict)


# ---------------------------------------------------------------------------
# build_track_features
# ---------------------------------------------------------------------------

def test_build_track_features_popularity_range():
    """popularity_score must be in [0, 1] for all tracks."""
    features = build_track_features(TRAIN, TRACKS)
    assert (features["popularity_score"] >= 0).all()
    assert (features["popularity_score"] <= 1).all()


def test_build_track_features_unknown_track():
    """Track absent from training data must have popularity_score == 0."""
    features = build_track_features(TRAIN, TRACKS)
    score = features.loc[features["track_id"] == 12, "popularity_score"].iloc[0]
    assert score == 0.0


def test_build_track_features_most_popular_is_one():
    """The most-played track must have popularity_score == 1.0."""
    features = build_track_features(TRAIN, TRACKS)
    # track 10 has 2 plays (max)
    score = features.loc[features["track_id"] == 10, "popularity_score"].iloc[0]
    assert score == pytest.approx(1.0)


def test_build_track_features_global_play_count():
    features = build_track_features(TRAIN, TRACKS)
    count = features.loc[features["track_id"] == 10, "global_play_count"].iloc[0]
    assert count == 2


def test_build_track_features_unique_listener_count():
    features = build_track_features(TRAIN, TRACKS)
    # track 10 was heard by users 1 and 2
    listeners = features.loc[features["track_id"] == 10, "unique_listener_count"].iloc[0]
    assert listeners == 2


# ---------------------------------------------------------------------------
# compute_genre_affinity
# ---------------------------------------------------------------------------

def test_genre_affinity_values_range():
    """All genre affinity values must be in [0, 1]."""
    profiles = build_user_profiles(TRAIN, TRACKS, ALBUMS)
    affinity = compute_genre_affinity(profiles)
    assert (affinity.values >= 0).all()
    assert (affinity.values <= 1).all()


def test_genre_affinity_index_is_users():
    profiles = build_user_profiles(TRAIN, TRACKS, ALBUMS)
    affinity = compute_genre_affinity(profiles)
    assert set(affinity.index) == {1, 2}


def test_genre_affinity_row_sums_to_one():
    """Each user's genre affinities must sum to 1 (they partition the user's interactions)."""
    profiles = build_user_profiles(TRAIN, TRACKS, ALBUMS)
    affinity = compute_genre_affinity(profiles)
    for uid in affinity.index:
        assert affinity.loc[uid].sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_artist_affinity
# ---------------------------------------------------------------------------

def test_artist_affinity_values_range():
    """All artist affinity values must be in [0, 1]."""
    profiles = build_user_profiles(TRAIN, TRACKS, ALBUMS)
    affinity = compute_artist_affinity(profiles)
    assert (affinity.values >= 0).all()
    assert (affinity.values <= 1).all()


def test_artist_affinity_row_sums_to_one():
    profiles = build_user_profiles(TRAIN, TRACKS, ALBUMS)
    affinity = compute_artist_affinity(profiles)
    for uid in affinity.index:
        assert affinity.loc[uid].sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def test_run_returns_all_keys():
    """feature_engineering.run() must return a dict with the four expected keys."""
    data = {
        "train": TRAIN,
        "test": TRAIN,
        "tracks": TRACKS,
        "albums": ALBUMS,
        "artists": pd.DataFrame({"artist_id": [200, 201]}),
        "genres": pd.DataFrame({"genre_id": [1, 2, 3]}),
    }
    result = run(data)
    assert set(result.keys()) == {
        "user_profiles", "track_features", "genre_affinity", "artist_affinity"
    }


# ---------------------------------------------------------------------------
# compute_user_track_features — output structure
# ---------------------------------------------------------------------------

PAIRS = pd.DataFrame([
    {"user_id": 1, "track_id": 12},  # user has seen this album/artist but not this track
    {"user_id": 2, "track_id": 11},  # user has NOT seen album 101 or artist 201
    {"user_id": 1, "track_id": 10},  # track with two genres
])

EXPECTED_COLS = {
    "user_id", "track_id",
    "album_score", "artist_score",
    "genre_count", "genre_max", "genre_min", "genre_mean",
    "genre_variance", "genre_median", "genre_sum", "genre_range",
    "genre_nonzero_count", "genre_weighted_mean",
}


def test_compute_user_track_features_shape():
    """Result must have one row per input pair."""
    result = compute_user_track_features(PAIRS, TRAIN, TRACKS)
    assert len(result) == len(PAIRS)


def test_compute_user_track_features_columns():
    """All expected feature columns must be present."""
    result = compute_user_track_features(PAIRS, TRAIN, TRACKS)
    assert EXPECTED_COLS.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# compute_user_track_features — album_score and artist_score
# ---------------------------------------------------------------------------

def test_album_score_is_not_nan_when_user_knows_album():
    """User 1 interacted with album 100 (via track 10) → album_score is not NaN."""
    result = compute_user_track_features(PAIRS, TRAIN, TRACKS)
    row = result[(result["user_id"] == 1) & (result["track_id"] == 12)].iloc[0]
    assert not pd.isna(row["album_score"])


def test_album_score_value():
    """User 1 has 2 total interactions; 1 is on album 100 → album_score == 0.5."""
    result = compute_user_track_features(PAIRS, TRAIN, TRACKS)
    row = result[(result["user_id"] == 1) & (result["track_id"] == 12)].iloc[0]
    assert row["album_score"] == pytest.approx(0.5)


def test_album_score_is_nan_for_unseen_album():
    """User 2 has never interacted with album 101 → album_score is NaN."""
    result = compute_user_track_features(PAIRS, TRAIN, TRACKS)
    row = result[(result["user_id"] == 2) & (result["track_id"] == 11)].iloc[0]
    assert pd.isna(row["album_score"])


def test_artist_score_value():
    """User 1 has 1 interaction with artist 200 out of 2 total → artist_score == 0.5."""
    result = compute_user_track_features(PAIRS, TRAIN, TRACKS)
    row = result[(result["user_id"] == 1) & (result["track_id"] == 12)].iloc[0]
    assert row["artist_score"] == pytest.approx(0.5)


def test_artist_score_is_nan_for_unseen_artist():
    """User 2 has never interacted with artist 201 → artist_score is NaN."""
    result = compute_user_track_features(PAIRS, TRAIN, TRACKS)
    row = result[(result["user_id"] == 2) & (result["track_id"] == 11)].iloc[0]
    assert pd.isna(row["artist_score"])


# ---------------------------------------------------------------------------
# compute_user_track_features — genre statistics
# ---------------------------------------------------------------------------

def test_genre_scores_in_range():
    """All non-NaN genre stat values must lie in [0, 1]."""
    result = compute_user_track_features(PAIRS, TRAIN, TRACKS)
    for col in ["genre_max", "genre_min", "genre_mean", "genre_weighted_mean"]:
        vals = result[col].dropna()
        assert (vals >= 0).all(), f"{col} has values < 0"
        assert (vals <= 1).all(), f"{col} has values > 1"


def test_genre_count_equals_track_genre_list_length():
    """genre_count must equal len(track.genre_ids) for every pair."""
    result = compute_user_track_features(PAIRS, TRAIN, TRACKS)
    # track 12 has genres [3]  → count 1
    # track 11 has genres [2,3] → count 2
    # track 10 has genres [1,2] → count 2
    for track_id, expected in [(12, 1), (11, 2), (10, 2)]:
        count = result.loc[result["track_id"] == track_id, "genre_count"].iloc[0]
        assert int(count) == expected, f"track {track_id}: expected genre_count={expected}, got {count}"


def test_genre_nonzero_count_le_genre_count():
    """genre_nonzero_count must be <= genre_count for every row."""
    result = compute_user_track_features(PAIRS, TRAIN, TRACKS)
    assert (result["genre_nonzero_count"] <= result["genre_count"]).all()


def test_genre_range_equals_max_minus_min():
    """genre_range must equal genre_max - genre_min for every row."""
    result = compute_user_track_features(PAIRS, TRAIN, TRACKS)
    for _, row in result.iterrows():
        if not pd.isna(row["genre_range"]):
            assert row["genre_range"] == pytest.approx(row["genre_max"] - row["genre_min"])


def test_genre_nonzero_count_for_unseen_genres():
    """User who has never interacted with any of the track's genres → genre_nonzero_count == 0."""
    train = pd.DataFrame([{"user_id": 1, "track_id": 10}])
    tracks = pd.DataFrame([
        {"track_id": 10, "album_id": 100, "artist_id": 200, "genre_ids": [1]},
        {"track_id": 11, "album_id": 101, "artist_id": 201, "genre_ids": [99]},
    ])
    pairs = pd.DataFrame([{"user_id": 1, "track_id": 11}])
    result = compute_user_track_features(pairs, train, tracks)
    row = result.iloc[0]
    assert row["genre_nonzero_count"] == 0
    assert row["genre_mean"] == pytest.approx(0.0)


def test_genre_weighted_mean_differs_from_unweighted():
    """When genre interaction counts vary, weighted mean should differ from unweighted mean.

    Setup:
      user 1 interacts with track 10 twice (genre 1 gets raw count 2)
                       and track 11 once  (genre 2 gets raw count 1)
      For track 12 (genres [1, 2]):
        genre 1 score = 2/3,  genre 2 score = 1/3
        unweighted mean = 0.5
        weighted mean   = (2/3 × 2 + 1/3 × 1) / 3 = 5/9 ≈ 0.5556
    """
    train = pd.DataFrame([
        {"user_id": 1, "track_id": 10},
        {"user_id": 1, "track_id": 10},
        {"user_id": 1, "track_id": 11},
    ])
    tracks = pd.DataFrame([
        {"track_id": 10, "album_id": 100, "artist_id": 200, "genre_ids": [1]},
        {"track_id": 11, "album_id": 101, "artist_id": 201, "genre_ids": [2]},
        {"track_id": 12, "album_id": 102, "artist_id": 202, "genre_ids": [1, 2]},
    ])
    pairs = pd.DataFrame([{"user_id": 1, "track_id": 12}])
    result = compute_user_track_features(pairs, train, tracks)
    row = result.iloc[0]
    assert row["genre_mean"] == pytest.approx(0.5)
    assert row["genre_weighted_mean"] == pytest.approx(5 / 9)


def test_no_genre_track_produces_nan_stats():
    """Track with empty genre_ids list → genre stat columns are NaN (no genres to aggregate)."""
    train = pd.DataFrame([{"user_id": 1, "track_id": 10}])
    tracks = pd.DataFrame([
        {"track_id": 10, "album_id": 100, "artist_id": 200, "genre_ids": [1]},
        {"track_id": 11, "album_id": 101, "artist_id": 201, "genre_ids": []},
    ])
    pairs = pd.DataFrame([{"user_id": 1, "track_id": 11}])
    result = compute_user_track_features(pairs, train, tracks)
    row = result.iloc[0]
    for col in ["genre_max", "genre_min", "genre_mean", "genre_variance",
                "genre_median", "genre_range", "genre_weighted_mean"]:
        assert pd.isna(row[col]), f"expected NaN for {col} on a no-genre track"
