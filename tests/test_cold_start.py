"""
test_cold_start.py
==================
Unit tests for src/cold_start.py.

Test strategy:
  - All tests use small in-memory DataFrames (no data/ files required).
  - One test class / section per public function.
  - Coverage includes each fallback level of resolve_album_score, both
    aggregation modes of intra_album_proxy_score, and edge cases
    (empty train, unknown track_id, cold user, no siblings).

Shared fixture data
-------------------
Training interactions (multiple rows = multiple plays):

  user 1: track 10 ×2 (album 100, artist 200)
          track 11 ×1 (album 100, artist 201)
          track 13 ×1 (album 101, artist 202)
  → total 4 interactions; albums {100: 3, 101: 1}; artists {200: 2, 201: 1, 202: 1}

  user 2: track 10 ×1 (album 100, artist 200)
          track 14 ×1 (album 102, artist 203)
  → total 2 interactions; albums {100: 1, 102: 1}; artists {200: 1, 203: 1}

  user 3 is absent from training (cold user).

Catalogue tracks:
  10 — album 100, artist 200  (warm)
  11 — album 100, artist 201  (warm)
  12 — album 101, artist 202  (cold: never in train; sibling of 13)
  13 — album 101, artist 202  (warm; sibling of 12)
  14 — album 102, artist 203  (warm)
  15 — album 103, artist 204  (cold: no interactions at all; no siblings)
  16 — album 100, artist 200  (cold: never in train; siblings are 10 and 11)
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.cold_start import (
    compute_global_imputed_scores,
    global_popularity_fallback,
    intra_album_proxy_score,
    is_cold_track,
    is_cold_user,
    recommend_for_cold_track,
    recommend_for_new_user,
    resolve_album_score,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TRAIN = pd.DataFrame([
    {"user_id": 1, "track_id": 10},
    {"user_id": 1, "track_id": 10},   # played twice
    {"user_id": 1, "track_id": 11},
    {"user_id": 1, "track_id": 13},
    {"user_id": 2, "track_id": 10},
    {"user_id": 2, "track_id": 14},
])

TRACKS = pd.DataFrame([
    {"track_id": 10, "album_id": 100, "artist_id": 200, "genre_ids": [1, 2]},
    {"track_id": 11, "album_id": 100, "artist_id": 201, "genre_ids": [2, 3]},
    {"track_id": 12, "album_id": 101, "artist_id": 202, "genre_ids": [3]},
    {"track_id": 13, "album_id": 101, "artist_id": 202, "genre_ids": [3]},
    {"track_id": 14, "album_id": 102, "artist_id": 203, "genre_ids": [1]},
    {"track_id": 15, "album_id": 103, "artist_id": 204, "genre_ids": [4]},
    {"track_id": 16, "album_id": 100, "artist_id": 200, "genre_ids": [1]},
])

# track_features: popularity_score is normalised global play count.
# play counts from TRAIN: 10→3, 11→1, 13→1, 14→1; 12,15,16→0
# max = 3 → scores: 10=1.0, 11≈0.333, 13≈0.333, 14≈0.333, 12=0.0, 15=0.0, 16=0.0
TRACK_FEATURES = pd.DataFrame([
    {"track_id": 10, "popularity_score": 1.0},
    {"track_id": 11, "popularity_score": 1 / 3},
    {"track_id": 12, "popularity_score": 0.0},
    {"track_id": 13, "popularity_score": 1 / 3},
    {"track_id": 14, "popularity_score": 1 / 3},
    {"track_id": 15, "popularity_score": 0.0},
    {"track_id": 16, "popularity_score": 0.0},
])

CONFIG = {
    "cold_start": {
        "new_user_strategy": "global_popularity",
        "intra_album_agg": "mean",
        "cold_track_slot_fraction": 0.20,
    }
}


# ---------------------------------------------------------------------------
# is_cold_user
# ---------------------------------------------------------------------------

class TestIsColdUser:
    def test_warm_user_is_not_cold(self):
        assert is_cold_user(1, TRAIN) is False

    def test_absent_user_is_cold(self):
        assert is_cold_user(3, TRAIN) is True

    def test_empty_train_always_cold(self):
        empty = pd.DataFrame(columns=["user_id", "track_id"])
        assert is_cold_user(1, empty) is True


# ---------------------------------------------------------------------------
# is_cold_track
# ---------------------------------------------------------------------------

class TestIsColdTrack:
    def test_warm_track_is_not_cold(self):
        assert is_cold_track(10, TRAIN) is False

    def test_absent_track_is_cold(self):
        assert is_cold_track(12, TRAIN) is True

    def test_empty_train_always_cold(self):
        empty = pd.DataFrame(columns=["user_id", "track_id"])
        assert is_cold_track(10, empty) is True


# ---------------------------------------------------------------------------
# compute_global_imputed_scores
# ---------------------------------------------------------------------------

class TestComputeGlobalImputedScores:
    def test_returns_expected_keys(self):
        result = compute_global_imputed_scores(TRAIN, TRACKS)
        assert "album_score_mean" in result
        assert "artist_score_mean" in result

    def test_album_score_mean_in_range(self):
        result = compute_global_imputed_scores(TRAIN, TRACKS)
        assert 0.0 < result["album_score_mean"] <= 1.0

    def test_artist_score_mean_in_range(self):
        result = compute_global_imputed_scores(TRAIN, TRACKS)
        assert 0.0 < result["artist_score_mean"] <= 1.0

    def test_album_score_mean_exact(self):
        # User 1: album 100 → 3/4=0.75, album 101 → 1/4=0.25
        # User 2: album 100 → 1/2=0.5,  album 102 → 1/2=0.5
        # Mean = (0.75 + 0.25 + 0.5 + 0.5) / 4 = 0.5
        result = compute_global_imputed_scores(TRAIN, TRACKS)
        assert result["album_score_mean"] == pytest.approx(0.5)

    def test_artist_score_mean_exact(self):
        # User 1: artist 200 → 2/4=0.5, 201 → 1/4=0.25, 202 → 1/4=0.25
        # User 2: artist 200 → 1/2=0.5, 203 → 1/2=0.5
        # Mean = (0.5 + 0.25 + 0.25 + 0.5 + 0.5) / 5 = 0.4
        result = compute_global_imputed_scores(TRAIN, TRACKS)
        assert result["artist_score_mean"] == pytest.approx(0.4)

    def test_empty_train_returns_zeros(self):
        empty = pd.DataFrame(columns=["user_id", "track_id"])
        result = compute_global_imputed_scores(empty, TRACKS)
        assert result["album_score_mean"] == 0.0
        assert result["artist_score_mean"] == 0.0


# ---------------------------------------------------------------------------
# intra_album_proxy_score
# ---------------------------------------------------------------------------

class TestIntraAlbumProxyScore:
    def test_returns_none_for_cold_user(self):
        # User 3 has no training rows → no sibling interactions
        assert intra_album_proxy_score(3, 12, TRAIN, TRACKS, CONFIG) is None

    def test_returns_none_when_no_sibling_interactions(self):
        # User 2 has never interacted with any track on album 101 (tracks 12, 13)
        assert intra_album_proxy_score(2, 12, TRAIN, TRACKS, CONFIG) is None

    def test_returns_none_when_unknown_track_id(self):
        assert intra_album_proxy_score(1, 999, TRAIN, TRACKS, CONFIG) is None

    def test_returns_none_when_no_siblings_exist(self):
        # Track 15 is the only track on album 103 — no siblings
        assert intra_album_proxy_score(1, 15, TRAIN, TRACKS, CONFIG) is None

    def test_proxy_mean_single_sibling(self):
        # Track 12 on album 101; sibling = track 13
        # User 1 played track 13 once out of 4 total → normalised score = 0.25
        result = intra_album_proxy_score(1, 12, TRAIN, TRACKS, CONFIG)
        assert result == pytest.approx(0.25)

    def test_proxy_mean_multiple_siblings(self):
        # Track 16 on album 100; siblings = track 10, track 11
        # User 1: track 10 → 2/4=0.5, track 11 → 1/4=0.25 → mean = 0.375
        result = intra_album_proxy_score(1, 16, TRAIN, TRACKS, CONFIG)
        assert result == pytest.approx(0.375)

    def test_proxy_max_multiple_siblings(self):
        # Same as above but agg='max' → should return 0.5 (track 10 score)
        cfg = {**CONFIG, "cold_start": {**CONFIG["cold_start"], "intra_album_agg": "max"}}
        result = intra_album_proxy_score(1, 16, TRAIN, TRACKS, cfg)
        assert result == pytest.approx(0.5)

    def test_proxy_result_in_unit_interval(self):
        result = intra_album_proxy_score(1, 12, TRAIN, TRACKS, CONFIG)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_max_result_geq_mean(self):
        cfg_max = {**CONFIG, "cold_start": {**CONFIG["cold_start"], "intra_album_agg": "max"}}
        mean_val = intra_album_proxy_score(1, 16, TRAIN, TRACKS, CONFIG)
        max_val = intra_album_proxy_score(1, 16, TRAIN, TRACKS, cfg_max)
        assert max_val >= mean_val


# ---------------------------------------------------------------------------
# resolve_album_score — fallback priority chain
# ---------------------------------------------------------------------------

class TestResolveAlbumScore:
    """Each test exercises exactly one level of the priority chain."""

    def test_priority_1_direct_score_returned_when_not_nan(self):
        # Direct album_score = 0.75 → must be returned unchanged
        result = resolve_album_score(1, 12, 0.75, TRAIN, TRACKS, CONFIG)
        assert result == pytest.approx(0.75)

    def test_priority_1_zero_direct_score_is_valid_not_nan(self):
        # album_score = 0.0 is a valid direct score (not missing)
        result = resolve_album_score(1, 12, 0.0, TRAIN, TRACKS, CONFIG)
        assert result == pytest.approx(0.0)

    def test_priority_2_intra_album_proxy_used_when_direct_is_nan(self):
        # album_score = NaN → falls back to intra-album proxy
        # User 1 + track 12 → proxy via track 13 = 0.25
        result = resolve_album_score(1, 12, float("nan"), TRAIN, TRACKS, CONFIG)
        assert result == pytest.approx(0.25)

    def test_priority_2_intra_album_proxy_used_when_direct_is_none(self):
        result = resolve_album_score(1, 12, None, TRAIN, TRACKS, CONFIG)
        assert result == pytest.approx(0.25)

    def test_priority_3_global_mean_used_when_no_proxy(self):
        # User 2 has no sibling interactions for track 12 (album 101)
        # → priority 2 returns None → use global_album_mean = 0.5
        result = resolve_album_score(
            2, 12, float("nan"), TRAIN, TRACKS, CONFIG, global_album_mean=0.5
        )
        assert result == pytest.approx(0.5)

    def test_priority_4_zero_when_all_signals_absent(self):
        # User 2, track 12, no proxy, no global mean
        result = resolve_album_score(
            2, 12, float("nan"), TRAIN, TRACKS, CONFIG, global_album_mean=None
        )
        assert result == pytest.approx(0.0)

    def test_priority_4_zero_when_global_mean_is_nan(self):
        result = resolve_album_score(
            2, 12, float("nan"), TRAIN, TRACKS, CONFIG, global_album_mean=float("nan")
        )
        assert result == pytest.approx(0.0)

    def test_result_is_always_finite(self):
        for album_score in [float("nan"), None, 0.3]:
            result = resolve_album_score(
                1, 12, album_score, TRAIN, TRACKS, CONFIG, global_album_mean=0.5
            )
            assert math.isfinite(result)

    def test_result_is_always_non_negative(self):
        result = resolve_album_score(
            2, 12, float("nan"), TRAIN, TRACKS, CONFIG, global_album_mean=None
        )
        assert result >= 0.0


# ---------------------------------------------------------------------------
# global_popularity_fallback
# ---------------------------------------------------------------------------

class TestGlobalPopularityFallback:
    def test_returns_list(self):
        result = global_popularity_fallback([10, 11, 12], TRACK_FEATURES)
        assert isinstance(result, list)

    def test_most_popular_track_is_first(self):
        result = global_popularity_fallback([10, 11, 12], TRACK_FEATURES)
        assert result[0] == 10

    def test_respects_n(self):
        result = global_popularity_fallback([10, 11, 12, 13, 14], TRACK_FEATURES, n=2)
        assert len(result) == 2

    def test_fewer_candidates_than_n(self):
        # Should return all candidates without error
        result = global_popularity_fallback([10], TRACK_FEATURES, n=100)
        assert result == [10]

    def test_empty_candidates_returns_empty(self):
        result = global_popularity_fallback([], TRACK_FEATURES, n=10)
        assert result == []

    def test_tie_broken_by_track_id_ascending(self):
        # Tracks 12, 15, 16 all have popularity_score = 0.0; lowest id first
        result = global_popularity_fallback([12, 15, 16], TRACK_FEATURES, n=3)
        assert result == [12, 15, 16]

    def test_candidates_not_in_track_features_are_excluded(self):
        result = global_popularity_fallback([999], TRACK_FEATURES, n=5)
        assert result == []


# ---------------------------------------------------------------------------
# recommend_for_new_user
# ---------------------------------------------------------------------------

class TestRecommendForNewUser:
    def test_returns_list(self):
        result = recommend_for_new_user(3, [10, 11, 12], TRACK_FEATURES, CONFIG, n=3)
        assert isinstance(result, list)

    def test_most_popular_first(self):
        result = recommend_for_new_user(3, [10, 11, 12], TRACK_FEATURES, CONFIG, n=3)
        assert result[0] == 10

    def test_respects_n(self):
        result = recommend_for_new_user(3, [10, 11, 12, 13, 14], TRACK_FEATURES, CONFIG, n=2)
        assert len(result) == 2

    def test_cold_user_same_result_as_warm_user(self):
        # Popularity is user-agnostic; cold and warm users get same ranking
        r_cold = recommend_for_new_user(3, [10, 11, 14], TRACK_FEATURES, CONFIG, n=3)
        r_warm = recommend_for_new_user(1, [10, 11, 14], TRACK_FEATURES, CONFIG, n=3)
        assert r_cold == r_warm

    def test_returns_subset_of_candidates(self):
        candidates = [11, 12, 14]
        result = recommend_for_new_user(3, candidates, TRACK_FEATURES, CONFIG, n=10)
        assert set(result).issubset(set(candidates))


# ---------------------------------------------------------------------------
# recommend_for_cold_track
# ---------------------------------------------------------------------------

class TestRecommendForColdTrack:
    FEATURES = {"track_features": TRACK_FEATURES}

    def test_returns_list(self):
        result = recommend_for_cold_track(
            1, [12, 15], [11, 14, 10], self.FEATURES, CONFIG, n=5
        )
        assert isinstance(result, list)

    def test_respects_n(self):
        result = recommend_for_cold_track(
            1, [12, 15], [11, 14, 10], self.FEATURES, CONFIG, n=5
        )
        assert len(result) == 5

    def test_warm_tracks_appear_before_cold_tracks(self):
        # cold_fraction=0.20, n=5 → n_cold=ceil(1)=1, n_warm=4
        # warm=[11,14,10], cold ranked by popularity: [12,15] both 0.0 → tie-broken to [12,15]
        result = recommend_for_cold_track(
            1, [12, 15], [11, 14, 10], self.FEATURES, CONFIG, n=5
        )
        # First n_warm slots should come from warm list
        assert result[0] == 11
        assert result[1] == 14
        assert result[2] == 10

    def test_cold_tracks_fill_cold_slots(self):
        # n_cold=1, so exactly one cold track should appear
        result = recommend_for_cold_track(
            1, [12, 15], [11, 14, 10], self.FEATURES, CONFIG, n=5
        )
        cold_in_result = [t for t in result if t in {12, 15}]
        assert len(cold_in_result) >= 1

    def test_no_duplicates(self):
        result = recommend_for_cold_track(
            1, [12, 15], [11, 14, 10], self.FEATURES, CONFIG, n=5
        )
        assert len(result) == len(set(result))

    def test_all_ids_from_input_pools(self):
        warm = [11, 14, 10]
        cold = [12, 15]
        result = recommend_for_cold_track(1, cold, warm, self.FEATURES, CONFIG, n=5)
        assert set(result).issubset(set(warm + cold))

    def test_backfills_when_cold_pool_is_small(self):
        # Only 1 cold track available; remaining slots should backfill from warm
        result = recommend_for_cold_track(
            1, [12], [11, 14, 10], self.FEATURES, CONFIG, n=4
        )
        assert len(result) == 4
        assert 12 in result

    def test_zero_cold_fraction_gives_all_warm(self):
        cfg = {**CONFIG, "cold_start": {**CONFIG["cold_start"], "cold_track_slot_fraction": 0.0}}
        result = recommend_for_cold_track(
            1, [12, 15], [11, 14, 10], self.FEATURES, cfg, n=3
        )
        # n_cold = ceil(0*3)=0, so all 3 slots go to warm
        assert set(result) == {11, 14, 10}
