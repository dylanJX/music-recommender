"""
test_scorer.py
==============
Unit tests for src/scorer.py.

Test strategy:
  - Shared synthetic DataFrames so every test is self-contained (no real files).
  - Signal functions tested individually with known analytic values.
  - Rule functions tested against hand-computed expected scores.
  - write_submission verified for CSV format, TrackID format, and Predictor values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.scorer import (
    album_match_score,
    artist_match_score,
    combine_scores,
    genre_match_score,
    rank_candidates,
    run_all_rules,
    score_rule1,
    score_rule2,
    score_rule3,
    write_submission,
)

# ---------------------------------------------------------------------------
# Shared fixtures
#
# Training interactions:
#   user 1 → track 10  (album 100, artist 200, genres [1, 2])
#   user 1 → track 11  (album 101, artist 201, genres [2, 3])
#   user 2 → track 10  (album 100, artist 200, genres [1, 2])
#
# Genre affinities for user 1:
#   genre 1 appears 1 time, genre 2 appears 2 times, genre 3 appears 1 time
#   total = 4 → {1: 0.25, 2: 0.50, 3: 0.25}
#
# Artist affinities for user 1:
#   artist 200 × 1, artist 201 × 1 → total = 2 → {200: 0.5, 201: 0.5}
#
# Album affinities for user 1:
#   album 100 × 1, album 101 × 1 → total = 2 → {100: 0.5, 101: 0.5}
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
    {"track_id": 13, "album_id": 102, "artist_id": 202, "genre_ids": [4]},  # cold track
])

CONFIG = {
    "scorer": {
        "weights": {
            "genre_match": 0.35,
            "artist_match": 0.30,
            "album_match": 0.15,
            "popularity": 0.15,
            "novelty": 0.05,
        },
        "popularity_log_base": 10,
        "rule1_weights": {"genre_max": 0.60, "artist_score": 0.40},
        "rule2_weights": {
            "album_score": 0.20,
            "artist_score": 0.25,
            "genre_mean": 0.25,
            "cf_svd_score": 0.15,
            "cf_user_user_score": 0.15,
        },
        "rule3_alpha": 0.20,
    },
    "collab_filtering": {"cf_top_k": 20, "n_components": 2},
    "pipeline": {"top_n": 100},
}


def _make_features() -> dict:
    """Build a feature dict from TRAIN + TRACKS using feature_engineering."""
    from src.feature_engineering import (
        build_track_features,
        build_user_profiles,
        compute_artist_affinity,
        compute_genre_affinity,
    )
    albums = pd.DataFrame(columns=["album_id", "artist_id", "genre_ids"])
    user_profiles = build_user_profiles(TRAIN, TRACKS, albums)
    track_features = build_track_features(TRAIN, TRACKS)
    genre_affinity = compute_genre_affinity(user_profiles)
    artist_affinity = compute_artist_affinity(user_profiles)
    return {
        "user_profiles": user_profiles,
        "track_features": track_features,
        "genre_affinity": genre_affinity,
        "artist_affinity": artist_affinity,
    }


# Minimal pair_features DataFrame for rule function tests (no feature_engineering required)
PAIR_FEATURES = pd.DataFrame([
    {
        "user_id": 1, "track_id": 10,
        "genre_max": 0.8, "genre_mean": 0.5, "artist_score": 0.6,
        "album_score": 0.5,
        "cf_svd_score": 0.7, "cf_user_user_score": 0.6,
        "popularity_score": 0.9,
    },
    {
        "user_id": 1, "track_id": 11,
        "genre_max": 0.3, "genre_mean": 0.2, "artist_score": 0.2,
        "album_score": 0.1,
        "cf_svd_score": 0.3, "cf_user_user_score": 0.2,
        "popularity_score": 0.4,
    },
])


# ---------------------------------------------------------------------------
# TestGenreMatchScore
# ---------------------------------------------------------------------------

class TestGenreMatchScore:
    def test_mean_of_known_genres(self):
        aff = pd.Series({1: 0.25, 2: 0.50, 3: 0.25})
        # genres [1, 2] → mean(0.25, 0.50) = 0.375
        result = genre_match_score(aff, [1, 2])
        assert result == pytest.approx(0.375)

    def test_single_known_genre(self):
        aff = pd.Series({1: 0.25, 2: 0.50, 3: 0.25})
        result = genre_match_score(aff, [2])
        assert result == pytest.approx(0.50)

    def test_unknown_genre_returns_zero(self):
        aff = pd.Series({1: 0.25, 2: 0.50})
        # genre 99 not in affinity → 0
        result = genre_match_score(aff, [99])
        assert result == pytest.approx(0.0)

    def test_empty_track_genres_returns_zero(self):
        aff = pd.Series({1: 0.5})
        assert genre_match_score(aff, []) == pytest.approx(0.0)

    def test_empty_user_affinity_returns_zero(self):
        aff = pd.Series(dtype=float)
        assert genre_match_score(aff, [1, 2]) == pytest.approx(0.0)

    def test_partial_overlap_averages_with_zeros(self):
        aff = pd.Series({1: 0.6})
        # genres [1, 99] → mean(0.6, 0.0) = 0.3
        result = genre_match_score(aff, [1, 99])
        assert result == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# TestArtistMatchScore
# ---------------------------------------------------------------------------

class TestArtistMatchScore:
    def test_known_artist(self):
        aff = pd.Series({200: 0.5, 201: 0.5})
        assert artist_match_score(aff, 200) == pytest.approx(0.5)

    def test_unknown_artist_returns_zero(self):
        aff = pd.Series({200: 0.5})
        assert artist_match_score(aff, 999) == pytest.approx(0.0)

    def test_empty_affinity_returns_zero(self):
        aff = pd.Series(dtype=float)
        assert artist_match_score(aff, 200) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestAlbumMatchScore
# ---------------------------------------------------------------------------

class TestAlbumMatchScore:
    def test_known_album(self):
        aff = pd.Series({100: 0.5, 101: 0.5})
        assert album_match_score(aff, 100) == pytest.approx(0.5)

    def test_unknown_album_returns_zero(self):
        aff = pd.Series({100: 0.5})
        assert album_match_score(aff, 999) == pytest.approx(0.0)

    def test_empty_affinity_returns_zero(self):
        aff = pd.Series(dtype=float)
        assert album_match_score(aff, 100) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestCombineScores
# ---------------------------------------------------------------------------

class TestCombineScores:
    def test_basic_weighted_sum(self):
        signals = {"genre_match": 0.5, "artist_match": 1.0}
        weights = {"genre_match": 1.0, "artist_match": 1.0}
        # (1.0*0.5 + 1.0*1.0) / 2.0 = 0.75
        assert combine_scores(signals, weights) == pytest.approx(0.75)

    def test_weights_normalised_internally(self):
        # Doubling all weights should give the same result
        signals = {"a": 0.4, "b": 0.8}
        weights_1 = {"a": 1.0, "b": 1.0}
        weights_2 = {"a": 2.0, "b": 2.0}
        assert combine_scores(signals, weights_1) == pytest.approx(
            combine_scores(signals, weights_2)
        )

    def test_zero_total_weight_returns_zero(self):
        signals = {"a": 0.5}
        weights = {"a": 0.0}
        assert combine_scores(signals, weights) == pytest.approx(0.0)

    def test_signal_not_in_weights_is_ignored(self):
        signals = {"genre_match": 0.5, "unknown_signal": 0.9}
        weights = {"genre_match": 1.0}
        # Only genre_match contributes; unknown_signal has weight 0
        assert combine_scores(signals, weights) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# TestRankCandidates
# ---------------------------------------------------------------------------

class TestRankCandidates:
    def test_returns_all_candidate_ids(self):
        features = _make_features()
        candidates = [12, 13]
        result = rank_candidates(1, candidates, features, CONFIG)
        assert set(result) == {12, 13}
        assert len(result) == 2

    def test_known_affinity_track_ranked_first(self):
        # Track 12: album 100 (user 1 knows it) + artist 200 (user 1 knows it)
        # Track 13: cold album + cold artist → score should be lower
        features = _make_features()
        result = rank_candidates(1, [12, 13], features, CONFIG)
        assert result[0] == 12

    def test_cold_user_returns_all_candidates(self):
        features = _make_features()
        # User 99 has no training history — should still return a ranking
        candidates = [12, 13]
        result = rank_candidates(99, candidates, features, CONFIG)
        assert set(result) == {12, 13}

    def test_single_candidate(self):
        features = _make_features()
        result = rank_candidates(1, [12], features, CONFIG)
        assert result == [12]

    def test_empty_candidates(self):
        features = _make_features()
        assert rank_candidates(1, [], features, CONFIG) == []


# ---------------------------------------------------------------------------
# TestScoreRule1
# ---------------------------------------------------------------------------

class TestScoreRule1:
    def test_formula_row0(self):
        # row0: genre_max=0.8, artist_score=0.6, weights={genre_max:0.6, artist_score:0.4}
        # score = (0.6*0.8 + 0.4*0.6) / 1.0 = 0.48 + 0.24 = 0.72
        scores = score_rule1(PAIR_FEATURES, CONFIG)
        assert scores.iloc[0] == pytest.approx(0.72)

    def test_formula_row1(self):
        # row1: genre_max=0.3, artist_score=0.2
        # score = (0.6*0.3 + 0.4*0.2) / 1.0 = 0.18 + 0.08 = 0.26
        scores = score_rule1(PAIR_FEATURES, CONFIG)
        assert scores.iloc[1] == pytest.approx(0.26)

    def test_nan_treated_as_zero(self):
        df = PAIR_FEATURES.copy()
        df.loc[0, "artist_score"] = float("nan")
        # row0: (0.6*0.8 + 0.4*0.0) / 1.0 = 0.48
        scores = score_rule1(df, CONFIG)
        assert scores.iloc[0] == pytest.approx(0.48)

    def test_missing_column_treated_as_zero(self):
        df = PAIR_FEATURES.drop(columns=["genre_max"])
        # genre_max treated as 0; artist_score weight still applies
        # row0: (0.6*0.0 + 0.4*0.6) / 1.0 = 0.24
        scores = score_rule1(df, CONFIG)
        assert scores.iloc[0] == pytest.approx(0.24)

    def test_returns_series_same_length(self):
        scores = score_rule1(PAIR_FEATURES, CONFIG)
        assert len(scores) == len(PAIR_FEATURES)

    def test_higher_score_for_row0(self):
        scores = score_rule1(PAIR_FEATURES, CONFIG)
        assert scores.iloc[0] > scores.iloc[1]


# ---------------------------------------------------------------------------
# TestScoreRule2
# ---------------------------------------------------------------------------

class TestScoreRule2:
    def test_formula_row0(self):
        # weights: album=0.20, artist=0.25, genre_mean=0.25, cf_svd=0.15, cf_uu=0.15
        # total_w = 1.0
        # row0: 0.20*0.5 + 0.25*0.6 + 0.25*0.5 + 0.15*0.7 + 0.15*0.6
        #      = 0.10 + 0.15 + 0.125 + 0.105 + 0.09 = 0.57
        scores = score_rule2(PAIR_FEATURES, CONFIG)
        assert scores.iloc[0] == pytest.approx(0.57)

    def test_formula_row1(self):
        # row1: 0.20*0.1 + 0.25*0.2 + 0.25*0.2 + 0.15*0.3 + 0.15*0.2
        #      = 0.02 + 0.05 + 0.05 + 0.045 + 0.03 = 0.195
        scores = score_rule2(PAIR_FEATURES, CONFIG)
        assert scores.iloc[1] == pytest.approx(0.195)

    def test_missing_cf_columns_treated_as_zero(self):
        df = PAIR_FEATURES.drop(columns=["cf_svd_score", "cf_user_user_score"])
        # Missing CF columns become 0; same total_w normalization applies
        # row0: (0.20*0.5 + 0.25*0.6 + 0.25*0.5 + 0.15*0 + 0.15*0) / 1.0
        #      = 0.10 + 0.15 + 0.125 = 0.375
        scores = score_rule2(df, CONFIG)
        assert scores.iloc[0] == pytest.approx(0.375)

    def test_nan_treated_as_zero(self):
        df = PAIR_FEATURES.copy()
        df.loc[0, "album_score"] = float("nan")
        scores_nan = score_rule2(df, CONFIG)
        scores_orig = score_rule2(PAIR_FEATURES, CONFIG)
        # row0 with album_score=0 should be lower than original (album_score=0.5)
        assert scores_nan.iloc[0] < scores_orig.iloc[0]

    def test_returns_series_same_length(self):
        scores = score_rule2(PAIR_FEATURES, CONFIG)
        assert len(scores) == len(PAIR_FEATURES)


# ---------------------------------------------------------------------------
# TestScoreRule3
# ---------------------------------------------------------------------------

class TestScoreRule3:
    def test_alpha_zero_equals_rule2(self):
        cfg = {**CONFIG, "scorer": {**CONFIG["scorer"], "rule3_alpha": 0.0}}
        r2 = score_rule2(PAIR_FEATURES, cfg)
        r3 = score_rule3(PAIR_FEATURES, cfg)
        pd.testing.assert_series_equal(r3, r2)

    def test_alpha_one_equals_popularity(self):
        cfg = {**CONFIG, "scorer": {**CONFIG["scorer"], "rule3_alpha": 1.0}}
        pop = PAIR_FEATURES["popularity_score"].fillna(0.0)
        r3 = score_rule3(PAIR_FEATURES, cfg)
        np.testing.assert_allclose(r3.values, pop.values)

    def test_default_alpha_blends_correctly(self):
        # alpha=0.2: score = 0.8 * rule2 + 0.2 * popularity
        r2 = score_rule2(PAIR_FEATURES, CONFIG)
        pop = PAIR_FEATURES["popularity_score"].fillna(0.0)
        expected = 0.8 * r2 + 0.2 * pop
        r3 = score_rule3(PAIR_FEATURES, CONFIG)
        np.testing.assert_allclose(r3.values, expected.values, rtol=1e-6)

    def test_missing_popularity_treated_as_zero(self):
        df = PAIR_FEATURES.drop(columns=["popularity_score"])
        r2 = score_rule2(df, CONFIG)
        r3 = score_rule3(df, CONFIG)
        # alpha=0.2, pop=0 → score = 0.8 * rule2
        np.testing.assert_allclose(r3.values, 0.8 * r2.values, rtol=1e-6)

    def test_alpha_clamped_above_one(self):
        cfg = {**CONFIG, "scorer": {**CONFIG["scorer"], "rule3_alpha": 5.0}}
        # Should clamp to 1.0 → returns popularity only
        pop = PAIR_FEATURES["popularity_score"].fillna(0.0)
        r3 = score_rule3(PAIR_FEATURES, cfg)
        np.testing.assert_allclose(r3.values, pop.values)


# ---------------------------------------------------------------------------
# TestWriteSubmission
# ---------------------------------------------------------------------------

class TestWriteSubmission:
    def test_creates_file(self, tmp_path):
        pairs = pd.DataFrame({"user_id": [1, 1], "track_id": [10, 11]})
        scores = pd.Series([0.9, 0.3])
        out = tmp_path / "sub.csv"
        write_submission(pairs, scores, out, top_n=1)
        assert out.exists()

    def test_columns_are_trackid_and_predictor(self, tmp_path):
        pairs = pd.DataFrame({"user_id": [1, 1], "track_id": [10, 11]})
        scores = pd.Series([0.9, 0.3])
        out = tmp_path / "sub.csv"
        write_submission(pairs, scores, out, top_n=1)
        df = pd.read_csv(out)
        assert list(df.columns) == ["TrackID", "Predictor"]

    def test_trackid_format(self, tmp_path):
        pairs = pd.DataFrame({"user_id": [1, 1], "track_id": [10, 11]})
        scores = pd.Series([0.9, 0.3])
        out = tmp_path / "sub.csv"
        write_submission(pairs, scores, out, top_n=2)
        df = pd.read_csv(out)
        assert set(df["TrackID"]) == {"1_10", "1_11"}

    def test_top_n_marks_correct_predictor(self, tmp_path):
        pairs = pd.DataFrame({"user_id": [1, 1, 1], "track_id": [10, 11, 12]})
        scores = pd.Series([0.9, 0.5, 0.1])
        out = tmp_path / "sub.csv"
        write_submission(pairs, scores, out, top_n=2)
        df = pd.read_csv(out).set_index("TrackID")
        # Top 2: track 10 (0.9) and track 11 (0.5) → Predictor=1
        assert df.loc["1_10", "Predictor"] == 1
        assert df.loc["1_11", "Predictor"] == 1
        assert df.loc["1_12", "Predictor"] == 0

    def test_top_n_per_user_independent(self, tmp_path):
        pairs = pd.DataFrame({
            "user_id": [1, 1, 2, 2],
            "track_id": [10, 11, 10, 11],
        })
        scores = pd.Series([0.9, 0.1, 0.2, 0.8])
        out = tmp_path / "sub.csv"
        write_submission(pairs, scores, out, top_n=1)
        df = pd.read_csv(out).set_index("TrackID")
        # User 1: top track is 10 (0.9); user 2: top track is 11 (0.8)
        assert df.loc["1_10", "Predictor"] == 1
        assert df.loc["1_11", "Predictor"] == 0
        assert df.loc["2_11", "Predictor"] == 1
        assert df.loc["2_10", "Predictor"] == 0

    def test_creates_parent_directory(self, tmp_path):
        pairs = pd.DataFrame({"user_id": [1], "track_id": [10]})
        scores = pd.Series([0.5])
        out = tmp_path / "new_dir" / "sub.csv"
        write_submission(pairs, scores, out, top_n=1)
        assert out.exists()


# ---------------------------------------------------------------------------
# TestRunAllRules
# ---------------------------------------------------------------------------

class TestRunAllRules:
    def test_creates_three_submission_files(self, tmp_path):
        features = _make_features()
        pairs = pd.DataFrame({
            "user_id": [1, 1, 2, 2],
            "track_id": [12, 13, 12, 13],
        })
        run_all_rules(pairs, TRAIN, features, CONFIG, submissions_dir=str(tmp_path))
        assert (tmp_path / "submission_rule1.csv").exists()
        assert (tmp_path / "submission_rule2.csv").exists()
        assert (tmp_path / "submission_rule3.csv").exists()

    def test_submission_files_have_correct_columns(self, tmp_path):
        features = _make_features()
        pairs = pd.DataFrame({
            "user_id": [1, 1, 2, 2],
            "track_id": [12, 13, 12, 13],
        })
        run_all_rules(pairs, TRAIN, features, CONFIG, submissions_dir=str(tmp_path))
        for name in ("submission_rule1.csv", "submission_rule2.csv", "submission_rule3.csv"):
            df = pd.read_csv(tmp_path / name)
            assert list(df.columns) == ["TrackID", "Predictor"]
            assert len(df) == len(pairs)

    def test_submission_predictor_is_binary(self, tmp_path):
        features = _make_features()
        pairs = pd.DataFrame({
            "user_id": [1, 1, 2, 2],
            "track_id": [12, 13, 12, 13],
        })
        run_all_rules(pairs, TRAIN, features, CONFIG, submissions_dir=str(tmp_path))
        for name in ("submission_rule1.csv", "submission_rule2.csv", "submission_rule3.csv"):
            df = pd.read_csv(tmp_path / name)
            assert set(df["Predictor"]).issubset({0, 1})
