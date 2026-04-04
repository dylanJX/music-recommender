"""
test_collab_features.py
=======================
Unit tests for src/collab_features.py.

Test strategy:
  - Small mock interaction matrix with known structure so similarity values
    can be derived analytically.
  - Verify output shape, column names, value ranges, cold-start behaviour,
    and directional correctness (similar users/items score higher).

Fixture interaction matrix
--------------------------
         t10  t11  t12  t13
  user1:   1    1    0    0
  user2:   1    1    1    0
  user3:   0    0    1    1

  user4 = cold user (absent from TRAIN).
  track14 = cold track (absent from TRAIN).

Derived similarities (for reference):
  cosine(user1, user2) = 2/sqrt(6) ≈ 0.8165   (share tracks 10 & 11)
  cosine(user1, user3) = 0                     (no common tracks)
  cosine(track12, track10) = 1/2               (share user2 only)
  cosine(track12, track11) = 1/2               (share user2 only)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.collab_features import build_interaction_matrix, compute_cf_features


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TRAIN = pd.DataFrame([
    {"user_id": 1, "track_id": 10},
    {"user_id": 1, "track_id": 11},
    {"user_id": 2, "track_id": 10},
    {"user_id": 2, "track_id": 11},
    {"user_id": 2, "track_id": 12},
    {"user_id": 3, "track_id": 12},
    {"user_id": 3, "track_id": 13},
])

CONFIG = {
    "collab_filtering": {
        "cf_top_k": 20,
        "n_components": 2,
    }
}

EXPECTED_CF_COLS = {"user_id", "track_id",
                    "cf_user_user_score", "cf_item_item_score", "cf_svd_score"}


def _score_for(result: pd.DataFrame, uid: int, tid: int, col: str) -> float:
    row = result[(result["user_id"] == uid) & (result["track_id"] == tid)]
    return float(row.iloc[0][col])


# ---------------------------------------------------------------------------
# build_interaction_matrix
# ---------------------------------------------------------------------------

class TestBuildInteractionMatrix:
    def test_shape(self):
        """Matrix must have shape (n_unique_users, n_unique_tracks)."""
        mat, u2i, t2i = build_interaction_matrix(TRAIN)
        assert mat.shape == (3, 4)

    def test_user_index_keys(self):
        """user_to_idx must cover exactly the users in TRAIN."""
        _, u2i, _ = build_interaction_matrix(TRAIN)
        assert set(u2i.keys()) == {1, 2, 3}

    def test_track_index_keys(self):
        """track_to_idx must cover exactly the tracks in TRAIN."""
        _, _, t2i = build_interaction_matrix(TRAIN)
        assert set(t2i.keys()) == {10, 11, 12, 13}

    def test_interaction_count(self):
        """Entry (user2, track10) must equal 1 (appeared once)."""
        mat, u2i, t2i = build_interaction_matrix(TRAIN)
        assert mat[u2i[2], t2i[10]] == 1.0

    def test_duplicate_interactions_summed(self):
        """Duplicate (user, track) rows must be summed into a single count."""
        train = pd.DataFrame([
            {"user_id": 1, "track_id": 10},
            {"user_id": 1, "track_id": 10},
        ])
        mat, u2i, t2i = build_interaction_matrix(train)
        assert mat[u2i[1], t2i[10]] == 2.0

    def test_zero_for_unseen_pair(self):
        """User1 never interacted with track12 → entry must be 0."""
        mat, u2i, t2i = build_interaction_matrix(TRAIN)
        assert mat[u2i[1], t2i[12]] == 0.0

    def test_empty_train_returns_empty_matrix(self):
        """Empty training set must return a (0, 0) matrix and empty dicts."""
        mat, u2i, t2i = build_interaction_matrix(pd.DataFrame(columns=["user_id", "track_id"]))
        assert mat.shape == (0, 0)
        assert u2i == {}
        assert t2i == {}


# ---------------------------------------------------------------------------
# compute_cf_features — output structure
# ---------------------------------------------------------------------------

class TestCfFeaturesStructure:
    def setup_method(self):
        self.pairs = pd.DataFrame([
            {"user_id": 1, "track_id": 12},
            {"user_id": 2, "track_id": 13},
            {"user_id": 3, "track_id": 10},
        ])
        self.result = compute_cf_features(self.pairs, TRAIN, CONFIG)

    def test_shape(self):
        """Output must have one row per input pair."""
        assert len(self.result) == len(self.pairs)

    def test_required_columns(self):
        """All expected CF columns must be present."""
        assert EXPECTED_CF_COLS.issubset(set(self.result.columns))

    def test_no_nan_values(self):
        """No NaN values in any CF score column (cold start returns 0.0)."""
        for col in ["cf_user_user_score", "cf_item_item_score", "cf_svd_score"]:
            assert self.result[col].notna().all(), f"{col} contains NaN"

    def test_scores_in_range(self):
        """All CF scores must lie in [0, 1]."""
        for col in ["cf_user_user_score", "cf_item_item_score", "cf_svd_score"]:
            assert (self.result[col] >= 0.0).all(), f"{col} has values < 0"
            assert (self.result[col] <= 1.0).all(), f"{col} has values > 1"


# ---------------------------------------------------------------------------
# compute_cf_features — cold-start behaviour
# ---------------------------------------------------------------------------

class TestColdStart:
    def test_cold_user_uu_is_zero(self):
        """User absent from training → cf_user_user_score == 0.0."""
        pairs = pd.DataFrame([{"user_id": 4, "track_id": 10}])
        result = compute_cf_features(pairs, TRAIN, CONFIG)
        assert _score_for(result, 4, 10, "cf_user_user_score") == 0.0

    def test_cold_user_ii_is_zero(self):
        """User absent from training → cf_item_item_score == 0.0."""
        pairs = pd.DataFrame([{"user_id": 4, "track_id": 10}])
        result = compute_cf_features(pairs, TRAIN, CONFIG)
        assert _score_for(result, 4, 10, "cf_item_item_score") == 0.0

    def test_cold_user_svd_is_zero(self):
        """User absent from training → cf_svd_score == 0.0."""
        pairs = pd.DataFrame([{"user_id": 4, "track_id": 10}])
        result = compute_cf_features(pairs, TRAIN, CONFIG)
        assert _score_for(result, 4, 10, "cf_svd_score") == 0.0

    def test_cold_track_uu_is_zero(self):
        """Track absent from training → cf_user_user_score == 0.0."""
        pairs = pd.DataFrame([{"user_id": 1, "track_id": 14}])
        result = compute_cf_features(pairs, TRAIN, CONFIG)
        assert _score_for(result, 1, 14, "cf_user_user_score") == 0.0

    def test_cold_track_ii_is_zero(self):
        """Track absent from training → cf_item_item_score == 0.0."""
        pairs = pd.DataFrame([{"user_id": 1, "track_id": 14}])
        result = compute_cf_features(pairs, TRAIN, CONFIG)
        assert _score_for(result, 1, 14, "cf_item_item_score") == 0.0

    def test_cold_track_svd_is_zero(self):
        """Track absent from training → cf_svd_score == 0.0."""
        pairs = pd.DataFrame([{"user_id": 1, "track_id": 14}])
        result = compute_cf_features(pairs, TRAIN, CONFIG)
        assert _score_for(result, 1, 14, "cf_svd_score") == 0.0

    def test_empty_train_returns_zeros(self):
        """Empty training set → all CF scores are 0.0."""
        pairs = pd.DataFrame([{"user_id": 1, "track_id": 10}])
        result = compute_cf_features(pairs, pd.DataFrame(columns=["user_id", "track_id"]), CONFIG)
        for col in ["cf_user_user_score", "cf_item_item_score", "cf_svd_score"]:
            assert result.iloc[0][col] == 0.0, f"{col} should be 0.0 for empty train"


# ---------------------------------------------------------------------------
# compute_cf_features — user-user CF correctness
# ---------------------------------------------------------------------------

class TestUserUserCF:
    def test_similar_user_track_scores_positive(self):
        """User1 and User2 are highly similar; User2 listened to track12.
        Therefore (user1, track12) should have a positive uu score.
        """
        pairs = pd.DataFrame([{"user_id": 1, "track_id": 12}])
        result = compute_cf_features(pairs, TRAIN, CONFIG)
        score = _score_for(result, 1, 12, "cf_user_user_score")
        assert score > 0.0

    def test_uu_score_is_one_when_only_similar_neighbour_interacted(self):
        """User1's only positively-similar neighbour is User2 (sim ≈ 0.816).
        User2 interacted with track12 (binary = 1).
        Weighted average = (0.816 * 1) / 0.816 = 1.0.
        """
        pairs = pd.DataFrame([{"user_id": 1, "track_id": 12}])
        result = compute_cf_features(pairs, TRAIN, CONFIG)
        score = _score_for(result, 1, 12, "cf_user_user_score")
        assert score == pytest.approx(1.0)

    def test_dissimilar_user_track_scores_zero(self):
        """User1 and User3 share no tracks → cosine similarity == 0.
        User3 is the only user who interacted with track13.
        (user1, track13): no neighbour has positive similarity → uu score == 0.
        """
        pairs = pd.DataFrame([{"user_id": 1, "track_id": 13}])
        result = compute_cf_features(pairs, TRAIN, CONFIG)
        score = _score_for(result, 1, 13, "cf_user_user_score")
        assert score == pytest.approx(0.0)

    def test_single_user_train_uu_is_zero(self):
        """With only one user in training there are no neighbours → uu score == 0."""
        train = pd.DataFrame([
            {"user_id": 1, "track_id": 10},
            {"user_id": 1, "track_id": 11},
        ])
        pairs = pd.DataFrame([{"user_id": 1, "track_id": 10}])
        result = compute_cf_features(pairs, train, CONFIG)
        assert _score_for(result, 1, 10, "cf_user_user_score") == 0.0


# ---------------------------------------------------------------------------
# compute_cf_features — item-item CF correctness
# ---------------------------------------------------------------------------

class TestItemItemCF:
    def test_cooccurring_items_score_positive(self):
        """Track12 and Track10 both appear in User2's history.
        cosine(track12, track10) = 0.5 > 0.
        (user1, track12): user1 listened to track10 and track11,
        both of which have cosine_sim 0.5 with track12 → ii score > 0.
        """
        pairs = pd.DataFrame([{"user_id": 1, "track_id": 12}])
        result = compute_cf_features(pairs, TRAIN, CONFIG)
        score = _score_for(result, 1, 12, "cf_item_item_score")
        assert score > 0.0

    def test_ii_exact_value(self):
        """cosine(track12, track10) == cosine(track12, track11) == 0.5.
        User1's history (excluding candidate) is {track10, track11}.
        ii score = mean(0.5, 0.5) = 0.5.
        """
        pairs = pd.DataFrame([{"user_id": 1, "track_id": 12}])
        result = compute_cf_features(pairs, TRAIN, CONFIG)
        score = _score_for(result, 1, 12, "cf_item_item_score")
        assert score == pytest.approx(0.5)

    def test_no_common_users_scores_zero(self):
        """Track13 is only listened to by User3.  User1 has no history overlap
        with User3 → cosine(track13, track10) == cosine(track13, track11) == 0
        → ii score == 0.
        """
        pairs = pd.DataFrame([{"user_id": 1, "track_id": 13}])
        result = compute_cf_features(pairs, TRAIN, CONFIG)
        score = _score_for(result, 1, 13, "cf_item_item_score")
        assert score == pytest.approx(0.0)

    def test_user_with_single_interaction_no_error(self):
        """A user with only one training interaction does not raise and returns
        a valid score.
        """
        train = pd.DataFrame([
            {"user_id": 1, "track_id": 10},   # only one interaction
            {"user_id": 2, "track_id": 10},
            {"user_id": 2, "track_id": 11},
        ])
        pairs = pd.DataFrame([{"user_id": 1, "track_id": 11}])
        result = compute_cf_features(pairs, train, CONFIG)
        score = _score_for(result, 1, 11, "cf_item_item_score")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# compute_cf_features — SVD correctness
# ---------------------------------------------------------------------------

class TestSVDCF:
    def test_svd_score_is_finite(self):
        """SVD scores must be finite (not inf or NaN) for warm pairs."""
        pairs = pd.DataFrame([
            {"user_id": 1, "track_id": 10},
            {"user_id": 2, "track_id": 12},
            {"user_id": 3, "track_id": 13},
        ])
        result = compute_cf_features(pairs, TRAIN, CONFIG)
        assert np.isfinite(result["cf_svd_score"].values).all()

    def test_svd_score_in_range(self):
        """SVD scores must be in [0, 1] for all warm pairs."""
        pairs = pd.DataFrame([
            {"user_id": 1, "track_id": 10},
            {"user_id": 1, "track_id": 11},
            {"user_id": 2, "track_id": 10},
            {"user_id": 3, "track_id": 12},
        ])
        result = compute_cf_features(pairs, TRAIN, CONFIG)
        assert (result["cf_svd_score"] >= 0.0).all()
        assert (result["cf_svd_score"] <= 1.0).all()

    def test_svd_matrix_too_small_returns_zero(self):
        """When n_comp < 1 (matrix has only 1 user or 1 track), SVD cannot be
        fitted and the score falls back to 0.0.
        """
        train = pd.DataFrame([{"user_id": 1, "track_id": 10}])
        pairs = pd.DataFrame([{"user_id": 1, "track_id": 10}])
        result = compute_cf_features(pairs, train, CONFIG)
        # n_comp = min(2, 1-1, 1-1) = 0 → SVD skipped → 0.0
        assert _score_for(result, 1, 10, "cf_svd_score") == 0.0

    def test_svd_n_components_respected(self):
        """Reducing n_components to 1 must still produce valid [0,1] scores."""
        cfg = {"collab_filtering": {"cf_top_k": 20, "n_components": 1}}
        pairs = pd.DataFrame([{"user_id": 1, "track_id": 12}])
        result = compute_cf_features(pairs, TRAIN, cfg)
        score = _score_for(result, 1, 12, "cf_svd_score")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# compute_cf_features — sparse / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_sparse_matrix_all_pairs_score(self):
        """All warm (user, track) pairs must produce a valid float score."""
        pairs = pd.DataFrame([
            {"user_id": uid, "track_id": tid}
            for uid in [1, 2, 3]
            for tid in [10, 11, 12, 13]
        ])
        result = compute_cf_features(pairs, TRAIN, CONFIG)
        assert len(result) == 12
        for col in ["cf_user_user_score", "cf_item_item_score", "cf_svd_score"]:
            assert result[col].between(0.0, 1.0).all()

    def test_duplicate_pairs_handled(self):
        """Duplicate rows in user_track_pairs produce independent output rows."""
        pairs = pd.DataFrame([
            {"user_id": 1, "track_id": 12},
            {"user_id": 1, "track_id": 12},
        ])
        result = compute_cf_features(pairs, TRAIN, CONFIG)
        assert len(result) == 2
        assert result.iloc[0]["cf_user_user_score"] == result.iloc[1]["cf_user_user_score"]

    def test_mixed_warm_cold_pairs(self):
        """A batch mixing warm and cold pairs returns correct 0.0 for cold."""
        pairs = pd.DataFrame([
            {"user_id": 1, "track_id": 12},   # warm pair
            {"user_id": 4, "track_id": 12},   # cold user
            {"user_id": 1, "track_id": 14},   # cold track
        ])
        result = compute_cf_features(pairs, TRAIN, CONFIG)
        assert result.iloc[0]["cf_user_user_score"] > 0.0
        assert result.iloc[1]["cf_user_user_score"] == 0.0
        assert result.iloc[2]["cf_user_user_score"] == 0.0

    def test_cf_top_k_larger_than_users_no_error(self):
        """cf_top_k larger than the number of training users must not crash."""
        cfg = {"collab_filtering": {"cf_top_k": 1000, "n_components": 2}}
        pairs = pd.DataFrame([{"user_id": 1, "track_id": 12}])
        result = compute_cf_features(pairs, TRAIN, cfg)
        score = _score_for(result, 1, 12, "cf_user_user_score")
        assert 0.0 <= score <= 1.0
