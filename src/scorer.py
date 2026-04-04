"""
scorer.py  (Step 1b)
====================
Rule-based ranking module.  Given a user and a candidate set of tracks,
produces a ranked list by combining multiple weighted signals into a single
score.

Design goals:
  - All weights and thresholds are read from config.yaml so that tuning
    never requires code changes.
  - Each signal component is computed in its own function to keep the
    contribution of each factor auditable and testable in isolation.
  - The final score is a weighted linear combination; non-linear transforms
    (e.g. log-damping of popularity) are applied per signal before combining.

Signals (weights configured in config.yaml under scorer.weights):
  - genre_match_score   : affinity between user's top genres and track's genres
  - artist_match_score  : affinity between user's top artists and track's artist
  - album_match_score   : affinity between user's top albums and track's album
  - popularity_score    : global track popularity (log-damped)
  - novelty_score       : inverse of how many times user already heard this track
                          (always 1 for unseen tracks; used to de-rank re-listens
                           if the task requires recommending new items)

Three additional scoring rules operate on a pre-merged pair_features DataFrame
(columns from compute_user_track_features + compute_cf_features + track_features):

  Rule 1 — Max Genre Focus
      genre_max * w1 + artist_score * w2
      Weights: scorer.rule1_weights in config.yaml.

  Rule 2 — Weighted Hybrid (Content + CF)
      album_score * w1 + artist_score * w2 + genre_mean * w3
          + cf_svd_score * w4 + cf_user_user_score * w5
      Weights: scorer.rule2_weights in config.yaml.

  Rule 3 — Popularity Boosted Hybrid
      rule2_score * (1 - alpha) + popularity_score * alpha
      Alpha:   scorer.rule3_alpha in config.yaml (default 0.20).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual signal functions
# ---------------------------------------------------------------------------

def genre_match_score(
    user_genre_affinity: pd.Series,
    track_genres: list[int],
) -> float:
    """Compute how well a track's genres align with a user's genre preferences.

    Parameters
    ----------
    user_genre_affinity : pd.Series
        Series indexed by genre_id, values in [0, 1].
    track_genres : list[int]
        List of genre IDs associated with the track.

    Returns
    -------
    float in [0, 1]
        Mean user affinity over all of the track's genres.
        Returns 0.0 when the track has no genres or the user has no genre history.
    """
    if not track_genres:
        return 0.0
    scores = [
        float(user_genre_affinity[gid]) if gid in user_genre_affinity.index else 0.0
        for gid in track_genres
    ]
    return float(np.mean(scores))


def artist_match_score(
    user_artist_affinity: pd.Series,
    artist_id: int,
) -> float:
    """Compute how well a track's artist aligns with a user's artist preferences.

    Parameters
    ----------
    user_artist_affinity : pd.Series
        Series indexed by artist_id, values in [0, 1].
    artist_id : int
        Artist of the candidate track.

    Returns
    -------
    float in [0, 1]
    """
    return float(user_artist_affinity[artist_id]) if artist_id in user_artist_affinity.index else 0.0


def album_match_score(
    user_album_affinity: pd.Series,
    album_id: int,
) -> float:
    """Compute how well a track's album aligns with a user's album preferences.

    Parameters
    ----------
    user_album_affinity : pd.Series
        Series indexed by album_id, values in [0, 1].
    album_id : int
        Album of the candidate track.

    Returns
    -------
    float in [0, 1]
    """
    return float(user_album_affinity[album_id]) if album_id in user_album_affinity.index else 0.0


def combine_scores(
    signals: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Compute the final weighted linear combination of all signal scores.

    Parameters
    ----------
    signals : dict[str, float]
        Mapping of signal name → raw score value.
    weights : dict[str, float]
        Mapping of signal name → weight.  Weights are normalised internally so
        they do not need to sum to 1.

    Returns
    -------
    float — final composite score for a (user, track) pair.
    """
    total_w = sum(weights.get(k, 0.0) for k in signals)
    if total_w == 0.0:
        return 0.0
    return float(
        sum(weights.get(k, 0.0) * v for k, v in signals.items()) / total_w
    )


# ---------------------------------------------------------------------------
# Original rank_candidates entry point  (uses scorer.weights from config)
# ---------------------------------------------------------------------------

def rank_candidates(
    user_id: int,
    candidate_track_ids: list[int],
    features: dict[str, Any],
    config: dict[str, Any],
) -> list[int]:
    """Rank a list of candidate tracks for a given user.

    Uses the five weighted signals in ``config['scorer']['weights']``:
    genre_match, artist_match, album_match, popularity (log-damped), novelty.
    Novelty is always 1.0 because the pipeline only passes unseen tracks.

    Parameters
    ----------
    user_id : int
        The user to rank for.
    candidate_track_ids : list[int]
        Pool of tracks to score and sort.
    features : dict[str, Any]
        Feature artifact dict produced by feature_engineering.run().
        Expected keys: 'user_profiles', 'track_features', 'genre_affinity',
        'artist_affinity'.
    config : dict[str, Any]
        Parsed config.yaml dict (passed through from pipeline.py).

    Returns
    -------
    list[int]
        Track IDs sorted descending by composite score.
    """
    scorer_cfg = config.get("scorer", {})
    weights = scorer_cfg.get("weights", {})
    log_base = float(scorer_cfg.get("popularity_log_base", 10))

    user_profiles = features.get("user_profiles", pd.DataFrame())
    track_features = features.get("track_features", pd.DataFrame())
    genre_affinity = features.get("genre_affinity", pd.DataFrame())
    artist_affinity = features.get("artist_affinity", pd.DataFrame())

    # ---- User affinities -------------------------------------------------
    if not genre_affinity.empty and user_id in genre_affinity.index:
        user_genre_aff: pd.Series = genre_affinity.loc[user_id]
    else:
        user_genre_aff = pd.Series(dtype=float)

    if not artist_affinity.empty and user_id in artist_affinity.index:
        user_artist_aff: pd.Series = artist_affinity.loc[user_id]
    else:
        user_artist_aff = pd.Series(dtype=float)

    # Album affinity: normalised interaction share per album for this user
    if not user_profiles.empty:
        user_row = user_profiles[user_profiles["user_id"] == user_id]
    else:
        user_row = pd.DataFrame()

    if not user_row.empty:
        raw_album_counts: dict[int, int] = user_row.iloc[0].get("album_counts") or {}
        total = sum(raw_album_counts.values())
        user_album_aff: pd.Series = (
            pd.Series({k: v / total for k, v in raw_album_counts.items()})
            if total > 0 else pd.Series(dtype=float)
        )
    else:
        user_album_aff = pd.Series(dtype=float)

    # ---- Log-damped popularity range ------------------------------------
    if not track_features.empty and "global_play_count" in track_features.columns:
        max_play = int(track_features["global_play_count"].max())
    else:
        max_play = 0
    log_max = math.log(1 + max_play, log_base) if max_play > 0 else 1.0

    # Index track_features once for O(1) lookups
    tf_indexed = (
        track_features.set_index("track_id")
        if not track_features.empty and "track_id" in track_features.columns
        else pd.DataFrame()
    )

    # ---- Score each candidate -------------------------------------------
    scores: dict[int, float] = {}
    for tid in candidate_track_ids:
        if not tf_indexed.empty and tid in tf_indexed.index:
            tr = tf_indexed.loc[tid]
            genres: list[int] = tr.get("genre_ids") or []
            artist = int(tr.get("artist_id", -1))
            album = int(tr.get("album_id", -1))
            play_count = int(tr.get("global_play_count", 0))
        else:
            genres, artist, album, play_count = [], -1, -1, 0

        g_score = genre_match_score(user_genre_aff, genres)
        a_score = artist_match_score(user_artist_aff, artist)
        al_score = album_match_score(user_album_aff, album)
        p_score = math.log(1 + play_count, log_base) / log_max
        n_score = 1.0  # novelty is always 1 for unseen candidates

        signals = {
            "genre_match": g_score,
            "artist_match": a_score,
            "album_match": al_score,
            "popularity": p_score,
            "novelty": n_score,
        }
        scores[tid] = combine_scores(signals, weights)

    return sorted(candidate_track_ids, key=lambda t: scores.get(t, 0.0), reverse=True)


# ---------------------------------------------------------------------------
# Rule 1 — Max Genre Focus
# ---------------------------------------------------------------------------

def score_rule1(pair_features: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    """
    Rule 1 — hw4 Baseline (Content Only)
    =====================================
    Mirrors hw4.py's ``score_candidate()`` weighting exactly using IDF-weighted
    preference signals pre-computed by feature_engineering.compute_hw4_features():

        score = 0.45 * hw4_track_score
              + 0.10 * hw4_album_score
              + 0.25 * hw4_artist_score
              + 0.15 * hw4_genre_score
              + 0.05 * hw4_pop_score

    Weights: scorer.rule1_weights in config.yaml (defaults match hw4.py).

    Parameters
    ----------
    pair_features : pd.DataFrame
        Columns used: hw4_track_score, hw4_album_score, hw4_artist_score,
        hw4_genre_score, hw4_pop_score.  Missing columns or NaN → 0.
    config : dict
        Parsed config.yaml.

    Returns
    -------
    pd.Series
        Float scores with the same index as pair_features.
    """
    defaults: dict[str, float] = {
        "hw4_track_score": 0.45,
        "hw4_album_score": 0.10,
        "hw4_artist_score": 0.25,
        "hw4_genre_score": 0.15,
        "hw4_pop_score": 0.05,
    }
    w = {**defaults, **config.get("scorer", {}).get("rule1_weights", {})}

    def _col(name: str) -> pd.Series:
        if name in pair_features.columns:
            return pair_features[name].fillna(0.0)
        return pd.Series(np.zeros(len(pair_features)), index=pair_features.index)

    total_w = sum(w.values())
    if total_w == 0.0:
        return pd.Series(np.zeros(len(pair_features)), index=pair_features.index)

    return (sum(w[k] * _col(k) for k in w) / total_w).astype(float)


# ---------------------------------------------------------------------------
# Rule 2 — Weighted Hybrid (Content + CF)
# ---------------------------------------------------------------------------

def score_rule2(pair_features: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    """
    Rule 2 — Weighted Hybrid (hw4 Content + CF)
    =============================================
    Blends the hw4 baseline content score (Rule 1) with collaborative-filtering
    signals. The content score is per-user min-max normalised to [0, 1] before
    mixing so that the unbounded hw4 raw values and the already-[0,1]-bounded CF
    scores contribute with their configured weights.

    Formula:
        content_norm = per-user min-max normalised Rule 1 score
        score = w_content * content_norm
              + w_svd     * cf_svd_score
              + w_uu      * cf_user_user_score

    Weights: scorer.rule2_weights in config.yaml
             (defaults: w_content=0.70, w_svd=0.15, w_uu=0.15)

    Parameters
    ----------
    pair_features : pd.DataFrame
        Must contain the hw4_* columns required by score_rule1, plus
        cf_svd_score and cf_user_user_score (missing columns → 0).
    config : dict
        Parsed config.yaml.

    Returns
    -------
    pd.Series
        Float scores with the same index as pair_features.
    """
    defaults: dict[str, float] = {
        "w_content": 0.70,
        "w_svd": 0.15,
        "w_uu": 0.15,
    }
    w = {**defaults, **config.get("scorer", {}).get("rule2_weights", {})}

    # Raw content score (hw4 baseline)
    content_raw = score_rule1(pair_features, config)

    # Per-user min-max normalisation so content and CF signals are on [0, 1]
    uid = pair_features["user_id"].values
    content_vals = content_raw.values
    user_ids_unique, inverse = np.unique(uid, return_inverse=True)
    u_min = np.zeros(len(user_ids_unique))
    u_max = np.ones(len(user_ids_unique))
    for i in range(len(user_ids_unique)):
        mask = inverse == i
        u_min[i] = content_vals[mask].min()
        u_max[i] = content_vals[mask].max()
    content_min = u_min[inverse]
    content_max = u_max[inverse]
    content_range = np.where(content_max > content_min, content_max - content_min, 1.0)
    content_norm = pd.Series(
        (content_vals - content_min) / content_range,
        index=pair_features.index,
    )

    def _col(name: str) -> pd.Series:
        if name in pair_features.columns:
            return pair_features[name].fillna(0.0)
        return pd.Series(np.zeros(len(pair_features)), index=pair_features.index)

    total_w = w["w_content"] + w["w_svd"] + w["w_uu"]
    if total_w == 0.0:
        return content_norm.astype(float)

    return (
        (w["w_content"] * content_norm
         + w["w_svd"] * _col("cf_svd_score")
         + w["w_uu"] * _col("cf_user_user_score"))
        / total_w
    ).astype(float)


# ---------------------------------------------------------------------------
# Rule 3 — Popularity Boosted Hybrid
# ---------------------------------------------------------------------------

def score_rule3(pair_features: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    """
    Rule 3 — Popularity Boosted Hybrid
    =====================================
    Rationale: hedges users with sparse interaction history by blending the
    personalised Rule 2 score with a global popularity signal.  The blend
    factor alpha controls how much popularity is injected: alpha=0 is pure
    Rule 2, alpha=1 is pure popularity.  This is especially valuable for
    cold-start-adjacent users who have very few interactions.

    Formula:
        score = rule2_score * (1 - alpha) + popularity_score * alpha

    Alpha: scorer.rule3_alpha in config.yaml (default 0.20, clamped to [0, 1]).

    Parameters
    ----------
    pair_features : pd.DataFrame
        Must contain all columns required by score_rule2 plus 'popularity_score'.
        NaN → 0.
    config : dict
        Parsed config.yaml; reads scorer.rule3_alpha.

    Returns
    -------
    pd.Series
        Float scores with the same index as pair_features.
    """
    alpha = float(config.get("scorer", {}).get("rule3_alpha", 0.20))
    alpha = max(0.0, min(1.0, alpha))

    rule2 = score_rule2(pair_features, config)

    if "popularity_score" in pair_features.columns:
        pop = pair_features["popularity_score"].fillna(0.0)
    else:
        pop = pd.Series(np.zeros(len(pair_features)), index=pair_features.index)

    return ((1.0 - alpha) * rule2 + alpha * pop).astype(float)


# ---------------------------------------------------------------------------
# Submission helpers
# ---------------------------------------------------------------------------

def write_submission(
    user_track_pairs: pd.DataFrame,
    scores: pd.Series,
    output_path: "Path | str",
    top_n: int = 100,
    soft_rank_probs: list[float] | None = None,
) -> None:
    """Write a submission CSV matching the TrackID/Predictor format.

    When *soft_rank_probs* is provided (recommended), each candidate in a
    user's set is assigned the probability at its rank position: rank 1 →
    soft_rank_probs[0], rank 2 → soft_rank_probs[1], etc.  Candidates ranked
    beyond len(soft_rank_probs) receive the last listed probability.

    This mirrors hw4.py's approach (soft_rank_probs=[0.99,0.95,0.90,0.10,0.05,
    0.01]) which scored 0.851 AUC — the Kaggle metric is ROC AUC on the
    continuous Predictor column, so soft probabilities are essential.

    When soft_rank_probs is None, falls back to hard 0/1 labels (top_n
    candidates get 1, the rest get 0).

    Parameters
    ----------
    user_track_pairs : pd.DataFrame
        Columns ['user_id', 'track_id']; one row per (user, track) pair.
    scores : pd.Series
        Numeric score for each row in *user_track_pairs* (same positional order).
    output_path : Path or str
        Destination file path.  Parent directories are created if absent.
    top_n : int
        Used only when soft_rank_probs is None.
    soft_rank_probs : list[float] or None
        Rank-to-probability mapping.  Index 0 = rank 1 (best).
    """
    df = user_track_pairs[["user_id", "track_id"]].copy().reset_index(drop=True)
    df["score"] = scores.values
    df["rank"] = (
        df.groupby("user_id")["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    if soft_rank_probs is not None:
        probs = list(soft_rank_probs)
        df["Predictor"] = df["rank"].apply(
            lambda r: probs[r - 1] if r - 1 < len(probs) else probs[-1]
        )
    else:
        df["Predictor"] = (df["rank"] <= top_n).astype(int)

    df["TrackID"] = df["user_id"].astype(str) + "_" + df["track_id"].astype(str)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["TrackID", "Predictor"]].to_csv(out, index=False)


def run_all_rules(
    test_pairs: pd.DataFrame,
    train: pd.DataFrame,
    features: dict[str, Any],
    config: dict[str, Any],
    submissions_dir: "str | Path" = "submissions",
) -> None:
    """Compute all three scoring rules and write one CSV per rule.

    Builds per-pair content and CF feature vectors exactly once, then applies
    each rule in turn.  Writes:
      ``{submissions_dir}/submission_rule1.csv``
      ``{submissions_dir}/submission_rule2.csv``
      ``{submissions_dir}/submission_rule3.csv``

    Parameters
    ----------
    test_pairs : pd.DataFrame
        Columns ['user_id', 'track_id'] — all (user, track) pairs to evaluate.
    train : pd.DataFrame
        Training interactions with columns ['user_id', 'track_id'].
    features : dict
        Artifact dict from feature_engineering.run().  Must contain
        'track_features' (pd.DataFrame) with columns track_id, album_id,
        artist_id, genre_ids, popularity_score.
    config : dict
        Parsed config.yaml.
    submissions_dir : str or Path
        Directory in which to write the three CSV files.
    """
    from src.feature_engineering import compute_user_track_features
    from src.collab_features import compute_cf_features

    top_n = int(config.get("pipeline", {}).get("top_n", 100))
    track_features = features["track_features"]

    # Build per-pair content features and CF features exactly once
    content_feats = compute_user_track_features(test_pairs, train, track_features)
    cf_feats = compute_cf_features(test_pairs, train, config)

    # Merge content + CF + popularity into a single feature DataFrame
    pair_features = (
        content_feats
        .merge(
            cf_feats[["user_id", "track_id",
                      "cf_user_user_score", "cf_item_item_score", "cf_svd_score"]],
            on=["user_id", "track_id"],
            how="left",
        )
        .merge(
            track_features[["track_id", "popularity_score"]],
            on="track_id",
            how="left",
        )
    )

    out_dir = Path(submissions_dir)
    write_submission(
        test_pairs, score_rule1(pair_features, config),
        out_dir / "submission_rule1.csv", top_n,
    )
    write_submission(
        test_pairs, score_rule2(pair_features, config),
        out_dir / "submission_rule2.csv", top_n,
    )
    write_submission(
        test_pairs, score_rule3(pair_features, config),
        out_dir / "submission_rule3.csv", top_n,
    )
