"""
cold_start.py  (Steps 2a and 2b)
=================================
Handles recommendation for two cold-start scenarios where the standard
collaborative / rule-based scorer cannot operate effectively:

  2a. New user (no interaction history):
      The user appears in the test set but has zero training interactions.
      Strategy: fall back to a popularity-based global ranking, optionally
      filtered by any metadata available (e.g. registration country, age
      group if provided). Fallback strategy is configurable in config.yaml
      under cold_start.new_user_strategy.

  2b. New track (item cold-start):
      A track appears in the test set's candidate pool but was never seen
      during training, so no popularity signals exist for it.
      Strategy: propagate signals from the track's album and/or artist,
      using their aggregate popularity and genre match against user profiles.
      Configurable under cold_start.new_track_strategy.

Both strategies should degrade gracefully to a global popularity fallback
if even artist/album/genre metadata is absent.

Fallback priority for album score resolution (resolve_album_score):
  1. Direct album_score from feature engineering (not NaN / None).
  2. Intra-album proxy (Strategy B): aggregate of user's normalised
     interaction shares for sibling tracks on the same album.
  3. Global imputed album mean (Strategy A): mean album score across all
     warm (user, album) pairs in training.
  4. Zero — when all signals are absent.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Cold-start detection helpers
# ---------------------------------------------------------------------------

def is_cold_user(user_id: int, train: pd.DataFrame) -> bool:
    """Return True if user_id has no rows in the training interaction set.

    Parameters
    ----------
    user_id : int
        User to check.
    train : pd.DataFrame
        Training interactions with at least column 'user_id'.

    Returns
    -------
    bool
    """
    return user_id not in train["user_id"].values


def is_cold_track(track_id: int, train: pd.DataFrame) -> bool:
    """Return True if track_id never appeared in the training interaction set.

    Parameters
    ----------
    track_id : int
        Track to check.
    train : pd.DataFrame
        Training interactions with at least column 'track_id'.

    Returns
    -------
    bool
    """
    return track_id not in train["track_id"].values


# ---------------------------------------------------------------------------
# Strategy A — Global imputed scores
# ---------------------------------------------------------------------------

def compute_global_imputed_scores(
    train: pd.DataFrame,
    tracks: pd.DataFrame,
) -> dict[str, float]:
    """Compute global mean feature scores to impute values for cold users.

    Heuristic (Strategy A):
      For each warm user, compute normalised interaction shares per album and
      artist (entity_count / user_total_interactions).  The global imputed
      value is the mean of these normalised shares across ALL (user, entity)
      pairs seen in training.

      Rationale: a cold user with no personal history is treated as an
      'average' warm user — they receive the expected affinity signal one
      would observe for a randomly sampled (user, entity) pair.

    Parameters
    ----------
    train : pd.DataFrame
        Training interactions with columns ['user_id', 'track_id'].
    tracks : pd.DataFrame
        Track metadata with at least columns ['track_id', 'album_id', 'artist_id'].

    Returns
    -------
    dict with keys:
        'album_score_mean'  (float) global mean normalised album score.
        'artist_score_mean' (float) global mean normalised artist score.
    """
    enriched = train[["user_id", "track_id"]].merge(
        tracks[["track_id", "album_id", "artist_id"]], on="track_id", how="left"
    )
    if enriched.empty:
        return {"album_score_mean": 0.0, "artist_score_mean": 0.0}

    album_counts = enriched.groupby(["user_id", "album_id"]).size()
    artist_counts = enriched.groupby(["user_id", "artist_id"]).size()

    album_scores = album_counts / album_counts.groupby(level=0).transform("sum")
    artist_scores = artist_counts / artist_counts.groupby(level=0).transform("sum")

    return {
        "album_score_mean": float(album_scores.mean()),
        "artist_score_mean": float(artist_scores.mean()),
    }


# ---------------------------------------------------------------------------
# Strategy B — Intra-album proxy score
# ---------------------------------------------------------------------------

def intra_album_proxy_score(
    user_id: int,
    track_id: int,
    train: pd.DataFrame,
    tracks: pd.DataFrame,
    config: dict[str, Any],
) -> float | None:
    """Compute a proxy album score from sibling-track interactions (Strategy B).

    Identifies all other tracks on the same album as *track_id* ('siblings'),
    looks up how often *user_id* interacted with each sibling in training, and
    aggregates those normalised interaction shares (count / user_total) into a
    single proxy score.

    The aggregation method is read from config['cold_start']['intra_album_agg']:
      - ``'mean'`` (default): average across all siblings the user interacted with.
      - ``'max'``: maximum sibling normalised score.

    Parameters
    ----------
    user_id : int
        Target user.
    track_id : int
        Candidate track whose direct album score is missing or NaN.
    train : pd.DataFrame
        Training interactions with columns ['user_id', 'track_id'].
    tracks : pd.DataFrame
        Track metadata with columns ['track_id', 'album_id'].
    config : dict
        Parsed config.yaml; reads cold_start.intra_album_agg (default 'mean').

    Returns
    -------
    float | None
        Proxy album score in [0, 1], or None when the user has no interactions
        with any sibling track on the same album (including the case where the
        user is a cold user entirely).
    """
    agg = config.get("cold_start", {}).get("intra_album_agg", "mean")

    track_row = tracks[tracks["track_id"] == track_id]
    if track_row.empty:
        return None
    album_id = int(track_row.iloc[0]["album_id"])

    # Find sibling tracks: same album, different track_id
    siblings = tracks[
        (tracks["album_id"] == album_id) & (tracks["track_id"] != track_id)
    ]["track_id"].tolist()
    if not siblings:
        return None

    user_rows = train[train["user_id"] == user_id]
    if user_rows.empty:
        return None

    user_total = len(user_rows)
    sibling_interactions = user_rows[user_rows["track_id"].isin(siblings)]
    if sibling_interactions.empty:
        return None

    sibling_counts = sibling_interactions.groupby("track_id").size()
    scores = (sibling_counts / user_total).values.astype(float)

    if agg == "max":
        return float(scores.max())
    return float(scores.mean())  # default: mean


# ---------------------------------------------------------------------------
# Priority chain for album score resolution
# ---------------------------------------------------------------------------

def resolve_album_score(
    user_id: int,
    track_id: int,
    album_score: float | None,
    train: pd.DataFrame,
    tracks: pd.DataFrame,
    config: dict[str, Any],
    global_album_mean: float | None = None,
) -> float:
    """Resolve the best available album affinity score using a priority chain.

    Priority order:
    1. Direct album_score from feature engineering (if not NaN / None).
    2. Intra-album proxy score (Strategy B): aggregate of user's normalised
       interaction counts for sibling tracks on the same album.
    3. Global imputed album mean (Strategy A): mean album score across all
       warm (user, album) pairs; use compute_global_imputed_scores() to obtain.
    4. Zero — last resort when every signal is absent.

    Parameters
    ----------
    user_id : int
        Target user.
    track_id : int
        Candidate track.
    album_score : float | None
        Pre-computed album score from feature_engineering; pass NaN or None
        when unavailable.
    train : pd.DataFrame
        Training interactions with columns ['user_id', 'track_id'].
    tracks : pd.DataFrame
        Track metadata with columns ['track_id', 'album_id'].
    config : dict
        Parsed config; forwarded to intra_album_proxy_score.
    global_album_mean : float | None
        Pre-computed global mean album score (Strategy A).  Pass None or NaN
        to skip Strategy A and fall through to zero.

    Returns
    -------
    float
        Best available album score, guaranteed to be a finite float ≥ 0.
    """
    def _is_missing(v: float | None) -> bool:
        return v is None or (isinstance(v, float) and math.isnan(v))

    # Priority 1: direct score from feature engineering
    if not _is_missing(album_score):
        return float(album_score)

    # Priority 2: intra-album proxy (Strategy B)
    proxy = intra_album_proxy_score(user_id, track_id, train, tracks, config)
    if proxy is not None:
        return proxy

    # Priority 3: global imputed mean (Strategy A)
    if not _is_missing(global_album_mean):
        return float(global_album_mean)

    # Priority 4: zero
    return 0.0


# ---------------------------------------------------------------------------
# Popularity fallback
# ---------------------------------------------------------------------------

def global_popularity_fallback(
    candidate_track_ids: list[int],
    track_features: pd.DataFrame,
    n: int = 100,
) -> list[int]:
    """Return top-n tracks by global popularity_score.

    Used as the ultimate fallback when no user or item signals are available.
    Ties in popularity_score are broken by track_id ascending for determinism.

    Parameters
    ----------
    candidate_track_ids : list[int]
        Pool of tracks to choose from.
    track_features : pd.DataFrame
        Output of feature_engineering.build_track_features(); must contain
        columns ['track_id', 'popularity_score'].
    n : int
        Number of tracks to return.

    Returns
    -------
    list[int]
        Up to n track IDs ordered by descending popularity_score.
    """
    subset = track_features[
        track_features["track_id"].isin(candidate_track_ids)
    ].copy()
    subset = subset.sort_values(
        ["popularity_score", "track_id"], ascending=[False, True]
    )
    return subset["track_id"].head(n).tolist()


# ---------------------------------------------------------------------------
# Recommendation entry points
# ---------------------------------------------------------------------------

def recommend_for_new_user(
    user_id: int,
    candidate_track_ids: list[int],
    track_features: pd.DataFrame,
    config: dict[str, Any],
    n: int = 100,
) -> list[int]:
    """Generate recommendations for a user with no training history (Step 2a).

    Strategy A: rank candidates purely by global popularity_score, which
    captures aggregate listening frequency across all warm users.  A cold user
    with no personal history is modelled as an 'average' listener who prefers
    what is globally popular.

    If config['cold_start']['new_user_strategy'] is ``'global_popularity'``
    (default), candidates are sorted by ``track_features.popularity_score``
    descending.  The ``'genre_filtered_popularity'`` option requires user
    metadata not available in the zero-history cold-start scenario and
    therefore degrades to global popularity until user metadata is added.

    Parameters
    ----------
    user_id : int
        Cold-start user identifier (used for logging/debugging only).
    candidate_track_ids : list[int]
        Pool of tracks to choose from.
    track_features : pd.DataFrame
        Enriched track feature DataFrame from feature_engineering; must
        contain columns ['track_id', 'popularity_score'].
    config : dict[str, Any]
        Parsed config.yaml; reads cold_start.new_user_strategy.
    n : int
        Number of recommendations to return.

    Returns
    -------
    list[int]
        Up to n track IDs ranked by the chosen cold-start strategy.
    """
    # Strategy selection is read from config; only global_popularity implemented.
    return global_popularity_fallback(candidate_track_ids, track_features, n=n)


def recommend_for_cold_track(
    user_id: int,
    cold_track_ids: list[int],
    warm_track_ids: list[int],
    features: dict[str, Any],
    config: dict[str, Any],
    n: int = 100,
) -> list[int]:
    """Blend cold-track candidates with warm recommendations (Step 2b).

    Cold tracks (unseen in training) are scored by global popularity since
    no user-specific signal exists for them.  They occupy a fixed fraction of
    the final top-n list, controlled by
    config['cold_start']['cold_track_slot_fraction'] (default 0.10).

    Warm slots come from *warm_track_ids* (assumed pre-ranked by scorer.py);
    cold slots come from *cold_track_ids* ranked by popularity_score.
    If either pool is smaller than its quota, remaining slots are backfilled
    from the other pool.

    Parameters
    ----------
    user_id : int
        Target user.
    cold_track_ids : list[int]
        Tracks unseen during training that must still be rankable.
    warm_track_ids : list[int]
        Tracks with full training signal, assumed ranked best-first.
    features : dict[str, Any]
        Feature artifact dict from feature_engineering.run(); must contain key
        'track_features' (pd.DataFrame with columns ['track_id', 'popularity_score']).
    config : dict[str, Any]
        Parsed config.yaml; reads cold_start.cold_track_slot_fraction.
    n : int
        Final list length.

    Returns
    -------
    list[int]
        Merged, re-ranked list of up to n track IDs.
    """
    track_features = features["track_features"]
    cold_fraction = float(
        config.get("cold_start", {}).get("cold_track_slot_fraction", 0.10)
    )
    n_cold = max(0, min(len(cold_track_ids), math.ceil(cold_fraction * n)))
    n_warm = n - n_cold

    ranked_cold = global_popularity_fallback(cold_track_ids, track_features, n=n_cold)
    ranked_warm = list(warm_track_ids)[:n_warm]

    combined = ranked_warm + ranked_cold

    # Backfill if either pool was smaller than its quota
    if len(combined) < n:
        seen = set(combined)
        extras = [
            t for t in list(warm_track_ids) + list(cold_track_ids)
            if t not in seen
        ]
        combined.extend(extras[: n - len(combined)])

    return combined[:n]
