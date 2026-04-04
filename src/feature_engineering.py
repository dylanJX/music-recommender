"""
feature_engineering.py  (Step 1a)
==================================
Transforms raw loaded data into feature vectors and derived signals used
by the scorer and cold-start modules.

Responsibilities:
  - Compute per-user listening profiles (genre affinities, artist affinities,
    album affinities) from training interactions.
  - Compute per-track popularity signals (global play counts, recent trend).
  - Build user-track, user-artist, and user-genre co-occurrence matrices.
  - Produce any normalised feature columns required by config-driven weights
    in scorer.py.

All heavy computation should be cache-friendly: if an artifact already exists
on disk (e.g. a pre-computed affinity matrix) it can be loaded instead of
recomputed. Cache paths are controlled via config.yaml.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _enrich_interactions(train: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """Join training interactions with track metadata (album, artist, genre_ids)."""
    return train[["user_id", "track_id"]].merge(
        tracks[["track_id", "album_id", "artist_id", "genre_ids"]],
        on="track_id",
        how="left",
    )


def _series_to_ranked_list(counts: pd.Series, entity_col: str) -> pd.Series:
    """Convert a (user_id, entity_col) MultiIndex size Series to ranked lists per user.

    Parameters
    ----------
    counts : pd.Series
        MultiIndex Series with levels (user_id, entity_col) and size values.
    entity_col : str
        Name of the entity level (e.g. 'artist_id', 'genre_id').

    Returns
    -------
    pd.Series
        Index: user_id.  Values: list of entity IDs sorted by descending count.
    """
    df = counts.rename("count").reset_index()
    return (
        df.sort_values(["user_id", "count"], ascending=[True, False])
        .groupby("user_id")[entity_col]
        .apply(list)
    )


def _series_to_dict_per_user(counts: pd.Series, entity_col: str) -> pd.Series:
    """Convert a (user_id, entity_col) MultiIndex size Series to {entity_id: count} dicts.

    Parameters
    ----------
    counts : pd.Series
        MultiIndex Series with levels (user_id, entity_col) and size values.
    entity_col : str
        Name of the entity level.

    Returns
    -------
    pd.Series
        Index: user_id.  Values: dict mapping entity_id to raw interaction count.
    """
    df = counts.rename("count").reset_index()
    result: dict[int, dict] = {}
    for uid, grp in df.groupby("user_id"):
        result[uid] = dict(zip(grp[entity_col], grp["count"]))
    return pd.Series(result)


def _build_user_entity_scores(
    train: pd.DataFrame,
    tracks: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute normalised per-user interaction scores for albums, artists, and genres.

    Each score is the entity's share of that user's total interactions, so values
    lie in [0, 1] and sum to 1 within each user.

    Parameters
    ----------
    train : pd.DataFrame
        Training interactions with columns ['user_id', 'track_id'].
    tracks : pd.DataFrame
        Track metadata with columns ['track_id', 'album_id', 'artist_id', 'genre_ids'].

    Returns
    -------
    tuple of four pd.Series
        (user_album_scores, user_artist_scores, user_genre_scores, user_genre_raw_counts)

        The first three have MultiIndex (user_id, entity_id) and values in [0, 1].
        user_genre_raw_counts has the same MultiIndex but holds raw interaction counts.
    """
    enriched = _enrich_interactions(train, tracks)

    def _normalise(counts: pd.Series) -> pd.Series:
        return counts / counts.groupby(level=0).transform("sum")

    album_counts = enriched.groupby(["user_id", "album_id"]).size()
    artist_counts = enriched.groupby(["user_id", "artist_id"]).size()

    genre_rows = (
        enriched[["user_id", "genre_ids"]]
        .explode("genre_ids")
        .rename(columns={"genre_ids": "genre_id"})
        .dropna(subset=["genre_id"])
        .copy()
    )
    genre_rows["genre_id"] = genre_rows["genre_id"].astype(int)
    genre_raw = genre_rows.groupby(["user_id", "genre_id"]).size()

    return (
        _normalise(album_counts),
        _normalise(artist_counts),
        _normalise(genre_raw),
        genre_raw,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_user_profiles(
    train: pd.DataFrame,
    tracks: pd.DataFrame,
    albums: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate per-user listening history into feature vectors.

    Parameters
    ----------
    train : pd.DataFrame
        Raw training interaction DataFrame from data_loader (columns: user_id, track_id).
    tracks : pd.DataFrame
        Track metadata from data_loader (columns: track_id, album_id, artist_id, genre_ids).
    albums : pd.DataFrame
        Album metadata (reserved for album-level genre enrichment in future steps).

    Returns
    -------
    pd.DataFrame
        One row per user_id.  Columns:
          user_id, n_interactions,
          top_genres   (list of genre_id sorted by descending play count),
          top_artists  (list of artist_id),
          top_albums   (list of album_id),
          genre_counts  (dict genre_id → raw interaction count),
          artist_counts (dict artist_id → raw interaction count),
          album_counts  (dict album_id → raw interaction count).
    """
    enriched = _enrich_interactions(train, tracks)

    n_interactions = enriched.groupby("user_id").size().rename("n_interactions")
    artist_counts_s = enriched.groupby(["user_id", "artist_id"]).size()
    album_counts_s = enriched.groupby(["user_id", "album_id"]).size()

    genre_rows = (
        enriched[["user_id", "genre_ids"]]
        .explode("genre_ids")
        .rename(columns={"genre_ids": "genre_id"})
        .dropna(subset=["genre_id"])
        .copy()
    )
    genre_rows["genre_id"] = genre_rows["genre_id"].astype(int)
    genre_counts_s = genre_rows.groupby(["user_id", "genre_id"]).size()

    profiles = pd.concat(
        [
            n_interactions,
            _series_to_ranked_list(genre_counts_s, "genre_id").rename("top_genres"),
            _series_to_ranked_list(artist_counts_s, "artist_id").rename("top_artists"),
            _series_to_ranked_list(album_counts_s, "album_id").rename("top_albums"),
            _series_to_dict_per_user(genre_counts_s, "genre_id").rename("genre_counts"),
            _series_to_dict_per_user(artist_counts_s, "artist_id").rename("artist_counts"),
            _series_to_dict_per_user(album_counts_s, "album_id").rename("album_counts"),
        ],
        axis=1,
    )
    profiles.index.name = "user_id"
    return profiles.reset_index()


def build_track_features(
    train: pd.DataFrame,
    tracks: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich track metadata with popularity signals derived from training data.

    Parameters
    ----------
    train : pd.DataFrame
        Training interactions with columns ['user_id', 'track_id'].
    tracks : pd.DataFrame
        Track metadata with columns ['track_id', 'album_id', 'artist_id', 'genre_ids'].

    Returns
    -------
    pd.DataFrame
        tracks DataFrame with three additional columns:
          global_play_count    (int)   total interactions across all users,
          unique_listener_count (int)  number of distinct users who interacted,
          popularity_score     (float) normalised [0, 1] composite popularity.
        Tracks absent from training receive 0 for all three columns.
    """
    play_counts = train.groupby("track_id").size().rename("global_play_count")
    listener_counts = (
        train.groupby("track_id")["user_id"].nunique().rename("unique_listener_count")
    )

    result = tracks.copy()
    result = result.merge(play_counts.reset_index(), on="track_id", how="left")
    result = result.merge(listener_counts.reset_index(), on="track_id", how="left")
    result["global_play_count"] = result["global_play_count"].fillna(0).astype(int)
    result["unique_listener_count"] = result["unique_listener_count"].fillna(0).astype(int)

    max_count = result["global_play_count"].max()
    result["popularity_score"] = (
        result["global_play_count"] / max_count if max_count > 0 else 0.0
    )
    return result


def compute_genre_affinity(user_profiles: pd.DataFrame) -> pd.DataFrame:
    """Return a user × genre affinity matrix with values in [0, 1].

    Affinity for a (user, genre) pair equals that genre's share of the user's
    total genre interactions, so each user's row sums to 1.

    Parameters
    ----------
    user_profiles : pd.DataFrame
        Output of build_user_profiles(); must contain columns 'user_id' and
        'genre_counts' (dict genre_id → raw count).

    Returns
    -------
    pd.DataFrame
        Index: user_id.  Columns: genre_id.  Values in [0, 1].
        Missing (user, genre) combinations are filled with 0.
    """
    rows = []
    for _, row in user_profiles.iterrows():
        uid = row["user_id"]
        gc = row.get("genre_counts")
        if not isinstance(gc, dict):
            continue
        for gid, cnt in gc.items():
            rows.append({"user_id": uid, "genre_id": gid, "count": cnt})

    if not rows:
        return pd.DataFrame()

    long = pd.DataFrame(rows)
    user_totals = long.groupby("user_id")["count"].sum().rename("total")
    long = long.join(user_totals, on="user_id")
    long["affinity"] = long["count"] / long["total"]
    return long.pivot(index="user_id", columns="genre_id", values="affinity").fillna(0.0)


def compute_artist_affinity(user_profiles: pd.DataFrame) -> pd.DataFrame:
    """Return a user × artist affinity matrix with values in [0, 1].

    Affinity for a (user, artist) pair equals that artist's share of the user's
    total artist interactions.

    Parameters
    ----------
    user_profiles : pd.DataFrame
        Output of build_user_profiles(); must contain columns 'user_id' and
        'artist_counts' (dict artist_id → raw count).

    Returns
    -------
    pd.DataFrame
        Index: user_id.  Columns: artist_id.  Values in [0, 1].
        Missing (user, artist) combinations are filled with 0.
    """
    rows = []
    for _, row in user_profiles.iterrows():
        uid = row["user_id"]
        ac = row.get("artist_counts")
        if not isinstance(ac, dict):
            continue
        for aid, cnt in ac.items():
            rows.append({"user_id": uid, "artist_id": aid, "count": cnt})

    if not rows:
        return pd.DataFrame()

    long = pd.DataFrame(rows)
    user_totals = long.groupby("user_id")["count"].sum().rename("total")
    long = long.join(user_totals, on="user_id")
    long["affinity"] = long["count"] / long["total"]
    return long.pivot(index="user_id", columns="artist_id", values="affinity").fillna(0.0)


def compute_user_track_features(
    user_track_pairs: pd.DataFrame,
    train: pd.DataFrame,
    tracks: pd.DataFrame,
) -> pd.DataFrame:
    """Compute a feature vector for each (user_id, track_id) pair.

    Derives user preference scores for the track's album and artist from training
    interaction counts, then computes summary statistics over the track's genre list
    using the user's per-genre interaction share.

    Parameters
    ----------
    user_track_pairs : pd.DataFrame
        Columns: ['user_id', 'track_id']. One row per (user, track) pair to score.
    train : pd.DataFrame
        Training interactions with at minimum columns ['user_id', 'track_id'].
    tracks : pd.DataFrame
        Track metadata with columns ['track_id', 'album_id', 'artist_id', 'genre_ids'].

    Returns
    -------
    pd.DataFrame
        One row per input pair.  Columns:

        user_id, track_id
        album_score       -- user's normalised interaction share for this album
                             (NaN when the user has never heard any track on this album)
        artist_score      -- user's normalised interaction share for this artist
                             (NaN when unseen)

        Genre statistics computed over the user's scores for each of the track's genres.
        A score of 0 is assigned to genres the user has not interacted with.

        genre_count       -- total number of genres this track belongs to
        genre_max         -- highest user genre score among the track's genres
        genre_min         -- lowest user genre score
        genre_mean        -- mean user genre score
        genre_variance    -- variance (ddof=0) of user genre scores
        genre_median      -- median user genre score
        genre_sum         -- sum of user genre scores
        genre_range       -- genre_max - genre_min
        genre_nonzero_count -- number of the track's genres the user has interacted with
        genre_weighted_mean -- genre scores weighted by the user's raw genre interaction counts;
                               falls back to unweighted mean when the user has no genre history
    """
    album_scores, artist_scores, genre_scores, genre_raw = _build_user_entity_scores(
        train, tracks
    )

    df = user_track_pairs[["user_id", "track_id"]].copy().reset_index(drop=True)
    df = df.merge(
        tracks[["track_id", "album_id", "artist_id", "genre_ids"]],
        on="track_id",
        how="left",
    )

    # album_score: NaN when user has never interacted with this album
    album_df = album_scores.rename("album_score").reset_index()
    df = df.merge(album_df, on=["user_id", "album_id"], how="left")

    # artist_score: NaN when user has never interacted with this artist
    artist_df = artist_scores.rename("artist_score").reset_index()
    df = df.merge(artist_df, on=["user_id", "artist_id"], how="left")

    # ---------- genre statistics ----------
    df = df.reset_index(drop=True)
    df["_pair_idx"] = df.index

    genre_exp = (
        df[["_pair_idx", "user_id", "genre_ids"]]
        .explode("genre_ids")
        .rename(columns={"genre_ids": "genre_id"})
        .dropna(subset=["genre_id"])
        .copy()
    )
    genre_exp["genre_id"] = genre_exp["genre_id"].astype(int)

    # Look up per-genre normalised score (default 0 for unseen genres)
    g_score_df = genre_scores.rename("genre_score").reset_index()
    genre_exp = genre_exp.merge(g_score_df, on=["user_id", "genre_id"], how="left")
    genre_exp["genre_score"] = genre_exp["genre_score"].fillna(0.0)

    # Look up raw genre counts for weighted mean
    g_raw_df = genre_raw.rename("genre_raw").reset_index()
    genre_exp = genre_exp.merge(g_raw_df, on=["user_id", "genre_id"], how="left")
    genre_exp["genre_raw"] = genre_exp["genre_raw"].fillna(0.0)

    def _genre_agg(g: pd.DataFrame) -> pd.Series:
        s = g["genre_score"].values
        w = g["genre_raw"].values
        if len(s) == 0:
            nans = dict.fromkeys(
                ["genre_max", "genre_min", "genre_mean", "genre_variance",
                 "genre_median", "genre_range", "genre_weighted_mean"],
                np.nan,
            )
            return pd.Series({"genre_count": 0, "genre_sum": 0.0,
                               "genre_nonzero_count": 0, **nans})
        w_sum = float(w.sum())
        return pd.Series({
            "genre_count": len(s),
            "genre_max": float(s.max()),
            "genre_min": float(s.min()),
            "genre_mean": float(s.mean()),
            "genre_variance": float(s.var(ddof=0)),
            "genre_median": float(np.median(s)),
            "genre_sum": float(s.sum()),
            "genre_range": float(s.max() - s.min()),
            "genre_nonzero_count": int((s > 0).sum()),
            "genre_weighted_mean": (
                float(np.average(s, weights=w)) if w_sum > 0 else float(s.mean())
            ),
        })

    _GENRE_STAT_COLS = [
        "genre_count", "genre_max", "genre_min", "genre_mean", "genre_variance",
        "genre_median", "genre_sum", "genre_range", "genre_nonzero_count",
        "genre_weighted_mean",
    ]
    if genre_exp.empty:
        # All queried tracks have no genres; produce an empty aggregation with the
        # right column schema so the left merge below fills everything with NaN.
        genre_stats = pd.DataFrame(
            columns=_GENRE_STAT_COLS,
            index=pd.Index([], name="_pair_idx", dtype=int),
        )
    else:
        genre_stats = (
            genre_exp.groupby("_pair_idx")[["genre_score", "genre_raw"]]
            .apply(_genre_agg)
        )

    df = df.merge(genre_stats, left_on="_pair_idx", right_index=True, how="left")
    df = df.drop(columns=["_pair_idx", "album_id", "artist_id", "genre_ids"])

    return df[[
        "user_id", "track_id",
        "album_score", "artist_score",
        "genre_count", "genre_max", "genre_min", "genre_mean",
        "genre_variance", "genre_median", "genre_sum", "genre_range",
        "genre_nonzero_count", "genre_weighted_mean",
    ]]


def compute_idf_weights(
    train: pd.DataFrame,
    tracks: pd.DataFrame,
) -> dict[str, dict]:
    """Compute IDF weights for albums, artists, and genres.

    IDF(x) = log((1 + N) / (1 + df(x))) + 1.0
    where N = total distinct training users, df(x) = number of distinct users
    who interacted with any track carrying feature x.

    Parameters
    ----------
    train : pd.DataFrame
        Training interactions with columns ['user_id', 'track_id'].
    tracks : pd.DataFrame
        Track metadata with columns ['track_id', 'album_id', 'artist_id', 'genre_ids'].

    Returns
    -------
    dict with keys 'album_idf', 'artist_idf', 'genre_idf' — each a plain dict
    mapping entity_id (int) to its IDF weight (float >= 1.0).
    """
    enriched = _enrich_interactions(train, tracks)
    n_users = int(train["user_id"].nunique())

    def _idf(df_counts: pd.Series) -> dict:
        return {
            int(k): math.log((1 + n_users) / (1 + int(v))) + 1.0
            for k, v in df_counts.items()
        }

    album_df = (
        enriched.dropna(subset=["album_id"])
        .groupby("album_id")["user_id"].nunique()
    )
    artist_df = (
        enriched.dropna(subset=["artist_id"])
        .groupby("artist_id")["user_id"].nunique()
    )

    genre_rows = (
        enriched[["user_id", "genre_ids"]]
        .explode("genre_ids")
        .rename(columns={"genre_ids": "genre_id"})
        .dropna(subset=["genre_id"])
    )
    genre_rows["genre_id"] = genre_rows["genre_id"].astype(int)
    genre_df = genre_rows.groupby("genre_id")["user_id"].nunique()

    return {
        "album_idf": _idf(album_df),
        "artist_idf": _idf(artist_df),
        "genre_idf": _idf(genre_df),
    }


def compute_hw4_features(
    user_track_pairs: pd.DataFrame,
    train: pd.DataFrame,
    tracks: pd.DataFrame,
    idf_weights: dict,
) -> pd.DataFrame:
    """Compute hw4-style IDF-weighted per-pair scoring features.

    Mirrors hw4.py's ``score_candidate()`` feature set, producing numeric
    columns that scorer.py's rule functions combine into final scores.

    Parameters
    ----------
    user_track_pairs : pd.DataFrame
        Columns ['user_id', 'track_id'].
    train : pd.DataFrame
        Training interactions with columns ['user_id', 'track_id'] and
        optionally 'play_count' (used as rating weight / 100.0; defaults to 1.0
        per interaction when the column is absent).
    tracks : pd.DataFrame
        Track metadata with columns ['track_id', 'album_id', 'artist_id', 'genre_ids'].
    idf_weights : dict
        Output of compute_idf_weights(); keys 'album_idf', 'artist_idf', 'genre_idf'.

    Returns
    -------
    pd.DataFrame
        Same rows as user_track_pairs plus columns:
          hw4_track_score  -- user's play-count-weighted exact track preference.
          hw4_artist_score -- user artist preference * artist IDF.
          hw4_album_score  -- user album preference * album IDF.
          hw4_genre_score  -- sum(user genre pref * genre IDF) across track genres.
          hw4_pop_score    -- track's total play-count popularity, normalised [0, 1].
        All scores default to 0.0 for cold pairs.
    """
    album_idf = idf_weights.get("album_idf", {})
    artist_idf = idf_weights.get("artist_idf", {})
    genre_idf = idf_weights.get("genre_idf", {})

    has_pc = "play_count" in train.columns
    train_w = train.copy()
    train_w["_w"] = train_w["play_count"] / 100.0 if has_pc else 1.0

    # Enrich training with track metadata for computing user entity preferences
    enriched = train_w[["user_id", "track_id", "_w"]].merge(
        tracks[["track_id", "album_id", "artist_id", "genre_ids"]],
        on="track_id",
        how="left",
    )

    # -- User track preference (exact match, play-count weighted) ---------------
    user_track_pref = (
        train_w.groupby(["user_id", "track_id"])["_w"]
        .sum()
        .rename("hw4_track_score")
        .reset_index()
    )

    # -- User album preference × album IDF -------------------------------------
    album_idf_s = pd.Series(album_idf).rename("_aidf")
    user_album_pref = (
        enriched.dropna(subset=["album_id"])
        .assign(album_id=lambda d: d["album_id"].astype(int))
        .groupby(["user_id", "album_id"])["_w"]
        .sum()
        .rename("_ap")
        .reset_index()
    )
    user_album_pref = user_album_pref.join(album_idf_s, on="album_id")
    user_album_pref["hw4_album_score"] = (
        user_album_pref["_ap"] * user_album_pref["_aidf"].fillna(1.0)
    )

    # -- User artist preference × artist IDF -----------------------------------
    artist_idf_s = pd.Series(artist_idf).rename("_aridf")
    user_artist_pref = (
        enriched.dropna(subset=["artist_id"])
        .assign(artist_id=lambda d: d["artist_id"].astype(int))
        .groupby(["user_id", "artist_id"])["_w"]
        .sum()
        .rename("_arp")
        .reset_index()
    )
    user_artist_pref = user_artist_pref.join(artist_idf_s, on="artist_id")
    user_artist_pref["hw4_artist_score"] = (
        user_artist_pref["_arp"] * user_artist_pref["_aridf"].fillna(1.0)
    )

    # -- User genre preference × genre IDF -------------------------------------
    genre_idf_s = pd.Series(genre_idf).rename("_gidf")
    genre_rows_e = (
        enriched[["user_id", "genre_ids", "_w"]]
        .explode("genre_ids")
        .rename(columns={"genre_ids": "genre_id"})
        .dropna(subset=["genre_id"])
    )
    genre_rows_e["genre_id"] = genre_rows_e["genre_id"].astype(int)
    user_genre_pref = (
        genre_rows_e.groupby(["user_id", "genre_id"])["_w"]
        .sum()
        .rename("_gp")
        .reset_index()
    )
    user_genre_pref = user_genre_pref.join(genre_idf_s, on="genre_id")
    user_genre_pref["_gw"] = user_genre_pref["_gp"] * user_genre_pref["_gidf"].fillna(1.0)

    # -- Global track popularity -----------------------------------------------
    if has_pc:
        track_pop = train.groupby("track_id")["play_count"].sum()
    else:
        track_pop = train.groupby("track_id").size().astype(float)
    max_pop = float(track_pop.max()) if not track_pop.empty else 1.0
    track_pop_norm = (track_pop / max_pop).rename("hw4_pop_score").reset_index()

    # -- Assemble per-pair features -------------------------------------------
    df = user_track_pairs[["user_id", "track_id"]].copy().reset_index(drop=True)
    df["_pair_idx"] = df.index

    # Join track metadata
    df = df.merge(
        tracks[["track_id", "album_id", "artist_id", "genre_ids"]],
        on="track_id",
        how="left",
    )

    # Exact track preference
    df = df.merge(user_track_pref, on=["user_id", "track_id"], how="left")
    df["hw4_track_score"] = df["hw4_track_score"].fillna(0.0)

    # Album score (only where album_id is non-null)
    df_alb = df.dropna(subset=["album_id"]).copy()
    df_alb["album_id"] = df_alb["album_id"].astype(int)
    alb_pair = df_alb.merge(
        user_album_pref[["user_id", "album_id", "hw4_album_score"]],
        on=["user_id", "album_id"],
        how="left",
    )[["_pair_idx", "hw4_album_score"]]
    df = df.merge(alb_pair, on="_pair_idx", how="left")
    df["hw4_album_score"] = df["hw4_album_score"].fillna(0.0)

    # Artist score
    df_art = df.dropna(subset=["artist_id"]).copy()
    df_art["artist_id"] = df_art["artist_id"].astype(int)
    art_pair = df_art.merge(
        user_artist_pref[["user_id", "artist_id", "hw4_artist_score"]],
        on=["user_id", "artist_id"],
        how="left",
    )[["_pair_idx", "hw4_artist_score"]]
    df = df.merge(art_pair, on="_pair_idx", how="left")
    df["hw4_artist_score"] = df["hw4_artist_score"].fillna(0.0)

    # Genre score: explode, merge weighted genre preferences, sum per pair
    pair_genre = (
        df[["_pair_idx", "user_id", "genre_ids"]]
        .explode("genre_ids")
        .rename(columns={"genre_ids": "genre_id"})
        .dropna(subset=["genre_id"])
    )
    pair_genre["genre_id"] = pair_genre["genre_id"].astype(int)
    pair_genre = pair_genre.merge(
        user_genre_pref[["user_id", "genre_id", "_gw"]],
        on=["user_id", "genre_id"],
        how="left",
    )
    pair_genre["_gw"] = pair_genre["_gw"].fillna(0.0)
    genre_sum = (
        pair_genre.groupby("_pair_idx")["_gw"]
        .sum()
        .rename("hw4_genre_score")
    )
    df = df.merge(genre_sum, on="_pair_idx", how="left")
    df["hw4_genre_score"] = df["hw4_genre_score"].fillna(0.0)

    # Popularity
    df = df.merge(track_pop_norm, on="track_id", how="left")
    df["hw4_pop_score"] = df["hw4_pop_score"].fillna(0.0)

    return df[[
        "user_id", "track_id",
        "hw4_track_score", "hw4_artist_score", "hw4_album_score",
        "hw4_genre_score", "hw4_pop_score",
    ]]


def run(data: dict[str, Any]) -> dict[str, Any]:
    """Top-level entry point called by pipeline.py.

    Accepts the dict returned by data_loader.load_all() and returns a dict
    of all computed feature artifacts.

    Parameters
    ----------
    data : dict
        Must contain keys 'train', 'tracks', 'albums' (DataFrames from data_loader).

    Returns
    -------
    dict with keys:
        'user_profiles'   -> pd.DataFrame  (one row per user)
        'track_features'  -> pd.DataFrame  (tracks enriched with popularity columns)
        'genre_affinity'  -> pd.DataFrame  (user × genre matrix, values in [0, 1])
        'artist_affinity' -> pd.DataFrame  (user × artist matrix, values in [0, 1])
    """
    train = data["train"]
    tracks = data["tracks"]
    albums = data["albums"]

    user_profiles = build_user_profiles(train, tracks, albums)
    track_features = build_track_features(train, tracks)
    genre_affinity = compute_genre_affinity(user_profiles)
    artist_affinity = compute_artist_affinity(user_profiles)
    idf_weights = compute_idf_weights(train, tracks)

    return {
        "user_profiles": user_profiles,
        "track_features": track_features,
        "genre_affinity": genre_affinity,
        "artist_affinity": artist_affinity,
        "idf_weights": idf_weights,
    }
