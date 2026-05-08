"""
features.py — Build (userId, trackId) feature vectors for PySpark ML classifiers.

Features (13 total):
  User-level:  user_mean_rating, user_rating_count, user_rating_std, user_is_cold
  Track-level: track_mean_rating, track_rating_count, track_is_cold
  Interaction: user_artist_mean_rating, user_artist_count,
               user_album_mean_rating, user_album_count,
               user_genre_mean_rating, user_genre_overlap_count
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean, stdev

import pandas as pd


FEATURE_COLS = [
    "user_mean_rating", "user_rating_count", "user_rating_std", "user_is_cold",
    "track_mean_rating", "track_rating_count", "track_is_cold",
    "user_artist_mean_rating", "user_artist_count",
    "user_album_mean_rating", "user_album_count",
    "user_genre_mean_rating", "user_genre_overlap_count",
]


def _precompute(user_history, track_meta):
    """Pre-compute all aggregates needed for feature lookup."""

    # --- global mean rating ---
    all_ratings = [r for items in user_history.values() for _, r in items]
    global_mean = mean(all_ratings) if all_ratings else 50.0

    # --- per-user stats ---
    user_stats = {}
    for uid, items in user_history.items():
        ratings = [r for _, r in items]
        user_stats[uid] = {
            "mean": mean(ratings),
            "count": len(ratings),
            "std": stdev(ratings) if len(ratings) >= 2 else 0.0,
        }

    # --- per-track stats (aggregated across all users) ---
    track_ratings = defaultdict(list)
    for uid, items in user_history.items():
        for tid, r in items:
            track_ratings[tid].append(r)

    track_stats = {}
    for tid, ratings in track_ratings.items():
        track_stats[tid] = {
            "mean": mean(ratings),
            "count": len(ratings),
        }

    # --- per-(user, artist) stats ---
    # For each user, group their ratings by artist
    user_artist = defaultdict(lambda: defaultdict(list))
    user_album = defaultdict(lambda: defaultdict(list))
    user_genre = defaultdict(lambda: defaultdict(list))

    for uid, items in user_history.items():
        for tid, r in items:
            meta = track_meta.get(tid)
            if meta is None:
                continue
            if meta["artist"] is not None:
                user_artist[uid][meta["artist"]].append(r)
            if meta["album"] is not None:
                user_album[uid][meta["album"]].append(r)
            for g in meta["genres"]:
                user_genre[uid][g].append(r)

    return global_mean, user_stats, track_stats, user_artist, user_album, user_genre


def build_features(pairs, user_history, track_meta):
    """
    Build feature DataFrame for a list of (userId, trackId, label_or_None) tuples.

    Returns pandas DataFrame with columns:
        userId, trackId, label, + 13 feature columns
    """
    global_mean, user_stats, track_stats, user_artist, user_album, user_genre = \
        _precompute(user_history, track_meta)

    rows = []
    for uid, tid, label in pairs:
        meta = track_meta.get(tid)

        # --- user-level ---
        us = user_stats.get(uid)
        if us is not None:
            u_mean = us["mean"]
            u_count = us["count"]
            u_std = us["std"]
            u_cold = 0
        else:
            u_mean = global_mean
            u_count = 0
            u_std = 0.0
            u_cold = 1

        # --- track-level ---
        ts = track_stats.get(tid)
        if ts is not None:
            t_mean = ts["mean"]
            t_count = ts["count"]
            t_cold = 0
        else:
            t_mean = global_mean
            t_count = 0
            t_cold = 1

        # --- interaction: artist ---
        artist_id = meta["artist"] if meta else None
        ua_ratings = user_artist.get(uid, {}).get(artist_id) if artist_id else None
        if ua_ratings:
            ua_mean = mean(ua_ratings)
            ua_count = len(ua_ratings)
        else:
            ua_mean = global_mean
            ua_count = 0

        # --- interaction: album ---
        album_id = meta["album"] if meta else None
        ual_ratings = user_album.get(uid, {}).get(album_id) if album_id else None
        if ual_ratings:
            ual_mean = mean(ual_ratings)
            ual_count = len(ual_ratings)
        else:
            ual_mean = global_mean
            ual_count = 0

        # --- interaction: genre overlap ---
        genres = meta["genres"] if meta else set()
        ug_data = user_genre.get(uid, {})
        genre_ratings = []
        genre_overlap = 0
        for g in genres:
            if g in ug_data:
                genre_ratings.extend(ug_data[g])
                genre_overlap += 1

        ug_mean = mean(genre_ratings) if genre_ratings else global_mean
        ug_overlap = genre_overlap

        rows.append({
            "userId": uid,
            "trackId": tid,
            "label": label,
            "user_mean_rating": u_mean,
            "user_rating_count": u_count,
            "user_rating_std": u_std,
            "user_is_cold": u_cold,
            "track_mean_rating": t_mean,
            "track_rating_count": t_count,
            "track_is_cold": t_cold,
            "user_artist_mean_rating": ua_mean,
            "user_artist_count": ua_count,
            "user_album_mean_rating": ual_mean,
            "user_album_count": ual_count,
            "user_genre_mean_rating": ug_mean,
            "user_genre_overlap_count": ug_overlap,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from pathlib import Path
    from src.ml.parsers import read_train_history, read_track_metadata, DATA_DIR

    print("Loading data...")
    history = read_train_history(str(DATA_DIR / "trainItem2.txt"))
    meta = read_track_metadata(str(DATA_DIR / "trackData2.txt"))

    # Read first 10 pairs from test2_new.txt
    pairs = []
    with open(DATA_DIR / "test2_new.txt") as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            u, t, l = line.strip().split("|")
            pairs.append((int(u), int(t), int(l)))

    print(f"Building features for {len(pairs)} pairs...")
    df = build_features(pairs, history, meta)
    print(df.to_string())
    print(f"\nNull counts:\n{df.isnull().sum()}")
