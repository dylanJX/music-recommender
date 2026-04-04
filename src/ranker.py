"""
ranker.py
=========
LightGBM LambdaRank learn-to-rank model that sits on top of the existing
content + CF feature set.

Training strategy
-----------------
* Split training users 80/20 (by user, not by row) to prevent leakage.
* For each user in the 80 % split, all their deduplicated (user, track)
  interactions form one ranking group.  Label = 1 when that track's
  aggregated play_count >= the user's median play_count, else 0.
* Features are built using only the 80 % training context.

Validation
----------
* Val-split users are held out entirely; their interactions supply positives
  and we sample negatives from unseen tracks.
* AUC is computed on (positive, negative) pairs using the same protocol as
  pipeline._estimate_auc_80_20(), so numbers are directly comparable.

Submission
----------
* Test pairs are scored using the pre-built pair_features from the main
  pipeline (train_full context, includes CF) — avoids expensive recomputation.
* Cold test users receive popularity-based scores identical to other rules.
* Output: submissions/submission_lgbm.csv with soft_rank_probs applied.

Entry points
------------
* run(data, warm_pairs, test_pair_features, cold_pairs, features, config,
       submissions_dir)  — called from pipeline.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature columns fed to LightGBM (superset of what the user specified,
# including the hw4-style IDF-weighted signals that drove Rule 1 to 0.857).
# Missing columns are zero-filled; NaN → median-imputed.
# ---------------------------------------------------------------------------
FEATURE_COLS: list[str] = [
    # Content features (compute_user_track_features)
    "album_score",
    "artist_score",
    "genre_count",
    "genre_max",
    "genre_min",
    "genre_mean",
    "genre_variance",
    "genre_median",
    "genre_sum",
    "genre_range",
    "genre_nonzero_count",
    "genre_weighted_mean",
    # hw4-style IDF-weighted signals (compute_hw4_features)
    "hw4_track_score",
    "hw4_artist_score",
    "hw4_album_score",
    "hw4_genre_score",
    "hw4_pop_score",
    # Collaborative-filtering signals (compute_cf_features)
    "cf_svd_score",
    "cf_user_user_score",
    "cf_item_item_score",
    # Global popularity
    "popularity_score",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_lightgbm() -> None:
    """Exit cleanly with an install hint if lightgbm is missing."""
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        print("pip install lightgbm", flush=True)
        sys.exit(1)


def _impute(
    df: pd.DataFrame,
    medians: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fill NaN values in FEATURE_COLS with column medians.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing (a subset of) FEATURE_COLS.
    medians : dict or None
        If None, compute medians from df (training set).
        Pass a previously computed dict to apply the same imputation to
        a held-out or test set.

    Returns
    -------
    df_filled : pd.DataFrame
    medians : dict[str, float]  — computed or echoed back.
    """
    out = df.copy()
    if medians is None:
        medians = {}
        for col in FEATURE_COLS:
            if col in out.columns:
                m = float(out[col].median())
                medians[col] = 0.0 if np.isnan(m) else m
            else:
                medians[col] = 0.0

    for col in FEATURE_COLS:
        if col in out.columns:
            out[col] = out[col].fillna(medians.get(col, 0.0))
        else:
            out[col] = medians.get(col, 0.0)

    return out, medians


def _build_training_features(
    pairs: pd.DataFrame,
    train_ctx: pd.DataFrame,
    tracks: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build the full feature matrix for training (user, track) pairs.

    Uses *train_ctx* (the 80 % training split) as context for all feature
    computations so that no validation-side information leaks into training.

    Parameters
    ----------
    pairs : pd.DataFrame
        Columns ['user_id', 'track_id'].
    train_ctx : pd.DataFrame
        80 % training interactions with columns ['user_id', 'track_id'] and
        optionally 'play_count'.
    tracks : pd.DataFrame
        Track metadata from data_loader.
    config : dict

    Returns
    -------
    pd.DataFrame
        Merged feature DataFrame with user_id, track_id, and all FEATURE_COLS.
    """
    from src.feature_engineering import (
        build_track_features,
        compute_idf_weights,
        compute_user_track_features,
        compute_hw4_features,
    )
    from src.collab_features import compute_cf_features

    log.info("    [ranker] build_track_features + IDF on train_80 context ...")
    track_feats = build_track_features(train_ctx, tracks)
    idf = compute_idf_weights(train_ctx, tracks)

    log.info("    [ranker] content features (%d pairs) ...", len(pairs))
    content = compute_user_track_features(pairs, train_ctx, track_feats)

    log.info("    [ranker] hw4 features ...")
    hw4 = compute_hw4_features(pairs, train_ctx, track_feats, idf)

    log.info("    [ranker] CF features ...")
    cf = compute_cf_features(pairs, train_ctx, config)

    merged = (
        content
        .merge(
            hw4[["user_id", "track_id",
                 "hw4_track_score", "hw4_artist_score", "hw4_album_score",
                 "hw4_genre_score", "hw4_pop_score"]],
            on=["user_id", "track_id"],
            how="left",
        )
        .merge(
            cf[["user_id", "track_id",
                "cf_user_user_score", "cf_item_item_score", "cf_svd_score"]],
            on=["user_id", "track_id"],
            how="left",
        )
        .merge(
            track_feats[["track_id", "popularity_score"]],
            on="track_id",
            how="left",
        )
    )
    return merged


# ---------------------------------------------------------------------------
# Training-data construction
# ---------------------------------------------------------------------------

def _build_training_data(
    train_80: pd.DataFrame,
    tracks: pd.DataFrame,
    config: dict[str, Any],
    max_users: int | None = None,
    max_interactions_per_user: int | None = 50,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build (pair_features, labels, group_sizes) for LambdaRank.

    For each user in *train_80*, all their deduplicated interactions (up to
    *max_interactions_per_user*) form one ranking group.  Label = 1 when
    that track's play_count >= the user's median play_count, else = 0.

    Users with fewer than 2 distinct interactions are skipped.

    Parameters
    ----------
    train_80 : pd.DataFrame
        80 % training split.
    tracks : pd.DataFrame
        Track metadata.
    config : dict
    max_users : int or None
        Cap on how many users to include (sampled randomly).
    max_interactions_per_user : int or None
        Maximum number of interactions to keep per user.  Limiting this to
        ~50 keeps total training pairs manageable for CF computation, which
        has an O(n_pairs) inner loop.  Pass None to use all interactions.
    rng : np.random.Generator or None
        For reproducible sampling.

    Returns
    -------
    pair_features : pd.DataFrame
        Feature rows sorted by user_id (required by LambdaRank).
    labels : np.ndarray  (int, shape [n_pairs])
    groups : np.ndarray  (int, shape [n_users]) — pairs per user.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Aggregate play_counts per (user, track)
    has_pc = "play_count" in train_80.columns
    if has_pc:
        deduped = (
            train_80.groupby(["user_id", "track_id"])["play_count"]
            .sum()
            .reset_index()
        )
    else:
        deduped = (
            train_80.groupby(["user_id", "track_id"])
            .size()
            .reset_index(name="play_count")
        )

    # Drop users with < 2 distinct interactions
    user_n = deduped.groupby("user_id").size()
    valid_users = user_n[user_n >= 2].index
    deduped = deduped[deduped["user_id"].isin(valid_users)].copy()

    if deduped.empty:
        raise ValueError(
            "No training users with >= 2 interactions; "
            "cannot build LambdaRank training data."
        )

    # Optionally cap number of training users
    unique_users = deduped["user_id"].unique()
    if max_users is not None and len(unique_users) > max_users:
        sampled = rng.choice(unique_users, size=max_users, replace=False)
        deduped = deduped[deduped["user_id"].isin(sampled)].copy()
        log.info(
            "    [ranker] Sampled %d / %d train users (lgbm.max_train_users cap)",
            max_users, len(unique_users),
        )

    # Cap interactions per user to keep total pairs manageable for CF
    if max_interactions_per_user is not None:
        before = len(deduped)
        # Sort by play_count desc so we keep the most-played tracks
        deduped = (
            deduped
            .sort_values(["user_id", "play_count"], ascending=[True, False])
            .groupby("user_id", group_keys=False)
            .head(max_interactions_per_user)
            .reset_index(drop=True)
        )
        # Re-filter users that now have < 2 interactions (edge case)
        user_n2 = deduped.groupby("user_id").size()
        valid2 = user_n2[user_n2 >= 2].index
        deduped = deduped[deduped["user_id"].isin(valid2)].copy()
        after = len(deduped)
        log.info(
            "    [ranker] Interaction cap ≤%d/user: %d → %d pairs",
            max_interactions_per_user, before, after,
        )

    log.info(
        "    [ranker] Training groups: %d users | %d pairs",
        deduped["user_id"].nunique(), len(deduped),
    )

    # Assign binary labels: 1 = play_count >= user median, 0 otherwise
    user_medians = deduped.groupby("user_id")["play_count"].median().rename("_med")
    deduped = deduped.join(user_medians, on="user_id")
    deduped["label"] = (deduped["play_count"] >= deduped["_med"]).astype(int)

    pairs = deduped[["user_id", "track_id"]].copy()

    # Build features using train_80 as context
    feat_df = _build_training_features(pairs, train_80, tracks, config)

    # Re-join labels (merge may reorder)
    feat_df = feat_df.merge(
        deduped[["user_id", "track_id", "label"]],
        on=["user_id", "track_id"],
        how="left",
    )
    feat_df["label"] = feat_df["label"].fillna(0).astype(int)

    # Sort by user_id — LambdaRank requires contiguous groups
    feat_df = feat_df.sort_values("user_id").reset_index(drop=True)

    labels = feat_df["label"].values
    groups = (
        feat_df.groupby("user_id", sort=False)["user_id"]
        .count()
        .values
    )

    return feat_df, labels, groups


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_lgbm(
    pair_features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    config: dict[str, Any],
) -> Any:
    """Fit a LightGBM LambdaRank model.

    Parameters
    ----------
    pair_features : pd.DataFrame
        After median imputation; must contain all FEATURE_COLS.
    labels : np.ndarray
        Binary relevance labels aligned with pair_features rows.
    groups : np.ndarray
        Group sizes (sum must equal len(pair_features)).
    config : dict
        Reads section ``lgbm`` for hyper-parameters.

    Returns
    -------
    lgb.LGBMRanker — fitted model.
    """
    import lightgbm as lgb

    lgbm_cfg = config.get("lgbm", {})
    n_estimators  = int(lgbm_cfg.get("n_estimators", 500))
    learning_rate = float(lgbm_cfg.get("learning_rate", 0.05))
    num_leaves    = int(lgbm_cfg.get("num_leaves", 63))
    min_child     = int(lgbm_cfg.get("min_child_samples", 20))
    subsample     = float(lgbm_cfg.get("subsample", 0.8))
    colsample     = float(lgbm_cfg.get("colsample_bytree", 0.8))
    eval_at       = list(lgbm_cfg.get("eval_at", [3]))

    X = pair_features[FEATURE_COLS].values

    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child,
        subsample=subsample,
        colsample_bytree=colsample,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    log.info(
        "    [ranker] Fitting LGBMRanker (n_estimators=%d, num_leaves=%d, "
        "learning_rate=%.3f) on %d pairs across %d groups ...",
        n_estimators, num_leaves, learning_rate, len(X), len(groups),
    )
    model.fit(X, labels, group=groups, eval_at=eval_at)
    return model


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_pairs(
    model: Any,
    pair_features: pd.DataFrame,
    medians: dict[str, float],
) -> pd.Series:
    """Predict relevance scores for (user, track) pairs.

    Parameters
    ----------
    model : LGBMRanker
        Fitted model returned by train_lgbm.
    pair_features : pd.DataFrame
        Feature DataFrame (NaN → median imputed internally).
    medians : dict[str, float]
        Training-set medians for imputation (from _impute on training data).

    Returns
    -------
    pd.Series of float scores, same index as pair_features.
    """
    df_filled, _ = _impute(pair_features, medians)
    X = df_filled[FEATURE_COLS].values
    preds = model.predict(X)
    return pd.Series(preds, index=pair_features.index)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _evaluate(
    model: Any,
    medians: dict[str, float],
    val_users: np.ndarray,
    train_80: pd.DataFrame,
    train_full: pd.DataFrame,
    tracks: pd.DataFrame,
    config: dict[str, Any],
    n_eval_users: int = 3000,
) -> tuple[float, float]:
    """Compute AUC for LGBMRanker vs Rule 2 baseline on held-out val users.

    For each val user:
      - Positives: tracks from their interactions (up to 3).
      - Negatives: unseen tracks sampled 2× positive count.
    Features are built using *train_80* context only (no val leakage).

    Parameters
    ----------
    model : LGBMRanker
    medians : dict
    val_users : np.ndarray — user IDs in the validation split.
    train_80 : pd.DataFrame — 80 % training interactions (context for features).
    train_full : pd.DataFrame — full training set (to identify truly unseen tracks).
    tracks : pd.DataFrame
    config : dict
    n_eval_users : int — cap on number of val users to evaluate.

    Returns
    -------
    (lgbm_auc, rule2_auc) : tuple[float, float]
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        log.warning("scikit-learn not found — skipping AUC evaluation.")
        return 0.0, 0.0

    from src.feature_engineering import (
        build_track_features,
        compute_idf_weights,
        compute_hw4_features,
    )
    from src.feature_engineering import compute_user_track_features
    from src import scorer as scorer_mod

    rng = np.random.default_rng(0)

    # Cap val users
    if len(val_users) > n_eval_users:
        val_users = rng.choice(val_users, size=n_eval_users, replace=False)

    # Build track-level features once for val evaluation (train_80 context)
    track_feats_80 = build_track_features(train_80, tracks)
    idf_80 = compute_idf_weights(train_80, tracks)

    # Pre-index training history per user for fast "unseen" lookup
    full_history: dict[int, set] = (
        train_full.groupby("user_id")["track_id"].apply(set).to_dict()
    )
    all_track_ids = tracks["track_id"].values

    val_interactions = train_full[train_full["user_id"].isin(val_users)]

    eval_rows: list[dict] = []
    eval_labels: list[int] = []

    for uid in val_users:
        pos_tracks = val_interactions.loc[
            val_interactions["user_id"] == uid, "track_id"
        ].unique()
        if len(pos_tracks) == 0:
            continue
        n_pos = min(3, len(pos_tracks))
        pos_sample = rng.choice(pos_tracks, size=n_pos, replace=False)

        seen = full_history.get(int(uid), set())
        unseen_mask = ~np.isin(all_track_ids, list(seen))
        unseen = all_track_ids[unseen_mask]
        if len(unseen) == 0:
            continue
        n_neg = min(n_pos * 2, len(unseen))
        neg_sample = rng.choice(unseen, size=n_neg, replace=False)

        for t in pos_sample:
            eval_rows.append({"user_id": int(uid), "track_id": int(t)})
            eval_labels.append(1)
        for t in neg_sample:
            eval_rows.append({"user_id": int(uid), "track_id": int(t)})
            eval_labels.append(0)

    if not eval_rows:
        log.warning("No eval pairs built; returning AUC=0.")
        return 0.0, 0.0

    eval_pairs = pd.DataFrame(eval_rows)
    labels_arr = np.array(eval_labels)

    log.info(
        "    [ranker] Eval pairs: %d (%d pos, %d neg)",
        len(eval_pairs), int(labels_arr.sum()), int((labels_arr == 0).sum()),
    )

    # Build eval features (train_80 context)
    log.info("    [ranker] Building eval features ...")
    hw4_eval = compute_hw4_features(eval_pairs, train_80, track_feats_80, idf_80)
    content_eval = compute_user_track_features(eval_pairs, train_80, track_feats_80)

    pf_eval = (
        content_eval
        .merge(
            hw4_eval[["user_id", "track_id",
                       "hw4_track_score", "hw4_artist_score", "hw4_album_score",
                       "hw4_genre_score", "hw4_pop_score"]],
            on=["user_id", "track_id"],
            how="left",
        )
        .merge(
            track_feats_80[["track_id", "popularity_score"]],
            on="track_id",
            how="left",
        )
    )
    # Zero-fill CF (not recomputed for speed in validation, mirrors pipeline behaviour)
    for col in ["cf_svd_score", "cf_user_user_score", "cf_item_item_score"]:
        pf_eval[col] = 0.0
    pf_eval["popularity_score"] = pf_eval["popularity_score"].fillna(0.0)

    # LGBMRanker score
    lgbm_raw = score_pairs(model, pf_eval, medians).values
    try:
        lgbm_auc = float(roc_auc_score(labels_arr, lgbm_raw))
    except Exception as exc:
        log.warning("LGBM AUC computation failed: %s", exc)
        lgbm_auc = 0.0

    # Rule 2 baseline on same pairs (CF zeroed — matches pipeline eval protocol)
    rule2_raw = scorer_mod.score_rule2(pf_eval, config).values
    try:
        rule2_auc = float(roc_auc_score(labels_arr, rule2_raw))
    except Exception as exc:
        log.warning("Rule 2 AUC computation failed: %s", exc)
        rule2_auc = 0.0

    return lgbm_auc, rule2_auc


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    data: dict[str, Any],
    warm_pairs: pd.DataFrame,
    test_pair_features: pd.DataFrame,
    cold_pairs: pd.DataFrame,
    features: dict[str, Any],
    config: dict[str, Any],
    submissions_dir: Path | str = "submissions",
) -> None:
    """Train, validate, and produce submission_lgbm.csv.

    Designed to be called from pipeline.py after the main pair_features have
    already been built, so test CF features are not recomputed.

    Parameters
    ----------
    data : dict
        Output of data_loader.load_all() — provides 'train', 'test', 'tracks'.
    warm_pairs : pd.DataFrame
        Test pairs for warm users ['user_id', 'track_id'], row-aligned with
        *test_pair_features*.
    test_pair_features : pd.DataFrame
        Pre-built feature matrix for *warm_pairs* (train_full context, CF
        included) from pipeline._build_pair_features().
    cold_pairs : pd.DataFrame
        Test pairs for cold users (no training history).
    features : dict
        Artifact dict from feature_engineering.run(); provides 'track_features'.
    config : dict
        Parsed config.yaml.
    submissions_dir : Path or str
        Directory to write submission_lgbm.csv.
    """
    _check_lightgbm()

    from src.scorer import write_submission

    submissions_dir = Path(submissions_dir)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    train_full = data["train"]
    tracks     = data["tracks"]
    track_features = features["track_features"]

    lgbm_cfg  = config.get("lgbm", {})
    max_users = lgbm_cfg.get("max_train_users", None)
    if max_users is not None:
        max_users = int(max_users)
    max_ipu_raw = lgbm_cfg.get("max_interactions_per_user", 50)
    max_ipu = int(max_ipu_raw) if max_ipu_raw is not None else None

    # ── 1. 80/20 user split ──────────────────────────────────────────────────
    log.info("LGBM 1/5 — 80/20 user split ...")
    rng = np.random.default_rng(42)
    all_users = train_full["user_id"].unique().copy()
    rng.shuffle(all_users)
    n_train = int(len(all_users) * 0.80)
    train_user_set = set(all_users[:n_train])
    val_user_set   = set(all_users[n_train:])

    train_80   = train_full[train_full["user_id"].isin(train_user_set)].copy()
    log.info(
        "  Train users: %d (%d rows) | Val users: %d",
        len(train_user_set), len(train_80), len(val_user_set),
    )

    # ── 2. Build training data ───────────────────────────────────────────────
    log.info("LGBM 2/5 — Building training data (train_80 context) ...")
    train_pf, train_labels, train_groups = _build_training_data(
        train_80, tracks, config,
        max_users=max_users,
        max_interactions_per_user=max_ipu,
        rng=rng,
    )

    # Median imputation on training features
    train_pf, medians = _impute(train_pf)
    log.info(
        "  Training matrix: %d pairs x %d features",
        len(train_pf), len(FEATURE_COLS),
    )

    # ── 3. Train model ───────────────────────────────────────────────────────
    log.info("LGBM 3/5 — Training model ...")
    model = train_lgbm(train_pf, train_labels, train_groups, config)

    # ── 4. Feature importances ───────────────────────────────────────────────
    importance = model.feature_importances_
    fi = sorted(zip(FEATURE_COLS, importance), key=lambda x: x[1], reverse=True)

    print("\n-- Top 10 Feature Importances (LightGBM, gain) --------------------")
    print(f"  {'Rank':<5}  {'Feature':<28}  {'Importance':>12}")
    print(f"  {'-'*5}  {'-'*28}  {'-'*12}")
    for rank, (feat, imp) in enumerate(fi[:10], 1):
        print(f"  {rank:<5}  {feat:<28}  {imp:>12.1f}")
    print()

    # ── 5. Validate on held-out users ────────────────────────────────────────
    val_users = np.array(list(val_user_set))
    n_eval_users = int(lgbm_cfg.get("n_eval_users", 3000))
    log.info(
        "LGBM 4/5 — Evaluating on %d val users (cap=%d) ...",
        len(val_users), n_eval_users,
    )
    lgbm_auc, rule2_auc = _evaluate(
        model, medians,
        val_users, train_80, train_full, tracks, config,
        n_eval_users=n_eval_users,
    )

    print("-- Validation AUC (80/20 user split, CF zeroed for speed) ---------")
    print(f"  {'Model':<25}  {'AUC':>8}")
    print(f"  {'-'*25}  {'-'*8}")
    print(f"  {'Rule 2 (baseline)':<25}  {rule2_auc:>8.4f}")
    print(f"  {'LightGBM LambdaRank':<25}  {lgbm_auc:>8.4f}")
    delta = lgbm_auc - rule2_auc
    arrow = "^" if delta >= 0 else "v"
    print(f"  {'Delta vs Rule 2':<25}  {delta:>+8.4f}  ({arrow})")
    print()

    # ── 6. Score test pairs and write submission ─────────────────────────────
    log.info("LGBM 5/5 — Scoring %d test pairs and writing submission ...", len(warm_pairs))

    # Warm users: score using pre-built test_pair_features (train_full + CF)
    if not warm_pairs.empty and not test_pair_features.empty:
        warm_scores = score_pairs(model, test_pair_features, medians)
    else:
        warm_scores = pd.Series(dtype=float)

    # Cold users: fall back to popularity score
    pop_map = track_features.set_index("track_id")["popularity_score"].to_dict()
    cold_scores = cold_pairs["track_id"].map(pop_map).fillna(0.0).reset_index(drop=True)

    # Concatenate
    all_pairs = pd.concat(
        [warm_pairs.reset_index(drop=True), cold_pairs.reset_index(drop=True)],
        ignore_index=True,
    )
    all_scores = pd.concat(
        [warm_scores.reset_index(drop=True), cold_scores],
        ignore_index=True,
    )

    soft_rank_probs = config.get("pipeline", {}).get(
        "soft_rank_probs", [0.99, 0.95, 0.90, 0.10, 0.05, 0.01]
    )
    out_path = submissions_dir / "submission_lgbm.csv"
    write_submission(
        all_pairs, all_scores, out_path,
        top_n=int(config.get("pipeline", {}).get("top_n", 100)),
        soft_rank_probs=soft_rank_probs,
    )
    log.info("  Written -> %s  (%d rows)", out_path, len(all_pairs))
    print(f"Submission written: {out_path}")


# ===========================================================================
# V2 — Hard-negative mining + LambdaRank with rule meta-features
# V3 — Hard-negative mining + Binary LGBMClassifier
# ===========================================================================

# Extended feature set: adds Rule 1 and Rule 2 scores as meta-features so
# that LightGBM can learn to re-rank on top of the best heuristic.
FEATURE_COLS_V2: list[str] = FEATURE_COLS + ["rule1_score", "rule2_score"]


def _add_rule_scores(
    pf: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Append rule1_score and rule2_score columns to a pair-features DataFrame.

    Parameters
    ----------
    pf : pd.DataFrame
        Must contain all columns used by score_rule1 / score_rule2.
    config : dict

    Returns
    -------
    pd.DataFrame with two extra columns appended.
    """
    from src import scorer as scorer_mod
    out = pf.copy()
    out["rule1_score"] = scorer_mod.score_rule1(pf, config).values
    out["rule2_score"] = scorer_mod.score_rule2(pf, config).values
    return out


def _build_hard_neg_training_data(
    train_full: pd.DataFrame,
    tracks: pd.DataFrame,
    config: dict[str, Any],
    max_users: int | None = None,
    rng: np.random.Generator | None = None,
    extra_feat_fn=None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build training data that mimics the 6-candidate test structure.

    Instead of using all user interactions, for each user we:
      1. Hold out 20% of interactions as positives (interaction-level split).
      2. Sample up to 3 held-out tracks as positives.
      3. Find hard negatives: tracks sharing artist or album with the positives
         that the user has never interacted with (up to 3).  If not enough
         hard negatives exist, pad with random unseen tracks.
      4. Group = positives + hard negatives (~6 candidates, matching test).

    Features are built on the 80% context (no leakage).  rule1_score and
    rule2_score are then appended as meta-features.

    Parameters
    ----------
    train_full : pd.DataFrame
        Full training interactions.
    tracks : pd.DataFrame
        Track metadata with columns track_id, album_id, artist_id.
    config : dict
    max_users : int or None
        Cap on training users.
    rng : np.random.Generator or None

    Returns
    -------
    pair_features : pd.DataFrame — sorted by user_id, includes FEATURE_COLS_V2.
    labels : np.ndarray (int)
    groups : np.ndarray (int) — group sizes aligned with pair_features rows.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # ── 1. Interaction-level 80/20 split per user ─────────────────────────
    train_80_parts: list[pd.DataFrame] = []
    holdout_parts: list[pd.DataFrame] = []

    for _, grp in train_full.groupby("user_id"):
        n = len(grp)
        if n < 2:
            continue
        n_hold = max(1, round(n * 0.20))
        perm = rng.permutation(n)
        holdout_parts.append(grp.iloc[perm[:n_hold]])
        train_80_parts.append(grp.iloc[perm[n_hold:]])

    if not train_80_parts:
        raise ValueError("No users with >= 2 interactions.")

    train_80 = pd.concat(train_80_parts, ignore_index=True)
    holdout_df = pd.concat(holdout_parts, ignore_index=True)

    # ── 2. Fast lookup structures ─────────────────────────────────────────
    track_to_artist: dict[int, int] = {}
    track_to_album: dict[int, int] = {}
    artist_to_tracks: dict[int, list[int]] = {}
    album_to_tracks: dict[int, list[int]] = {}

    for _, row in tracks.iterrows():
        tid = int(row["track_id"])
        artist_id = row.get("artist_id")
        album_id  = row.get("album_id")
        if pd.notna(artist_id):
            aid = int(artist_id)
            track_to_artist[tid] = aid
            artist_to_tracks.setdefault(aid, []).append(tid)
        if pd.notna(album_id):
            alb = int(album_id)
            track_to_album[tid] = alb
            album_to_tracks.setdefault(alb, []).append(tid)

    all_track_ids_arr = tracks["track_id"].values
    full_history: dict[int, set] = (
        train_full.groupby("user_id")["track_id"].apply(set).to_dict()
    )

    # ── 3. Build pairs per user ───────────────────────────────────────────
    unique_users = holdout_df["user_id"].unique()
    if max_users is not None and len(unique_users) > max_users:
        unique_users = rng.choice(unique_users, size=int(max_users), replace=False)
        log.info(
            "    [ranker-v2] Sampled %d / %d users (max_train_users cap)",
            max_users, holdout_df["user_id"].nunique(),
        )

    pair_rows: list[dict] = []
    label_list: list[int] = []
    group_sizes: list[int] = []

    for uid in unique_users:
        user_seen = full_history.get(int(uid), set())
        pos_tracks = holdout_df.loc[holdout_df["user_id"] == uid, "track_id"].unique()
        if len(pos_tracks) == 0:
            continue

        n_pos = min(3, len(pos_tracks))
        pos_sample = rng.choice(pos_tracks, size=n_pos, replace=False)

        # Hard negatives: same artist or album as positives, not seen
        hard_cands: set[int] = set()
        for pt in pos_sample:
            artist_id = track_to_artist.get(int(pt))
            album_id  = track_to_album.get(int(pt))
            if artist_id is not None:
                hard_cands.update(artist_to_tracks.get(artist_id, []))
            if album_id is not None:
                hard_cands.update(album_to_tracks.get(album_id, []))
        hard_cands -= user_seen
        hard_cands -= set(int(t) for t in pos_sample)
        hard_arr = np.array(list(hard_cands))

        n_neg_target = 3
        if len(hard_arr) >= n_neg_target:
            neg_sample = rng.choice(hard_arr, size=n_neg_target, replace=False)
        else:
            neg_sample = hard_arr.copy()
            remaining = n_neg_target - len(neg_sample)
            excluded = user_seen | set(int(t) for t in pos_sample) | hard_cands
            unseen_mask = ~np.isin(all_track_ids_arr, list(excluded))
            unseen_arr = all_track_ids_arr[unseen_mask]
            if len(unseen_arr) > 0:
                pad = rng.choice(
                    unseen_arr,
                    size=min(remaining, len(unseen_arr)),
                    replace=False,
                )
                neg_sample = np.concatenate([neg_sample, pad])

        if len(neg_sample) == 0:
            continue

        group_size = n_pos + len(neg_sample)
        group_sizes.append(group_size)
        for t in pos_sample:
            pair_rows.append({"user_id": int(uid), "track_id": int(t)})
            label_list.append(1)
        for t in neg_sample:
            pair_rows.append({"user_id": int(uid), "track_id": int(t)})
            label_list.append(0)

    if not pair_rows:
        raise ValueError("No training pairs built for hard-negative training.")

    pairs_df = pd.DataFrame(pair_rows)
    labels_arr = np.array(label_list, dtype=int)

    log.info(
        "    [ranker-v2] %d users | %d pairs | %.1f%% positive",
        len(group_sizes), len(pairs_df), 100.0 * labels_arr.mean(),
    )

    # ── 4. Build features (train_80 context) ──────────────────────────────
    pairs_df["_label"] = labels_arr
    feat_df = _build_training_features(pairs_df[["user_id", "track_id"]], train_80, tracks, config)

    # Append rule meta-features
    feat_df = _add_rule_scores(feat_df, config)

    # Optional extra features (e.g. extended v4 features)
    if extra_feat_fn is not None:
        feat_df = extra_feat_fn(feat_df, train_80, tracks)

    # Re-join labels after potential merge reordering
    feat_df = feat_df.merge(
        pairs_df[["user_id", "track_id", "_label"]],
        on=["user_id", "track_id"],
        how="left",
    )
    feat_df["_label"] = feat_df["_label"].fillna(0).astype(int)

    # Sort by user_id — required for LambdaRank group alignment
    feat_df = feat_df.sort_values("user_id").reset_index(drop=True)
    labels_final = feat_df["_label"].values
    groups_final = (
        feat_df.groupby("user_id", sort=False)["user_id"].count().values
    )

    return feat_df, labels_final, groups_final


def _train_lgbm_ranker_v2(
    pair_features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    config: dict[str, Any],
) -> Any:
    """Fit LGBMRanker on FEATURE_COLS_V2 (includes rule meta-features)."""
    import lightgbm as lgb

    lgbm_cfg = config.get("lgbm", {})
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=int(lgbm_cfg.get("n_estimators", 500)),
        learning_rate=float(lgbm_cfg.get("learning_rate", 0.05)),
        num_leaves=int(lgbm_cfg.get("num_leaves", 63)),
        min_child_samples=int(lgbm_cfg.get("min_child_samples", 20)),
        subsample=float(lgbm_cfg.get("subsample", 0.8)),
        colsample_bytree=float(lgbm_cfg.get("colsample_bytree", 0.8)),
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    df_filled, medians = _impute(pair_features)
    # Ensure all FEATURE_COLS_V2 exist (fill missing with 0)
    for col in FEATURE_COLS_V2:
        if col not in df_filled.columns:
            df_filled[col] = 0.0
    X = df_filled[FEATURE_COLS_V2].values
    log.info(
        "    [ranker-v2] Fitting LGBMRanker on %d pairs x %d features ...",
        len(X), len(FEATURE_COLS_V2),
    )
    model.fit(X, labels, group=groups)
    return model, medians


def _train_lgbm_classifier(
    pair_features: pd.DataFrame,
    labels: np.ndarray,
    config: dict[str, Any],
) -> tuple[Any, dict[str, float]]:
    """Fit LGBMClassifier (binary) on FEATURE_COLS_V2."""
    import lightgbm as lgb

    lgbm_cfg = config.get("lgbm", {})
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=int(lgbm_cfg.get("n_estimators", 500)),
        learning_rate=float(lgbm_cfg.get("learning_rate", 0.05)),
        num_leaves=int(lgbm_cfg.get("num_leaves", 63)),
        min_child_samples=int(lgbm_cfg.get("min_child_samples", 20)),
        subsample=float(lgbm_cfg.get("subsample", 0.8)),
        colsample_bytree=float(lgbm_cfg.get("colsample_bytree", 0.8)),
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        scale_pos_weight=float((labels == 0).sum()) / max(1, int(labels.sum())),
    )
    df_filled, medians = _impute(pair_features)
    for col in FEATURE_COLS_V2:
        if col not in df_filled.columns:
            df_filled[col] = 0.0
    X = df_filled[FEATURE_COLS_V2].values
    log.info(
        "    [ranker-v3] Fitting LGBMClassifier on %d pairs x %d features ...",
        len(X), len(FEATURE_COLS_V2),
    )
    model.fit(X, labels)
    return model, medians


def _score_pairs_v2(
    model: Any,
    pair_features: pd.DataFrame,
    medians: dict[str, float],
    use_proba: bool = False,
) -> pd.Series:
    """Score pairs using FEATURE_COLS_V2; optionally use predict_proba (v3)."""
    df_filled, _ = _impute(pair_features, medians)
    for col in FEATURE_COLS_V2:
        if col not in df_filled.columns:
            df_filled[col] = medians.get(col, 0.0)
    X = df_filled[FEATURE_COLS_V2].values
    if use_proba:
        preds = model.predict_proba(X)[:, 1]
    else:
        preds = model.predict(X)
    return pd.Series(preds, index=pair_features.index)


def _print_feature_importances_v2(model: Any, label: str = "v2") -> None:
    importance = model.feature_importances_
    fi = sorted(zip(FEATURE_COLS_V2, importance), key=lambda x: x[1], reverse=True)
    print(f"\n-- Top 10 Feature Importances (LightGBM {label}) --------------------")
    print(f"  {'Rank':<5}  {'Feature':<28}  {'Importance':>12}")
    print(f"  {'-'*5}  {'-'*28}  {'-'*12}")
    for rank, (feat, imp) in enumerate(fi[:10], 1):
        print(f"  {rank:<5}  {feat:<28}  {imp:>12.1f}")
    print()


def _run_v2_v3_common(
    variant: str,
    data: dict[str, Any],
    warm_pairs: pd.DataFrame,
    test_pair_features: pd.DataFrame,
    cold_pairs: pd.DataFrame,
    features: dict[str, Any],
    config: dict[str, Any],
    submissions_dir: Path | str,
) -> None:
    """Shared logic for run_v2 (LambdaRank) and run_v3 (binary classifier).

    Parameters
    ----------
    variant : str
        'v2' for LambdaRank, 'v3' for binary classifier.
    """
    _check_lightgbm()
    from src.scorer import write_submission

    submissions_dir = Path(submissions_dir)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    train_full    = data["train"]
    tracks        = data["tracks"]
    track_features = features["track_features"]

    lgbm_cfg  = config.get("lgbm", {})
    max_users = lgbm_cfg.get("max_train_users", None)
    if max_users is not None:
        max_users = int(max_users)

    rng = np.random.default_rng(42)

    # ── 1. Build hard-negative training data ─────────────────────────────
    log.info("%s 1/4 — Building hard-negative training data ...", variant.upper())
    train_pf, train_labels, train_groups = _build_hard_neg_training_data(
        train_full, tracks, config,
        max_users=max_users,
        rng=rng,
    )

    # ── 2. Train model ────────────────────────────────────────────────────
    log.info("%s 2/4 — Training model ...", variant.upper())
    if variant == "v2":
        model, medians = _train_lgbm_ranker_v2(train_pf, train_labels, train_groups, config)
    else:
        model, medians = _train_lgbm_classifier(train_pf, train_labels, config)

    # ── 3. Feature importances ────────────────────────────────────────────
    _print_feature_importances_v2(model, label=variant)

    # ── 4. Score test pairs ───────────────────────────────────────────────
    log.info("%s 3/4 — Scoring test pairs ...", variant.upper())

    use_proba = (variant == "v3")

    # Warm: add rule scores to pre-built features
    if not warm_pairs.empty and not test_pair_features.empty:
        test_pf_v2 = _add_rule_scores(test_pair_features, config)
        warm_scores = _score_pairs_v2(model, test_pf_v2, medians, use_proba=use_proba)
    else:
        warm_scores = pd.Series(dtype=float)

    # Cold: popularity fallback
    pop_map = track_features.set_index("track_id")["popularity_score"].to_dict()
    cold_scores = cold_pairs["track_id"].map(pop_map).fillna(0.0).reset_index(drop=True)

    all_pairs = pd.concat(
        [warm_pairs.reset_index(drop=True), cold_pairs.reset_index(drop=True)],
        ignore_index=True,
    )
    all_scores = pd.concat(
        [warm_scores.reset_index(drop=True), cold_scores],
        ignore_index=True,
    )

    soft_rank_probs = config.get("pipeline", {}).get(
        "soft_rank_probs", [0.99, 0.95, 0.90, 0.10, 0.05, 0.01]
    )
    out_path = submissions_dir / f"submission_lgbm_{variant}.csv"
    log.info("%s 4/4 — Writing submission ...", variant.upper())
    write_submission(
        all_pairs, all_scores, out_path,
        top_n=int(config.get("pipeline", {}).get("top_n", 100)),
        soft_rank_probs=soft_rank_probs,
    )
    print(f"Submission written: {out_path}")


def run_v2(
    data: dict[str, Any],
    warm_pairs: pd.DataFrame,
    test_pair_features: pd.DataFrame,
    cold_pairs: pd.DataFrame,
    features: dict[str, Any],
    config: dict[str, Any],
    submissions_dir: Path | str = "submissions",
) -> None:
    """LambdaRank with hard-negative mining + rule meta-features.

    Trains on groups mimicking the test structure (3 positives + 3 hard
    negatives per user, where hard negatives share artist/album with positives).
    Adds rule1_score and rule2_score as meta-features so LightGBM re-ranks on
    top of the best heuristic.

    Output: submissions/submission_lgbm_v2.csv
    """
    _run_v2_v3_common(
        "v2", data, warm_pairs, test_pair_features,
        cold_pairs, features, config, submissions_dir,
    )


def run_v3(
    data: dict[str, Any],
    warm_pairs: pd.DataFrame,
    test_pair_features: pd.DataFrame,
    cold_pairs: pd.DataFrame,
    features: dict[str, Any],
    config: dict[str, Any],
    submissions_dir: Path | str = "submissions",
) -> None:
    """Binary LGBMClassifier with hard-negative mining + rule meta-features.

    Same training data as v2 but uses binary cross-entropy instead of
    LambdaRank; ranks test candidates by P(relevant=1).

    Output: submissions/submission_lgbm_v3.csv
    """
    _run_v2_v3_common(
        "v3", data, warm_pairs, test_pair_features,
        cold_pairs, features, config, submissions_dir,
    )


# ===========================================================================
# V4 — Hard-negative mining + LambdaRank + extended features + normalisation
# V5 — Hard-negative mining + Binary classifier + extended features + normalisation
# Ensemble — rule2_score * 0.4 + lgbm_v4_score * 0.6
# ===========================================================================

# 29-feature set: V2 base (23) + 6 engineered features
FEATURE_COLS_V4: list[str] = FEATURE_COLS_V2 + [
    "rule3_score",
    "score_rank",
    "album_artist_combined",
    "genre_coverage_ratio",
    "user_activity_score",
    "track_global_rank",
]


def _lgbm_progress_callback(n_total: int, period: int = 100):
    """Return a LightGBM callback that logs every *period* iterations."""
    def _cb(env):
        it = env.iteration
        if it > 0 and it % period == 0:
            log.info("    [lgbm] iteration %d / %d", it, n_total)
    return _cb


def _add_extended_features(
    pf: pd.DataFrame,
    train_context: pd.DataFrame,
    track_features: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Append 6 engineered features to a pair-features DataFrame.

    Adds rule3_score, score_rank, album_artist_combined, genre_coverage_ratio,
    user_activity_score, track_global_rank.

    Parameters
    ----------
    pf : pd.DataFrame
        Already contains FEATURE_COLS_V2 (rule1_score / rule2_score present).
    train_context : pd.DataFrame
        Training interactions for this context (train_80 or train_full).
    track_features : pd.DataFrame
        Full track features for global popularity ranking.
    config : dict
    """
    from src import scorer as scorer_mod

    out = pf.copy()

    # rule3_score
    out["rule3_score"] = scorer_mod.score_rule3(out, config).values

    # score_rank: rank of rule2_score within the user's candidate group (1 = best)
    if "rule2_score" in out.columns:
        out["score_rank"] = (
            out.groupby("user_id")["rule2_score"]
            .rank(ascending=False, method="first")
            .astype(float)
        )
    else:
        out["score_rank"] = 1.0

    # album_artist_combined
    alb = (
        out["album_score"] if "album_score" in out.columns
        else pd.Series(0.0, index=out.index)
    ).fillna(0.0)
    art = (
        out["artist_score"] if "artist_score" in out.columns
        else pd.Series(0.0, index=out.index)
    ).fillna(0.0)
    out["album_artist_combined"] = alb * 0.5 + art * 0.5

    # genre_coverage_ratio: fraction of user genre preferences the track covers
    nz = (
        out["genre_nonzero_count"] if "genre_nonzero_count" in out.columns
        else pd.Series(0.0, index=out.index)
    ).fillna(0.0)
    gc = (
        out["genre_count"] if "genre_count" in out.columns
        else pd.Series(1.0, index=out.index)
    ).fillna(1.0).replace(0, 1.0)
    out["genre_coverage_ratio"] = nz / gc

    # user_activity_score: log1p(total interactions per user in context)
    act = (
        train_context.groupby("user_id").size()
        .rename("_act").reset_index()
    )
    act["user_activity_score"] = np.log1p(act["_act"])
    out = out.merge(act[["user_id", "user_activity_score"]], on="user_id", how="left")
    out["user_activity_score"] = out["user_activity_score"].fillna(0.0)

    # track_global_rank: 1-based rank by descending global popularity
    if track_features is not None and "popularity_score" in track_features.columns:
        grank = track_features[["track_id", "popularity_score"]].copy()
        grank["track_global_rank"] = (
            grank["popularity_score"].rank(ascending=False, method="first").astype(float)
        )
        rank_map = grank.set_index("track_id")["track_global_rank"].to_dict()
        out["track_global_rank"] = (
            out["track_id"].map(rank_map).fillna(float(len(rank_map) + 1))
        )
    else:
        out["track_global_rank"] = 1.0

    return out


def _normalize_features(
    df: pd.DataFrame,
    cols: list[str],
    mins: dict[str, float] | None = None,
    maxs: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    """Min-max normalise *cols* to [0, 1].

    Pass *mins* / *maxs* from training to apply the same scaling to test data.

    Returns
    -------
    (normalised_df, mins, maxs)
    """
    out = df.copy()
    if mins is None:
        mins = {}
        maxs = {}
        for c in cols:
            if c in out.columns:
                vals = out[c].fillna(0.0)
                mins[c] = float(vals.min())
                maxs[c] = float(vals.max())
            else:
                mins[c] = 0.0
                maxs[c] = 1.0

    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
        lo = mins.get(c, 0.0)
        hi = maxs.get(c, 1.0)
        rng = hi - lo
        if rng > 1e-12:
            out[c] = (out[c].fillna(lo) - lo) / rng
        else:
            out[c] = 0.0

    return out, mins, maxs


def _impute_v4(
    df: pd.DataFrame,
    medians: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Impute NaN in FEATURE_COLS_V4 with column medians."""
    out = df.copy()
    if medians is None:
        medians = {}
        for col in FEATURE_COLS_V4:
            if col in out.columns:
                m = float(out[col].median())
                medians[col] = 0.0 if np.isnan(m) else m
            else:
                medians[col] = 0.0

    for col in FEATURE_COLS_V4:
        if col in out.columns:
            out[col] = out[col].fillna(medians.get(col, 0.0))
        else:
            out[col] = medians.get(col, 0.0)

    return out, medians


def _train_lgbm_ranker_v4(
    pair_features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    config: dict[str, Any],
) -> tuple[Any, dict[str, float], dict[str, float], dict[str, float]]:
    """Fit LGBMRanker on FEATURE_COLS_V4 with normalisation and progress logging."""
    import lightgbm as lgb

    lgbm_cfg = config.get("lgbm", {})
    n_est = int(lgbm_cfg.get("n_estimators", 1000))
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=n_est,
        learning_rate=float(lgbm_cfg.get("learning_rate", 0.02)),
        num_leaves=int(lgbm_cfg.get("num_leaves", 127)),
        min_child_samples=int(lgbm_cfg.get("min_child_samples", 10)),
        subsample=float(lgbm_cfg.get("subsample", 0.8)),
        colsample_bytree=float(lgbm_cfg.get("colsample_bytree", 0.8)),
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    df_filled, medians = _impute_v4(pair_features)
    for col in FEATURE_COLS_V4:
        if col not in df_filled.columns:
            df_filled[col] = 0.0
    df_norm, mins, maxs = _normalize_features(df_filled, FEATURE_COLS_V4)
    X = df_norm[FEATURE_COLS_V4].values

    n_pairs = len(X)
    log.info(
        "    [ranker-v4] Fitting LGBMRanker: %d pairs × %d features × %d trees",
        n_pairs, len(FEATURE_COLS_V4), n_est,
    )
    est_min = max(1, round(n_pairs * n_est / 2_400_000))
    log.info("    [ranker-v4] Estimated time: %d – %d min", est_min, est_min * 3)

    model.fit(
        X, labels, group=groups,
        callbacks=[_lgbm_progress_callback(n_est, period=100)],
    )
    return model, medians, mins, maxs


def _train_lgbm_classifier_v5(
    pair_features: pd.DataFrame,
    labels: np.ndarray,
    config: dict[str, Any],
) -> tuple[Any, dict[str, float], dict[str, float], dict[str, float]]:
    """Fit LGBMClassifier on FEATURE_COLS_V4 with normalisation and progress logging."""
    import lightgbm as lgb

    lgbm_cfg = config.get("lgbm", {})
    n_est = int(lgbm_cfg.get("n_estimators", 1000))
    n_neg = int((labels == 0).sum())
    n_pos = max(1, int(labels.sum()))
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_est,
        learning_rate=float(lgbm_cfg.get("learning_rate", 0.02)),
        num_leaves=int(lgbm_cfg.get("num_leaves", 127)),
        min_child_samples=int(lgbm_cfg.get("min_child_samples", 10)),
        subsample=float(lgbm_cfg.get("subsample", 0.8)),
        colsample_bytree=float(lgbm_cfg.get("colsample_bytree", 0.8)),
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        scale_pos_weight=float(n_neg) / n_pos,
    )
    df_filled, medians = _impute_v4(pair_features)
    for col in FEATURE_COLS_V4:
        if col not in df_filled.columns:
            df_filled[col] = 0.0
    df_norm, mins, maxs = _normalize_features(df_filled, FEATURE_COLS_V4)
    X = df_norm[FEATURE_COLS_V4].values

    n_pairs = len(X)
    log.info(
        "    [ranker-v5] Fitting LGBMClassifier: %d pairs × %d features × %d trees",
        n_pairs, len(FEATURE_COLS_V4), n_est,
    )
    est_min = max(1, round(n_pairs * n_est / 2_400_000))
    log.info("    [ranker-v5] Estimated time: %d – %d min", est_min, est_min * 3)

    model.fit(
        X, labels,
        callbacks=[_lgbm_progress_callback(n_est, period=100)],
    )
    return model, medians, mins, maxs


def _score_pairs_v4(
    model: Any,
    pair_features: pd.DataFrame,
    medians: dict[str, float],
    mins: dict[str, float],
    maxs: dict[str, float],
    use_proba: bool = False,
) -> pd.Series:
    """Score (user, track) pairs using FEATURE_COLS_V4 with saved normalisation."""
    df_filled, _ = _impute_v4(pair_features, medians)
    for col in FEATURE_COLS_V4:
        if col not in df_filled.columns:
            df_filled[col] = medians.get(col, 0.0)
    df_norm, _, _ = _normalize_features(df_filled, FEATURE_COLS_V4, mins=mins, maxs=maxs)
    X = df_norm[FEATURE_COLS_V4].values
    if use_proba:
        preds = model.predict_proba(X)[:, 1]
    else:
        preds = model.predict(X)
    return pd.Series(preds, index=pair_features.index)


def _print_feature_importances_v4(model: Any, label: str = "v4") -> None:
    """Print top-15 feature importances for a V4/V5 model."""
    importance = model.feature_importances_
    fi = sorted(zip(FEATURE_COLS_V4, importance), key=lambda x: x[1], reverse=True)
    print(f"\n-- Top 15 Feature Importances (LightGBM {label}) --------------------")
    print(f"  {'Rank':<5}  {'Feature':<30}  {'Importance':>12}")
    print(f"  {'-'*5}  {'-'*30}  {'-'*12}")
    for rank, (feat, imp) in enumerate(fi[:15], 1):
        print(f"  {rank:<5}  {feat:<30}  {imp:>12.1f}")
    print()


def _run_v4_v5_common(
    variant: str,
    data: dict[str, Any],
    warm_pairs: pd.DataFrame,
    test_pair_features: pd.DataFrame,
    cold_pairs: pd.DataFrame,
    features: dict[str, Any],
    config: dict[str, Any],
    submissions_dir: Path | str,
    write_file: bool = True,
) -> dict[str, Any]:
    """Shared logic for v4 (LambdaRank) and v5 (binary classifier).

    Parameters
    ----------
    write_file : bool
        Write the submission CSV when True (default).

    Returns
    -------
    dict with keys: warm_pairs, warm_scores, cold_pairs, cold_scores,
                    model, medians, mins, maxs, test_pf_v4.
    """
    _check_lightgbm()
    from src.scorer import write_submission

    submissions_dir = Path(submissions_dir)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    train_full     = data["train"]
    tracks         = data["tracks"]
    track_features = features["track_features"]

    lgbm_cfg  = config.get("lgbm", {})
    max_users = lgbm_cfg.get("max_train_users", None)
    if max_users is not None:
        max_users = int(max_users)

    rng = np.random.default_rng(42)

    # ── 1. Build hard-negative training data with extended features ───────
    log.info(
        "%s 1/4 — Building hard-negative training data with extended features ...",
        variant.upper(),
    )

    def _ext_fn(feat_df: pd.DataFrame, train_ctx: pd.DataFrame, trks: pd.DataFrame):
        return _add_extended_features(feat_df, train_ctx, track_features, config)

    train_pf, train_labels, train_groups = _build_hard_neg_training_data(
        train_full, tracks, config,
        max_users=max_users,
        rng=rng,
        extra_feat_fn=_ext_fn,
    )

    # ── 2. Train model ────────────────────────────────────────────────────
    log.info("%s 2/4 — Training model ...", variant.upper())
    if variant == "v4":
        model, medians, mins, maxs = _train_lgbm_ranker_v4(
            train_pf, train_labels, train_groups, config
        )
    else:
        model, medians, mins, maxs = _train_lgbm_classifier_v5(
            train_pf, train_labels, config
        )

    # ── 3. Feature importances ────────────────────────────────────────────
    _print_feature_importances_v4(model, label=variant)

    # ── 4. Score test pairs ───────────────────────────────────────────────
    log.info("%s 3/4 — Scoring test pairs ...", variant.upper())
    use_proba = (variant == "v5")

    if not warm_pairs.empty and not test_pair_features.empty:
        test_pf_v4 = _add_rule_scores(test_pair_features, config)
        test_pf_v4 = _add_extended_features(test_pf_v4, train_full, track_features, config)
        warm_scores = _score_pairs_v4(
            model, test_pf_v4, medians, mins, maxs, use_proba=use_proba
        )
    else:
        test_pf_v4 = pd.DataFrame()
        warm_scores = pd.Series(dtype=float)

    # Cold: popularity fallback
    pop_map = track_features.set_index("track_id")["popularity_score"].to_dict()
    cold_scores = (
        cold_pairs["track_id"].map(pop_map).fillna(0.0).reset_index(drop=True)
    )

    if write_file:
        all_pairs = pd.concat(
            [warm_pairs.reset_index(drop=True), cold_pairs.reset_index(drop=True)],
            ignore_index=True,
        )
        all_scores = pd.concat(
            [warm_scores.reset_index(drop=True), cold_scores],
            ignore_index=True,
        )
        soft_rank_probs = config.get("pipeline", {}).get(
            "soft_rank_probs", [0.99, 0.95, 0.90, 0.10, 0.05, 0.01]
        )
        out_path = submissions_dir / f"submission_lgbm_{variant}.csv"
        log.info("%s 4/4 — Writing submission ...", variant.upper())
        write_submission(
            all_pairs, all_scores, out_path,
            top_n=int(config.get("pipeline", {}).get("top_n", 100)),
            soft_rank_probs=soft_rank_probs,
        )
        print(f"Submission written: {out_path}")

    return {
        "warm_pairs":  warm_pairs,
        "warm_scores": warm_scores,
        "cold_pairs":  cold_pairs,
        "cold_scores": cold_scores,
        "model":       model,
        "medians":     medians,
        "mins":        mins,
        "maxs":        maxs,
        "test_pf_v4":  test_pf_v4,
    }


def run_v4(
    data: dict[str, Any],
    warm_pairs: pd.DataFrame,
    test_pair_features: pd.DataFrame,
    cold_pairs: pd.DataFrame,
    features: dict[str, Any],
    config: dict[str, Any],
    submissions_dir: Path | str = "submissions",
) -> None:
    """LambdaRank v4: hard-negatives + extended features (29 total) + normalisation.

    Extends v2 with rule3_score, score_rank, album_artist_combined,
    genre_coverage_ratio, user_activity_score, track_global_rank.
    All features are min-max normalised to [0, 1] before training.

    Output: submissions/submission_lgbm_v4.csv
    """
    _run_v4_v5_common(
        "v4", data, warm_pairs, test_pair_features,
        cold_pairs, features, config, submissions_dir,
    )


def run_v5(
    data: dict[str, Any],
    warm_pairs: pd.DataFrame,
    test_pair_features: pd.DataFrame,
    cold_pairs: pd.DataFrame,
    features: dict[str, Any],
    config: dict[str, Any],
    submissions_dir: Path | str = "submissions",
) -> None:
    """Binary LGBMClassifier v5: hard-negatives + extended features + normalisation.

    Same training data as v4 but uses binary cross-entropy; ranks by P(relevant=1).

    Output: submissions/submission_lgbm_v5.csv
    """
    _run_v4_v5_common(
        "v5", data, warm_pairs, test_pair_features,
        cold_pairs, features, config, submissions_dir,
    )


def run_ensemble(
    data: dict[str, Any],
    warm_pairs: pd.DataFrame,
    test_pair_features: pd.DataFrame,
    cold_pairs: pd.DataFrame,
    features: dict[str, Any],
    config: dict[str, Any],
    submissions_dir: Path | str = "submissions",
) -> None:
    """Ensemble: rule2_score * 0.4 + lgbm_v4_score * 0.6.

    Trains v4 (LambdaRank) internally, normalises its output to [0, 1],
    blends with rule2_score, and writes submission_ensemble.csv.

    Output: submissions/submission_ensemble.csv
    """
    from src import scorer as scorer_mod
    from src.scorer import write_submission

    submissions_dir = Path(submissions_dir)

    log.info("ENSEMBLE 1/3 — Training LambdaRank v4 for ensemble ...")
    result = _run_v4_v5_common(
        "v4", data, warm_pairs, test_pair_features,
        cold_pairs, features, config, submissions_dir,
        write_file=False,
    )

    lgbm_warm  = result["warm_scores"].reset_index(drop=True)
    cold_scores = result["cold_scores"]

    log.info("ENSEMBLE 2/3 — Blending rule2*0.4 + lgbm_v4*0.6 ...")
    if not warm_pairs.empty and not test_pair_features.empty:
        rule2_warm = scorer_mod.score_rule2(test_pair_features, config).reset_index(drop=True)
        # Normalise LGBM scores to [0, 1] so the blend weight is meaningful
        lo, hi = float(lgbm_warm.min()), float(lgbm_warm.max())
        lgbm_norm = (lgbm_warm - lo) / max(hi - lo, 1e-12)
        blend_warm = 0.4 * rule2_warm + 0.6 * lgbm_norm
    else:
        blend_warm = pd.Series(dtype=float)

    all_pairs = pd.concat(
        [warm_pairs.reset_index(drop=True), cold_pairs.reset_index(drop=True)],
        ignore_index=True,
    )
    all_scores = pd.concat(
        [blend_warm, cold_scores],
        ignore_index=True,
    )

    soft_rank_probs = config.get("pipeline", {}).get(
        "soft_rank_probs", [0.99, 0.95, 0.90, 0.10, 0.05, 0.01]
    )
    out_path = Path(submissions_dir) / "submission_ensemble.csv"
    log.info("ENSEMBLE 3/3 — Writing submission ...")
    write_submission(
        all_pairs, all_scores, out_path,
        top_n=int(config.get("pipeline", {}).get("top_n", 100)),
        soft_rank_probs=soft_rank_probs,
    )
    print(f"Submission written: {out_path}")
