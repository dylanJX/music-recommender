"""
pipeline.py
===========
Orchestrates the full recommendation pipeline from raw data to submission CSV.

Usage
-----
From the project root::

    python -m src.pipeline [--config config.yaml] [--rule all|rule1|rule2|rule3] [--run-name NAME]

``--rule`` selects which scoring rule(s) to apply:

  all    (default) — produce submission_rule1.csv, submission_rule2.csv,
                     submission_rule3.csv in submissions/
  rule1  — produce submissions/submission_rule1.csv  (Max Genre Focus)
  rule2  — produce submissions/submission_rule2.csv  (Weighted Hybrid)
  rule3  — produce submissions/submission_rule3.csv  (Popularity Boosted Hybrid)

Pipeline steps
--------------
0. Load config.yaml.
1. Load all raw data files via data_loader.load_all().
2. Build feature artifacts via feature_engineering.run().
3. Flatten test file into (user_id, track_id) pairs.
4. Identify cold users (zero training history).
5. Route warm users through the rule-based scorer; cold users through cold_start.
6. Write submission CSV(s) to submissions/.
7. Print a missing-value rate report per feature column.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Ensure project root is importable when run as __main__
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

from src import data_loader, feature_engineering
from src import scorer as scorer_mod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONFIG_PATH = _project_root / "config.yaml"
SUBMISSIONS_DIR = _project_root / "submissions"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Read and return the parsed config.yaml as a plain dict.

    Parameters
    ----------
    path : str or Path
        Path to config.yaml.

    Returns
    -------
    dict
    """
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Cold-start routing helpers
# ---------------------------------------------------------------------------

def _split_cold_warm_users(
    test_pairs: pd.DataFrame,
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split test pairs into cold-user rows and warm-user rows.

    Parameters
    ----------
    test_pairs : pd.DataFrame
        Columns ['user_id', 'track_id'].
    train : pd.DataFrame
        Training interactions.

    Returns
    -------
    cold_pairs, warm_pairs : pd.DataFrame, pd.DataFrame
    """
    train_users = set(train["user_id"].unique())
    is_cold = ~test_pairs["user_id"].isin(train_users)
    return test_pairs[is_cold].copy(), test_pairs[~is_cold].copy()


def _score_cold_users(
    cold_pairs: pd.DataFrame,
    track_features: pd.DataFrame,
) -> pd.Series:
    """Assign popularity-based scores to cold-user (user, track) pairs.

    Parameters
    ----------
    cold_pairs : pd.DataFrame
        Columns ['user_id', 'track_id'] for cold users only.
    track_features : pd.DataFrame
        Enriched track features with 'popularity_score'.

    Returns
    -------
    pd.Series
        Float scores aligned positionally with cold_pairs.
    """
    pop_map = track_features.set_index("track_id")["popularity_score"].to_dict()
    return cold_pairs["track_id"].map(pop_map).fillna(0.0).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pair-feature builder
# ---------------------------------------------------------------------------

def _build_pair_features(
    warm_pairs: pd.DataFrame,
    train: pd.DataFrame,
    features: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Compute merged content + CF + popularity features for warm pairs.

    Parameters
    ----------
    warm_pairs : pd.DataFrame
        (user_id, track_id) pairs for warm users.
    train : pd.DataFrame
        Training interactions.
    features : dict
        Artifact dict from feature_engineering.run().
    config : dict
        Parsed config.yaml.

    Returns
    -------
    pd.DataFrame
        Merged feature DataFrame ready for score_rule*() functions.
    """
    from src.feature_engineering import compute_user_track_features, compute_hw4_features
    from src.collab_features import compute_cf_features

    track_features = features["track_features"]
    idf_weights = features.get("idf_weights", {})

    log.info("  Computing content features for %d pairs ...", len(warm_pairs))
    content = compute_user_track_features(warm_pairs, train, track_features)

    log.info("  Computing hw4-style IDF-weighted features ...")
    hw4 = compute_hw4_features(warm_pairs, train, track_features, idf_weights)

    log.info("  Computing CF features ...")
    cf = compute_cf_features(warm_pairs, train, config)

    pair_features = (
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
            track_features[["track_id", "popularity_score"]],
            on="track_id",
            how="left",
        )
    )
    return pair_features


# ---------------------------------------------------------------------------
# Missing-value report
# ---------------------------------------------------------------------------

def _print_missing_report(pair_features: pd.DataFrame) -> None:
    """Print per-column missing-value rates to stdout."""
    n = len(pair_features)
    if n == 0:
        log.info("pair_features is empty — no missing-value report.")
        return

    feature_cols = [
        c for c in pair_features.columns
        if c not in ("user_id", "track_id")
    ]
    print("\n-- Missing-value rate per feature ------------------------------")
    print(f"  {'column':<30}  {'missing':>8}  {'rate':>8}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}")
    for col in feature_cols:
        n_missing = int(pair_features[col].isna().sum())
        rate = n_missing / n
        print(f"  {col:<30}  {n_missing:>8,}  {rate:>7.2%}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}")
    print(f"  {'Total pairs':<30}  {n:>8,}")
    print()


# ---------------------------------------------------------------------------
# Per-rule scoring and writing
# ---------------------------------------------------------------------------

def _score_and_write(
    rule_name: str,
    warm_pairs: pd.DataFrame,
    cold_pairs: pd.DataFrame,
    pair_features: pd.DataFrame,
    features: dict[str, Any],
    config: dict[str, Any],
    submissions_dir: Path,
) -> None:
    """Score warm + cold pairs with one rule and write a submission CSV.

    Parameters
    ----------
    rule_name : str
        'rule1', 'rule2', or 'rule3'.
    warm_pairs : pd.DataFrame
        (user_id, track_id) for warm users, row-aligned with pair_features.
    cold_pairs : pd.DataFrame
        (user_id, track_id) for cold users.
    pair_features : pd.DataFrame
        Pre-built feature DataFrame for warm_pairs.
    features : dict
        Artifact dict (for cold-start popularity lookup).
    config : dict
        Parsed config.yaml.
    submissions_dir : Path
        Output directory.
    """
    top_n = int(config.get("pipeline", {}).get("top_n", 100))
    soft_rank_probs = config.get("pipeline", {}).get(
        "soft_rank_probs", [0.99, 0.95, 0.90, 0.10, 0.05, 0.01]
    )
    track_features = features["track_features"]

    score_fn = {
        "rule1": scorer_mod.score_rule1,
        "rule2": scorer_mod.score_rule2,
        "rule3": scorer_mod.score_rule3,
    }[rule_name]

    # Score warm users via the requested rule
    warm_scores = score_fn(pair_features, config).reset_index(drop=True)

    # Cold users: popularity-based fallback
    cold_scores = _score_cold_users(cold_pairs, track_features)

    # Concatenate preserving per-user ranking
    all_pairs = pd.concat(
        [warm_pairs.reset_index(drop=True), cold_pairs.reset_index(drop=True)],
        ignore_index=True,
    )
    all_scores = pd.concat(
        [warm_scores, cold_scores],
        ignore_index=True,
    )

    out_path = submissions_dir / f"submission_{rule_name}.csv"
    scorer_mod.write_submission(
        all_pairs, all_scores, out_path,
        top_n=top_n, soft_rank_probs=soft_rank_probs,
    )
    log.info("  Written -> %s  (%d rows)", out_path, len(all_pairs))


# ---------------------------------------------------------------------------
# 80/20 AUC estimation
# ---------------------------------------------------------------------------

def _estimate_auc_80_20(
    data: dict[str, Any],
    config: dict[str, Any],
    n_eval_users: int = 5000,
) -> None:
    """Estimate AUC with an 80/20 per-user hold-out split.

    For each user, 20% of their interactions are held out as positives.
    The model is scored on the held-out positives plus an equal number of
    sampled negatives (tracks not in the user's full history).  Features are
    built only on the 80% training split.

    CF features are skipped here (they take ~5 min) so the AUC estimate
    reflects content-only scoring for Rules 1-3 with CF weights zeroed out.
    The true Rule 2/3 AUC on the full pipeline will be higher.

    Parameters
    ----------
    data : dict
        Output of data_loader.load_all(), keys 'train' and 'tracks'.
    config : dict
        Parsed config.yaml.
    n_eval_users : int
        Maximum number of users to evaluate (sampled randomly).
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        log.warning("  scikit-learn not found — skipping AUC evaluation.")
        return

    from src.feature_engineering import (
        build_track_features, compute_idf_weights, compute_hw4_features,
    )

    train = data["train"]
    tracks = data["tracks"]

    log.info("  Splitting train 80/20 per user ...")
    rng = np.random.default_rng(42)

    train_80_parts: list[pd.DataFrame] = []
    holdout_parts: list[pd.DataFrame] = []
    for _, grp in train.groupby("user_id"):
        n = len(grp)
        n_hold = max(1, round(n * 0.2))
        perm = rng.permutation(n)
        holdout_parts.append(grp.iloc[perm[:n_hold]])
        train_80_parts.append(grp.iloc[perm[n_hold:]])

    train_80 = pd.concat(train_80_parts, ignore_index=True)
    holdout = pd.concat(holdout_parts, ignore_index=True)
    log.info(
        "  train_80: %d rows | holdout: %d rows",
        len(train_80), len(holdout),
    )

    # Build features on train_80 (no CF — too slow)
    log.info("  Building track features and IDF weights on 80%% split ...")
    track_features_80 = build_track_features(train_80, tracks)
    idf_80 = compute_idf_weights(train_80, tracks)

    # Sample up to n_eval_users for evaluation
    all_eval_users = holdout["user_id"].unique()
    if len(all_eval_users) > n_eval_users:
        all_eval_users = rng.choice(all_eval_users, size=n_eval_users, replace=False)

    train_80_sets = train_80.groupby("user_id")["track_id"].apply(set).to_dict()
    all_track_ids = tracks["track_id"].values

    log.info("  Building eval pairs for %d users ...", len(all_eval_users))
    eval_rows: list[dict] = []
    eval_labels: list[int] = []

    for uid in all_eval_users:
        pos_tracks = holdout[holdout["user_id"] == uid]["track_id"].unique()
        if len(pos_tracks) == 0:
            continue
        n_pos = min(3, len(pos_tracks))
        pos_sample = rng.choice(pos_tracks, size=n_pos, replace=False)

        seen = train_80_sets.get(int(uid), set()) | set(pos_tracks)
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
        log.warning("  No eval pairs built — skipping AUC.")
        return

    eval_pairs = pd.DataFrame(eval_rows)
    labels = np.array(eval_labels)
    log.info(
        "  Eval pairs: %d total (%d positive, %d negative)",
        len(eval_pairs), int(labels.sum()), int((labels == 0).sum()),
    )

    # Compute hw4 features on eval pairs using train_80
    log.info("  Computing hw4 features for eval pairs ...")
    hw4_feats = compute_hw4_features(eval_pairs, train_80, tracks, idf_80)
    pair_features_eval = hw4_feats.merge(
        track_features_80[["track_id", "popularity_score"]],
        on="track_id", how="left",
    )
    pair_features_eval["popularity_score"] = (
        pair_features_eval["popularity_score"].fillna(0.0)
    )
    # Zero-fill CF columns (not computed for speed)
    for col in ["cf_svd_score", "cf_user_user_score", "cf_item_item_score"]:
        pair_features_eval[col] = 0.0

    print("\n-- 80/20 AUC Estimate (content only; CF zeroed out for speed) ----")
    print(f"  {'Rule':<8}  {'AUC':>8}")
    print(f"  {'-'*8}  {'-'*8}")
    for rule_name, score_fn in [
        ("rule1", scorer_mod.score_rule1),
        ("rule2", scorer_mod.score_rule2),
        ("rule3", scorer_mod.score_rule3),
    ]:
        scores = score_fn(pair_features_eval, config).values
        try:
            auc = roc_auc_score(labels, scores)
            print(f"  {rule_name:<8}  {auc:>8.4f}")
        except Exception as exc:
            print(f"  {rule_name:<8}  FAILED ({exc})")
    print(f"  Note: hw4.py baseline achieved 0.851 (with CF, Rule 2/3 will be higher)")
    print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(config: dict[str, Any], rule: str = "all") -> None:
    """Execute the full recommendation pipeline.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.
    rule : str
        'all' | 'rule1' | 'rule2' | 'rule3'.
    """
    t0 = time.time()
    data_dir = Path(config.get("data", {}).get("dir", "data/"))
    submissions_dir = Path("submissions")
    submissions_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 0: Load data ─────────────────────────────────────────────────
    log.info("Step 0 — Loading data from %s …", data_dir)
    data = data_loader.load_all(data_dir, config=config)
    train = data["train"]
    test  = data["test"]
    log.info(
        "  train: %d interactions, %d users | test: %d pairs, %d users",
        len(train), train["user_id"].nunique(),
        len(test),  test["user_id"].nunique(),
    )

    # ── Step 1: Build features ────────────────────────────────────────────
    log.info("Step 1 — Building feature artifacts …")
    features = feature_engineering.run(data)
    track_features = features["track_features"]
    log.info(
        "  user_profiles: %d users | track_features: %d tracks",
        len(features["user_profiles"]),
        len(track_features),
    )

    # ── Step 2: Build test pairs ──────────────────────────────────────────
    log.info("Step 2 — Building test pairs …")
    test_pairs = test[["user_id", "track_id"]].copy().reset_index(drop=True)
    log.info(
        "  %d (user, track) pairs across %d users",
        len(test_pairs), test_pairs["user_id"].nunique(),
    )

    # ── Step 3: Cold / warm split ─────────────────────────────────────────
    log.info("Step 3 — Routing cold / warm users …")
    cold_pairs, warm_pairs = _split_cold_warm_users(test_pairs, train)
    log.info(
        "  Cold users: %d (%d pairs) | Warm users: %d (%d pairs)",
        cold_pairs["user_id"].nunique(), len(cold_pairs),
        warm_pairs["user_id"].nunique(), len(warm_pairs),
    )

    # ── Step 4: Pair features for warm users ──────────────────────────────
    log.info("Step 4 — Building pair features for warm users …")
    if not warm_pairs.empty:
        pair_features = _build_pair_features(warm_pairs, train, features, config)
        _print_missing_report(pair_features)
    else:
        pair_features = pd.DataFrame()
        log.info("  No warm pairs — skipping feature computation.")

    # ── Step 5: Score and write ───────────────────────────────────────────
    log.info("Step 5 — Scoring and writing submission(s) [rule=%s] …", rule)
    rule_based = ["rule1", "rule2", "rule3"]
    if rule == "all":
        rules_to_run = rule_based
    elif rule in ("lgbm", "lgbm_v2", "lgbm_v3", "lgbm_v4", "lgbm_v5", "lgbm_ensemble"):
        rules_to_run = []
    else:
        rules_to_run = [rule]

    for r in rules_to_run:
        log.info("  Applying %s …", r)
        _score_and_write(
            r, warm_pairs, cold_pairs, pair_features, features, config,
            submissions_dir,
        )

    # ── LGBM rules ────────────────────────────────────────────────────────────
    from src import ranker as ranker_mod

    if rule in ("lgbm", "all"):
        log.info("Step 5b — LightGBM LambdaRank (v1) …")
        ranker_mod.run(
            data=data,
            warm_pairs=warm_pairs,
            test_pair_features=pair_features,
            cold_pairs=cold_pairs,
            features=features,
            config=config,
            submissions_dir=submissions_dir,
        )

    if rule in ("lgbm_v2", "all"):
        log.info("Step 5c — LightGBM LambdaRank v2 (hard negatives + meta-features) …")
        ranker_mod.run_v2(
            data=data,
            warm_pairs=warm_pairs,
            test_pair_features=pair_features,
            cold_pairs=cold_pairs,
            features=features,
            config=config,
            submissions_dir=submissions_dir,
        )

    if rule in ("lgbm_v3", "all"):
        log.info("Step 5d — LightGBM Binary Classifier v3 (hard negatives + meta-features) …")
        ranker_mod.run_v3(
            data=data,
            warm_pairs=warm_pairs,
            test_pair_features=pair_features,
            cold_pairs=cold_pairs,
            features=features,
            config=config,
            submissions_dir=submissions_dir,
        )

    if rule in ("lgbm_v4", "all"):
        log.info("Step 5e — LightGBM LambdaRank v4 (extended features + normalisation) …")
        ranker_mod.run_v4(
            data=data,
            warm_pairs=warm_pairs,
            test_pair_features=pair_features,
            cold_pairs=cold_pairs,
            features=features,
            config=config,
            submissions_dir=submissions_dir,
        )

    if rule in ("lgbm_v5", "all"):
        log.info("Step 5f — LightGBM Binary Classifier v5 (extended features + normalisation) …")
        ranker_mod.run_v5(
            data=data,
            warm_pairs=warm_pairs,
            test_pair_features=pair_features,
            cold_pairs=cold_pairs,
            features=features,
            config=config,
            submissions_dir=submissions_dir,
        )

    if rule in ("lgbm_ensemble", "all"):
        log.info("Step 5g — Ensemble (rule2*0.4 + lgbm_v4*0.6) …")
        ranker_mod.run_ensemble(
            data=data,
            warm_pairs=warm_pairs,
            test_pair_features=pair_features,
            cold_pairs=cold_pairs,
            features=features,
            config=config,
            submissions_dir=submissions_dir,
        )

    # ── Step 6: 80/20 AUC estimate ────────────────────────────────────────
    if config.get("pipeline", {}).get("eval_auc", False):
        log.info("Step 6 - Estimating AUC via 80/20 split ...")
        _estimate_auc_80_20(data, config)

    elapsed = time.time() - t0
    log.info(
        "Done in %.1f s.  Submissions written to %s/",
        elapsed, submissions_dir,
    )


# ---------------------------------------------------------------------------
# Legacy stubs kept for import compatibility
# ---------------------------------------------------------------------------

def generate_candidates(
    user_id: int,
    train: pd.DataFrame,
    tracks: pd.DataFrame,
    config: dict[str, Any],
) -> list[int]:
    """Return candidate track IDs for a user (kept for backwards compatibility).

    In the current pipeline the test file specifies the candidate set explicitly,
    so this function is not called by ``run()``.  It is retained so that any
    existing callers do not get an AttributeError.
    """
    strategy = config.get("pipeline", {}).get("candidate_strategy", "unseen")
    if strategy == "unseen":
        seen = set(train.loc[train["user_id"] == user_id, "track_id"].unique())
        return tracks.loc[~tracks["track_id"].isin(seen), "track_id"].tolist()
    return tracks["track_id"].tolist()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Music recommender pipeline — produce ranked submission CSVs.",
    )
    parser.add_argument(
        "--config", default=str(CONFIG_PATH),
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--rule", default="all",
        choices=["all", "rule1", "rule2", "rule3", "lgbm", "lgbm_v2", "lgbm_v3",
                 "lgbm_v4", "lgbm_v5", "lgbm_ensemble"],
        help=(
            "Scoring rule to apply.  'all' writes one CSV per rule including "
            "lgbm/lgbm_v2/lgbm_v3 (default: all)"
        ),
    )
    parser.add_argument(
        "--run-name", default=None,
        help="Override pipeline.run_name in config (used for log messages).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    config = load_config(args.config)
    if args.run_name:
        config.setdefault("pipeline", {})["run_name"] = args.run_name
    run(config, rule=args.rule)


if __name__ == "__main__":
    main()
