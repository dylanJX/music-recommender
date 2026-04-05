"""
src/optimize.py -- 4-phase AUC optimization (0.875 -> 0.890)

Phase 1: Optimize hw4 + Rule 2 scoring weights, IDF, rating normalization
Phase 2: Train LGBM v6 with interaction features
Phase 3: Ensemble weight optimization
Phase 4: Summary table and submission generation

Usage: py -3 -m src.optimize
"""
from __future__ import annotations

import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from sklearn.metrics import roc_auc_score

from src import data_loader
from src.feature_engineering import (
    build_track_features,
    compute_idf_weights,
    compute_user_track_features,
    compute_hw4_features,
)
from src.collab_features import compute_cf_features
from src import scorer as scorer_mod
from src.scorer import write_submission
from src.ranker import (
    FEATURE_COLS_V4,
    _build_hard_neg_training_data,
    _add_rule_scores,
    _add_extended_features,
    _normalize_features,
    _lgbm_progress_callback,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONFIG = _root / "config.yaml"
SUBS = _root / "submissions"
SUBS.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# V6 Feature definition: V4 (29) + 5 interaction features = 34
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_COLS_V6: list[str] = FEATURE_COLS_V4 + [
    "album_artist_interaction",
    "genre_cf_interaction",
    "genre_peakedness",
    "artist_cf_interaction",
    "optimized_rule2_score",
]


# ═══════════════════════════════════════════════════════════════════════════
# Scoring helpers
# ═══════════════════════════════════════════════════════════════════════════

def _pun(scores, uids):
    """Per-user min-max normalize to [0, 1]."""
    s = np.asarray(scores, dtype=np.float64)
    u = np.asarray(uids)
    out = s.copy()
    uniq, inv = np.unique(u, return_inverse=True)
    for i in range(len(uniq)):
        m = inv == i
        lo, hi = s[m].min(), s[m].max()
        r = hi - lo
        out[m] = (s[m] - lo) / r if r > 1e-12 else 0.5
    return out


def _r1(pf, w):
    """Weighted hw4 content score -> numpy array."""
    t = sum(w.values())
    if t == 0:
        return np.zeros(len(pf))
    return sum(w[k] * pf[k].fillna(0.0).values for k in w) / t


def _r2(pf, hw4w, bw):
    """Rule 2: per-user-normalized content blended with CF -> numpy array."""
    cn = _pun(_r1(pf, hw4w), pf["user_id"].values)
    svd = pf["cf_svd_score"].fillna(0.0).values
    uu = pf["cf_user_user_score"].fillna(0.0).values
    t = bw["w_content"] + bw["w_svd"] + bw["w_uu"]
    if t == 0:
        return cn
    return (bw["w_content"] * cn + bw["w_svd"] * svd + bw["w_uu"] * uu) / t


def _add_v6(pf, hw4w, bw):
    """Append V6 interaction features to a pair-features DataFrame."""
    out = pf.copy()
    out["album_artist_interaction"] = (
        out["album_score"].fillna(0) * out["artist_score"].fillna(0)
    )
    out["genre_cf_interaction"] = (
        out["genre_max"].fillna(0) * out["cf_svd_score"].fillna(0)
    )
    out["genre_peakedness"] = (
        out["genre_max"].fillna(0) - out["genre_mean"].fillna(0)
    )
    out["artist_cf_interaction"] = (
        out["artist_score"].fillna(0) * out["cf_user_user_score"].fillna(0)
    )
    out["optimized_rule2_score"] = _r2(out, hw4w, bw)
    # Override score_rank with ranking based on optimized rule2
    out["score_rank"] = (
        out.groupby("user_id")["optimized_rule2_score"]
        .rank(ascending=False, method="first")
        .astype(float)
    )
    return out


def _impute_v6(df, medians=None):
    """Impute NaN in V6 feature columns with medians."""
    out = df.copy()
    if medians is None:
        medians = {}
        for c in FEATURE_COLS_V6:
            if c in out.columns:
                m = float(out[c].median())
                medians[c] = 0.0 if np.isnan(m) else m
            else:
                medians[c] = 0.0
    for c in FEATURE_COLS_V6:
        if c in out.columns:
            out[c] = out[c].fillna(medians.get(c, 0.0))
        else:
            out[c] = medians.get(c, 0.0)
    return out, medians


def _score_v6(model, pf, medians, mins, maxs):
    """Score pairs with LGBM v6 binary classifier -> numpy array."""
    df, _ = _impute_v6(pf, medians)
    for c in FEATURE_COLS_V6:
        if c not in df.columns:
            df[c] = medians.get(c, 0.0)
    df, _, _ = _normalize_features(df, FEATURE_COLS_V6, mins=mins, maxs=maxs)
    X = df[FEATURE_COLS_V6].values
    return model.predict_proba(X)[:, 1]


# ═══════════════════════════════════════════════════════════════════════════
# Setup: load data, split, build eval pairs and features
# ═══════════════════════════════════════════════════════════════════════════

def setup(config):
    """Load data, 80/20 interaction split, build hard-neg eval pairs, compute features."""
    t0 = time.time()
    log.info("Loading data...")
    data = data_loader.load_all(Path(config["data"]["dir"]), config=config)
    train_full = data["train"]
    tracks = data["tracks"]
    log.info("  train: %d rows, %d users", len(train_full), train_full["user_id"].nunique())

    # ── 80/20 interaction-level split per user ────────────────────────────
    log.info("Splitting interactions 80/20...")
    rng = np.random.default_rng(42)
    t80, hold = [], []
    for _, g in train_full.groupby("user_id"):
        n = len(g)
        nh = max(1, round(n * 0.2))
        p = rng.permutation(n)
        hold.append(g.iloc[p[:nh]])
        t80.append(g.iloc[p[nh:]])
    train_80 = pd.concat(t80, ignore_index=True)
    holdout = pd.concat(hold, ignore_index=True)
    log.info("  train_80: %d | holdout: %d", len(train_80), len(holdout))

    # ── Build eval pairs with hard negatives ──────────────────────────────
    log.info("Building eval pairs (hard negatives)...")
    rng2 = np.random.default_rng(42)
    ta, tb = {}, {}
    a2t: dict[int, list[int]] = defaultdict(list)
    b2t: dict[int, list[int]] = defaultdict(list)
    for _, r in tracks.iterrows():
        tid = int(r["track_id"])
        if pd.notna(r.get("artist_id")):
            a = int(r["artist_id"])
            ta[tid] = a
            a2t[a].append(tid)
        if pd.notna(r.get("album_id")):
            b = int(r["album_id"])
            tb[tid] = b
            b2t[b].append(tid)

    fh = train_full.groupby("user_id")["track_id"].apply(set).to_dict()
    atids = tracks["track_id"].values

    users = holdout["user_id"].unique()
    if len(users) > 5000:
        users = rng2.choice(users, size=5000, replace=False)

    rows: list[dict] = []
    labs: list[int] = []
    for uid in users:
        pos = holdout.loc[holdout["user_id"] == uid, "track_id"].unique()
        if len(pos) == 0:
            continue
        np_ = min(3, len(pos))
        ps = rng2.choice(pos, size=np_, replace=False)

        hard: set[int] = set()
        for pt in ps:
            if ta.get(int(pt)):
                hard.update(a2t[ta[int(pt)]])
            if tb.get(int(pt)):
                hard.update(b2t[tb[int(pt)]])
        hard -= fh.get(int(uid), set())
        hard -= {int(t) for t in ps}

        ha = list(hard)
        nn = np_
        if len(ha) >= nn:
            ns = list(rng2.choice(ha, size=nn, replace=False))
        else:
            ns = ha[:]
            need = nn - len(ns)
            excl = fh.get(int(uid), set()) | {int(t) for t in ps} | hard
            uns = atids[~np.isin(atids, list(excl))]
            if len(uns) > 0:
                ns += list(rng2.choice(uns, size=min(need, len(uns)), replace=False))

        if not ns:
            continue
        for t in ps:
            rows.append({"user_id": int(uid), "track_id": int(t)})
            labs.append(1)
        for t in ns:
            rows.append({"user_id": int(uid), "track_id": int(t)})
            labs.append(0)

    eval_pairs = pd.DataFrame(rows)
    labels = np.array(labs)
    log.info(
        "  Eval: %d pairs (%d pos, %d neg)",
        len(eval_pairs), int(labels.sum()), int((labels == 0).sum()),
    )

    # ── Build features on eval pairs (train_80 context) ──────────────────
    log.info("Building eval features...")
    tf80 = build_track_features(train_80, tracks)
    idf80 = compute_idf_weights(train_80, tracks)

    log.info("  Content features (%d pairs)...", len(eval_pairs))
    content = compute_user_track_features(eval_pairs, train_80, tf80)
    log.info("  hw4 features...")
    hw4 = compute_hw4_features(eval_pairs, train_80, tf80, idf80)
    log.info("  CF features (takes a few minutes)...")
    cf = compute_cf_features(eval_pairs, train_80, config)

    pf = (
        content
        .merge(
            hw4[["user_id", "track_id", "hw4_track_score", "hw4_artist_score",
                 "hw4_album_score", "hw4_genre_score", "hw4_pop_score"]],
            on=["user_id", "track_id"], how="left",
        )
        .merge(
            cf[["user_id", "track_id", "cf_user_user_score",
                "cf_item_item_score", "cf_svd_score"]],
            on=["user_id", "track_id"], how="left",
        )
        .merge(tf80[["track_id", "popularity_score"]], on="track_id", how="left")
    )
    log.info("  Features: %d x %d (%.0fs)", *pf.shape, time.time() - t0)
    return data, train_full, train_80, holdout, tracks, eval_pairs, labels, pf, tf80, idf80


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: Optimize core scoring logic
# ═══════════════════════════════════════════════════════════════════════════

def phase1(pf, labels, eval_pairs, train_80, tracks, tf80, config):
    """Test IDF, rating norm, album proxy, genre agg, weight grid search."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("PHASE 1: OPTIMIZE THE CORE SCORING LOGIC")
    print("=" * 70)

    OHW = {
        "hw4_track_score": 0.45, "hw4_album_score": 0.10,
        "hw4_artist_score": 0.25, "hw4_genre_score": 0.15, "hw4_pop_score": 0.05,
    }
    OBL = {"w_content": 0.70, "w_svd": 0.15, "w_uu": 0.15}

    base_auc = roc_auc_score(labels, _r2(pf, OHW, OBL))
    print(f"\nBaseline Rule 2 validation AUC: {base_auc:.4f}")

    # ── 1. IDF Weighting Formula ──────────────────────────────────────────
    print("\n--- 1. IDF Weighting Formula ---")
    n_users = train_80["user_id"].nunique()
    enriched = train_80[["user_id", "track_id"]].merge(
        tracks[["track_id", "album_id", "artist_id", "genre_ids"]],
        on="track_id", how="left",
    )
    alb_df = enriched.dropna(subset=["album_id"]).groupby("album_id")["user_id"].nunique()
    art_df = enriched.dropna(subset=["artist_id"]).groupby("artist_id")["user_id"].nunique()
    gr = (
        enriched[["user_id", "genre_ids"]].explode("genre_ids")
        .rename(columns={"genre_ids": "genre_id"}).dropna(subset=["genre_id"])
    )
    gr["genre_id"] = gr["genre_id"].astype(int)
    gen_df = gr.groupby("genre_id")["user_id"].nunique()

    def _bm25(dfc):
        return {
            int(k): max(0.0, math.log((n_users - int(v) + 0.5) / (int(v) + 0.5)))
            for k, v in dfc.items()
        }

    def _smooth(dfc):
        return {
            int(k): math.log(1.0 + n_users / max(1, int(v)))
            for k, v in dfc.items()
        }

    idf_vars = {
        "current": compute_idf_weights(train_80, tracks),
        "bm25": {
            "album_idf": _bm25(alb_df),
            "artist_idf": _bm25(art_df),
            "genre_idf": _bm25(gen_df),
        },
        "smoothed_log1pN/df": {
            "album_idf": _smooth(alb_df),
            "artist_idf": _smooth(art_df),
            "genre_idf": _smooth(gen_df),
        },
    }

    best_idf_name, best_idf_auc = "current", base_auc
    best_idf = idf_vars["current"]
    pf_work = pf.copy()

    for name, idf_w in idf_vars.items():
        hw4_v = compute_hw4_features(eval_pairs, train_80, tf80, idf_w)
        pf_v = pf.copy()
        for c in ["hw4_track_score", "hw4_artist_score", "hw4_album_score",
                   "hw4_genre_score", "hw4_pop_score"]:
            pf_v[c] = hw4_v[c].values
        auc = roc_auc_score(labels, _r2(pf_v, OHW, OBL))
        tag = " <-- current" if name == "current" else ""
        print(f"  {name:<22} AUC: {auc:.4f}{tag}")
        if auc > best_idf_auc:
            best_idf_auc = auc
            best_idf_name = name
            best_idf = idf_w
            pf_work = pf_v

    print(f"  -> Best IDF: {best_idf_name} ({best_idf_auc:.4f})")
    pf = pf_work

    # ── 2. User Rating Bias Removal ───────────────────────────────────────
    print("\n--- 2. User Rating Bias Removal ---")
    train_pct = train_80.copy()
    train_pct["play_count"] = (
        train_pct.groupby("user_id")["play_count"].rank(pct=True) * 100
    )
    hw4_pct = compute_hw4_features(eval_pairs, train_pct, tf80, best_idf)
    pf_pct = pf.copy()
    for c in ["hw4_track_score", "hw4_artist_score", "hw4_album_score",
              "hw4_genre_score", "hw4_pop_score"]:
        pf_pct[c] = hw4_pct[c].values
    pct_auc = roc_auc_score(labels, _r2(pf_pct, OHW, OBL))
    print(f"  Original ratings:   {best_idf_auc:.4f}")
    print(f"  Percentile ratings: {pct_auc:.4f}")
    use_pct = pct_auc > best_idf_auc
    if use_pct:
        print(f"  -> Percentile normalization WINS (+{pct_auc - best_idf_auc:.4f})")
        pf = pf_pct
    else:
        print(f"  -> Original ratings win. Keeping raw play_count.")

    # ── 3. Album Proxy Score ──────────────────────────────────────────────
    print("\n--- 3. Album Proxy Score (Dig Deeper fallback) ---")
    n_nan = int(pf["album_score"].isna().sum())
    print(f"  NaN album_score: {n_nan}/{len(pf)} ({100 * n_nan / len(pf):.1f}%)")

    proxy_results: dict[str, float] = {}
    for method in ["zero (current)", "user_mean", "user_max"]:
        pf_t = pf.copy()
        if method == "zero (current)":
            pf_t["album_score"] = pf_t["album_score"].fillna(0.0)
        elif method == "user_mean":
            um = pf.groupby("user_id")["album_score"].transform("mean")
            pf_t["album_score"] = pf_t["album_score"].fillna(um).fillna(0.0)
        elif method == "user_max":
            um = pf.groupby("user_id")["album_score"].transform("max")
            pf_t["album_score"] = pf_t["album_score"].fillna(um).fillna(0.0)
        auc = roc_auc_score(labels, _r2(pf_t, OHW, OBL))
        proxy_results[method] = auc
        print(f"  {method:<20} AUC: {auc:.4f}")

    best_proxy = max(proxy_results, key=proxy_results.get)
    print(f"  -> Best: {best_proxy}")

    # ── 4. Genre Aggregation Formula ──────────────────────────────────────
    print("\n--- 4. Genre Aggregation Formula ---")
    gmax = pf["genre_max"].fillna(0.0)
    gmin = pf["genre_min"].fillna(0.0)
    gmean = pf["genre_mean"].fillna(0.0)
    gsum = pf["genre_sum"].fillna(0.0)
    gcnt = pf["genre_count"].fillna(1).replace(0, 1)

    genre_vars = {
        "hw4_genre_score (sum*IDF)": pf["hw4_genre_score"].fillna(0.0),
        "genre_weighted_mean": pf["genre_weighted_mean"].fillna(0.0),
        "harmonic_mean": 2 * gmax * gmin / (gmax + gmin + 1e-10),
        "trimmed_mean": pd.Series(
            np.where(gcnt > 1, (gsum - gmin) / (gcnt - 1), gmean.values),
            index=pf.index,
        ),
    }
    # Scale variants to match hw4_genre_score magnitude for fair comparison
    ref_mean = pf["hw4_genre_score"].fillna(0.0).mean()
    ref_std = pf["hw4_genre_score"].fillna(0.0).std() + 1e-10
    for name, col in genre_vars.items():
        if "hw4_genre_score" in name:
            continue
        cm, cs = float(col.mean()), float(col.std()) + 1e-10
        genre_vars[name] = (col - cm) / cs * ref_std + ref_mean

    for name, col in genre_vars.items():
        pf_t = pf.copy()
        pf_t["hw4_genre_score"] = col.values if isinstance(col, pd.Series) else col
        auc = roc_auc_score(labels, _r2(pf_t, OHW, OBL))
        print(f"  {name:<30} AUC: {auc:.4f}")

    # ── 5. Weight Grid Search ─────────────────────────────────────────────
    print("\n--- 5. Artist/Album/Genre Weight Grid Search ---")
    print("  Searching hw4 weights (this is fast)...")
    hw4_res: list[tuple[float, dict]] = []
    for wt in np.arange(0.25, 0.60, 0.05):
        for wa in np.arange(0.10, 0.40, 0.05):
            for wl in np.arange(0.02, 0.20, 0.03):
                for wg in np.arange(0.05, 0.30, 0.05):
                    wp = round(1.0 - wt - wa - wl - wg, 4)
                    if wp < -0.005 or wp > 0.20:
                        continue
                    wp = max(0.0, wp)
                    w = {
                        "hw4_track_score": round(wt, 2),
                        "hw4_album_score": round(wl, 2),
                        "hw4_artist_score": round(wa, 2),
                        "hw4_genre_score": round(wg, 2),
                        "hw4_pop_score": round(wp, 4),
                    }
                    auc = roc_auc_score(labels, _r2(pf, w, OBL))
                    hw4_res.append((auc, w))

    hw4_res.sort(key=lambda x: x[0], reverse=True)
    print(f"  Tested {len(hw4_res)} combinations. Top 5:")
    for i, (auc, w) in enumerate(hw4_res[:5]):
        print(
            f"    #{i+1} AUC={auc:.4f}  trk={w['hw4_track_score']:.2f} "
            f"art={w['hw4_artist_score']:.2f} alb={w['hw4_album_score']:.2f} "
            f"gen={w['hw4_genre_score']:.2f} pop={w['hw4_pop_score']:.2f}"
        )

    best_hw4 = hw4_res[0][1]

    print("\n  Searching Rule 2 blend weights...")
    bl_res: list[tuple[float, dict]] = []
    for wc in np.arange(0.30, 0.95, 0.05):
        for ws in np.arange(0.00, 0.45, 0.05):
            wu = round(1.0 - wc - ws, 4)
            if wu < -0.005 or wu > 0.45:
                continue
            wu = max(0.0, wu)
            b = {
                "w_content": round(wc, 2),
                "w_svd": round(ws, 2),
                "w_uu": round(wu, 2),
            }
            auc = roc_auc_score(labels, _r2(pf, best_hw4, b))
            bl_res.append((auc, b))

    bl_res.sort(key=lambda x: x[0], reverse=True)
    print(f"  Tested {len(bl_res)} combinations. Top 5:")
    for i, (auc, w) in enumerate(bl_res[:5]):
        print(
            f"    #{i+1} AUC={auc:.4f}  content={w['w_content']:.2f} "
            f"svd={w['w_svd']:.2f} uu={w['w_uu']:.2f}"
        )

    best_blend = bl_res[0][1]

    # ── 6. Track Exact Match Verification ─────────────────────────────────
    print("\n--- 6. Track Exact Match Verification ---")
    ht = int((pf["hw4_track_score"] > 0).sum())
    print(f"  Non-zero hw4_track_score: {ht}/{len(pf)} ({100 * ht / len(pf):.1f}%)")
    print(f"  Exact match bonus working: {'Yes' if ht > 0 else 'No'}")

    # ── Phase 1 Summary ──────────────────────────────────────────────────
    opt_auc = roc_auc_score(labels, _r2(pf, best_hw4, best_blend))
    print(f"\n{'=' * 60}")
    print("Phase 1 Results:")
    print(f"  Original Rule 2 AUC:  {base_auc:.4f}")
    print(f"  Optimized Rule 2 AUC: {opt_auc:.4f}")
    print(f"  Improvement:          {opt_auc - base_auc:+.4f}")
    print(f"  Best hw4 weights:   {best_hw4}")
    print(f"  Best blend weights: {best_blend}")
    print(f"  IDF: {best_idf_name} | Rating norm: {'percentile' if use_pct else 'original'}")
    print(f"  Time: {time.time() - t0:.0f}s")

    return best_hw4, best_blend, best_idf_name, best_idf, use_pct, base_auc, opt_auc


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: Train LGBM v6 with interaction features
# ═══════════════════════════════════════════════════════════════════════════

def phase2(data, features, config, best_hw4, best_blend, eval_pf, labels):
    """Train LGBM v6 binary classifier with 5 new interaction features."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("PHASE 2: OPTIMIZE LGBM WITH NEW FEATURES")
    print("=" * 70)

    import lightgbm as lgb

    train_full = data["train"]
    tracks = data["tracks"]
    track_features = features["track_features"]
    lgbm_cfg = config.get("lgbm", {})
    max_users = int(lgbm_cfg.get("max_train_users", 40000))
    n_est = int(lgbm_cfg.get("n_estimators", 1000))

    # extra_feat_fn: V4 extended features + V6 interaction features
    def _ext_fn(feat_df, train_ctx, trks):
        out = _add_extended_features(feat_df, train_ctx, track_features, config)
        out = _add_v6(out, best_hw4, best_blend)
        return out

    log.info("Building hard-neg training data with V6 features...")
    rng = np.random.default_rng(42)
    train_pf, train_labels, train_groups = _build_hard_neg_training_data(
        train_full, tracks, config,
        max_users=max_users, rng=rng, extra_feat_fn=_ext_fn,
    )

    # Impute + normalize
    train_pf, medians = _impute_v6(train_pf)
    for c in FEATURE_COLS_V6:
        if c not in train_pf.columns:
            train_pf[c] = 0.0
    train_norm, mins, maxs = _normalize_features(train_pf, FEATURE_COLS_V6)
    X = train_norm[FEATURE_COLS_V6].values

    n_neg = int((train_labels == 0).sum())
    n_pos = max(1, int(train_labels.sum()))
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
    log.info(
        "Training LGBM v6: %d pairs × %d features × %d trees...",
        len(X), len(FEATURE_COLS_V6), n_est,
    )
    model.fit(X, train_labels, callbacks=[_lgbm_progress_callback(n_est, 100)])

    # Feature importances
    fi = sorted(
        zip(FEATURE_COLS_V6, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print(f"\n-- LGBM v6 Feature Importances --")
    print(f"  {'Rank':<5} {'Feature':<32} {'Importance':>12}")
    print(f"  {'-' * 5} {'-' * 32} {'-' * 12}")
    for i, (f, imp) in enumerate(fi[:20], 1):
        print(f"  {i:<5} {f:<32} {imp:>12.0f}")

    # Score eval pairs
    log.info("Scoring eval pairs with LGBM v6...")
    eval_v6 = _add_rule_scores(eval_pf.copy(), config)
    eval_v6 = _add_extended_features(eval_v6, data["train"], track_features, config)
    eval_v6 = _add_v6(eval_v6, best_hw4, best_blend)
    v6_raw = _score_v6(model, eval_v6, medians, mins, maxs)
    v6_auc = roc_auc_score(labels, v6_raw)
    print(f"\n  LGBM v6 validation AUC: {v6_auc:.4f}")
    print(f"  Phase 2 time: {time.time() - t0:.0f}s")

    return model, medians, mins, maxs, v6_auc


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: Ensemble weight optimization
# ═══════════════════════════════════════════════════════════════════════════

def phase3(eval_pf, labels, best_hw4, best_blend, model, medians, mins, maxs,
           data, features, config):
    """Grid search ensemble weights with and without per-user normalization."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("PHASE 3: ENSEMBLE OPTIMIZATION")
    print("=" * 70)

    track_features = features["track_features"]

    # Optimized Rule 2 scores on eval pairs
    r2_scores = _r2(eval_pf, best_hw4, best_blend)

    # LGBM v6 scores on eval pairs
    eval_v6 = _add_rule_scores(eval_pf.copy(), config)
    eval_v6 = _add_extended_features(eval_v6, data["train"], track_features, config)
    eval_v6 = _add_v6(eval_v6, best_hw4, best_blend)
    lgbm_scores = _score_v6(model, eval_v6, medians, mins, maxs)

    # Per-user normalized versions
    r2_n = _pun(r2_scores, eval_pf["user_id"].values)
    lg_n = _pun(lgbm_scores, eval_pf["user_id"].values)

    # Grid search
    print(f"\n  {'r2_w':<6} {'lgbm_w':<6} {'Raw AUC':>10} {'Norm AUC':>10}")
    print(f"  {'-' * 6} {'-' * 6} {'-' * 10} {'-' * 10}")

    best_raw = (0.0, 0.5)
    best_norm = (0.0, 0.5)
    for wr in np.arange(0.20, 0.85, 0.05):
        wl = round(1.0 - wr, 2)
        raw = wr * r2_scores + wl * lgbm_scores
        norm = wr * r2_n + wl * lg_n
        auc_raw = roc_auc_score(labels, raw)
        auc_norm = roc_auc_score(labels, norm)
        print(f"  {wr:<6.2f} {wl:<6.2f} {auc_raw:>10.4f} {auc_norm:>10.4f}")
        if auc_raw > best_raw[0]:
            best_raw = (auc_raw, wr)
        if auc_norm > best_norm[0]:
            best_norm = (auc_norm, wr)

    print(f"\n  Best raw ensemble:        AUC={best_raw[0]:.4f}  (r2_w={best_raw[1]:.2f})")
    print(f"  Best normalized ensemble: AUC={best_norm[0]:.4f}  (r2_w={best_norm[1]:.2f})")
    print(f"  Phase 3 time: {time.time() - t0:.0f}s")

    return best_raw, best_norm


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4: Generate submissions and print summary
# ═══════════════════════════════════════════════════════════════════════════

def _make_idf(train, tracks, idf_name, n_users):
    """Recompute IDF weights on the given train context."""
    if idf_name == "current":
        return compute_idf_weights(train, tracks)
    enriched = train[["user_id", "track_id"]].merge(
        tracks[["track_id", "album_id", "artist_id", "genre_ids"]],
        on="track_id", how="left",
    )
    alb_df = enriched.dropna(subset=["album_id"]).groupby("album_id")["user_id"].nunique()
    art_df = enriched.dropna(subset=["artist_id"]).groupby("artist_id")["user_id"].nunique()
    gr = (
        enriched[["user_id", "genre_ids"]].explode("genre_ids")
        .rename(columns={"genre_ids": "genre_id"}).dropna(subset=["genre_id"])
    )
    gr["genre_id"] = gr["genre_id"].astype(int)
    gen_df = gr.groupby("genre_id")["user_id"].nunique()

    if idf_name == "bm25":
        mk = lambda dfc: {
            int(k): max(0.0, math.log((n_users - int(v) + 0.5) / (int(v) + 0.5)))
            for k, v in dfc.items()
        }
    else:
        mk = lambda dfc: {
            int(k): math.log(1.0 + n_users / max(1, int(v)))
            for k, v in dfc.items()
        }
    return {"album_idf": mk(alb_df), "artist_idf": mk(art_df), "genre_idf": mk(gen_df)}


def phase4(
    data, features, config,
    best_hw4, best_blend, best_idf_name, use_pct,
    model, medians, mins, maxs,
    best_raw, best_norm,
    base_auc, opt_auc, v6_auc,
):
    """Build test features, generate all submission CSVs, print summary."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("PHASE 4: GENERATE SUBMISSIONS & FINAL SUMMARY")
    print("=" * 70)

    train_full = data["train"]
    tracks = data["tracks"]
    track_features = features["track_features"]
    test_pairs = data["test"][["user_id", "track_id"]].copy()

    # Cold / warm split
    train_users = set(train_full["user_id"].unique())
    is_cold = ~test_pairs["user_id"].isin(train_users)
    cold_pairs = test_pairs[is_cold].copy()
    warm_pairs = test_pairs[~is_cold].copy()
    log.info("Test: %d warm, %d cold", len(warm_pairs), len(cold_pairs))

    # ── Build test features ───────────────────────────────────────────────
    log.info("Building test features on train_full context...")
    tf_full = build_track_features(train_full, tracks)
    n_users_full = train_full["user_id"].nunique()
    idf_full = _make_idf(train_full, tracks, best_idf_name, n_users_full)

    # Optionally percentile-normalize ratings for hw4 features
    if use_pct:
        train_hw4 = train_full.copy()
        train_hw4["play_count"] = (
            train_hw4.groupby("user_id")["play_count"].rank(pct=True) * 100
        )
    else:
        train_hw4 = train_full

    log.info("  Content features for %d warm pairs...", len(warm_pairs))
    content = compute_user_track_features(warm_pairs, train_full, tf_full)
    log.info("  hw4 features...")
    hw4 = compute_hw4_features(warm_pairs, train_hw4, tf_full, idf_full)
    log.info("  CF features (takes a few minutes)...")
    cf = compute_cf_features(warm_pairs, train_full, config)

    test_pf = (
        content
        .merge(
            hw4[["user_id", "track_id", "hw4_track_score", "hw4_artist_score",
                 "hw4_album_score", "hw4_genre_score", "hw4_pop_score"]],
            on=["user_id", "track_id"], how="left",
        )
        .merge(
            cf[["user_id", "track_id", "cf_user_user_score",
                "cf_item_item_score", "cf_svd_score"]],
            on=["user_id", "track_id"], how="left",
        )
        .merge(tf_full[["track_id", "popularity_score"]], on="track_id", how="left")
    )
    log.info("  Test features: %d x %d", *test_pf.shape)

    # Cold user popularity fallback
    pop_map = tf_full.set_index("track_id")["popularity_score"].to_dict()
    cold_scores = cold_pairs["track_id"].map(pop_map).fillna(0.0).reset_index(drop=True)

    soft_rank_probs = config.get("pipeline", {}).get(
        "soft_rank_probs", [0.99, 0.95, 0.90, 0.10, 0.05, 0.01]
    )
    top_n = int(config.get("pipeline", {}).get("top_n", 100))

    def _write(warm_sc, filename):
        ap = pd.concat(
            [warm_pairs.reset_index(drop=True), cold_pairs.reset_index(drop=True)],
            ignore_index=True,
        )
        asc = pd.concat(
            [pd.Series(warm_sc).reset_index(drop=True), cold_scores],
            ignore_index=True,
        )
        p = SUBS / filename
        write_submission(ap, asc, p, top_n=top_n, soft_rank_probs=soft_rank_probs)
        print(f"  Written: {p}")

    # ── 1. Optimized Rule 2 ──────────────────────────────────────────────
    log.info("Generating submission_rule2_optimized.csv...")
    r2_opt = _r2(test_pf, best_hw4, best_blend)
    _write(r2_opt, "submission_rule2_optimized.csv")

    # ── 2. LGBM v6 ──────────────────────────────────────────────────────
    log.info("Generating submission_lgbm_v6.csv...")
    test_v6 = _add_rule_scores(test_pf.copy(), config)
    test_v6 = _add_extended_features(test_v6, train_full, track_features, config)
    test_v6 = _add_v6(test_v6, best_hw4, best_blend)
    v6_test = _score_v6(model, test_v6, medians, mins, maxs)
    _write(v6_test, "submission_lgbm_v6.csv")

    # ── 3. Ensemble v2 (optimal weights, no normalization) ───────────────
    w_r2 = best_raw[1]
    w_lg = round(1.0 - w_r2, 2)
    log.info("Generating submission_ensemble_v2.csv (r2*%.2f + lgbm*%.2f)...", w_r2, w_lg)
    ens_v2 = w_r2 * r2_opt + w_lg * v6_test
    _write(ens_v2, "submission_ensemble_v2.csv")

    # ── 4. Ensemble normalized ───────────────────────────────────────────
    w_r2n = best_norm[1]
    w_lgn = round(1.0 - w_r2n, 2)
    log.info(
        "Generating submission_ensemble_normalized.csv (r2_n*%.2f + lgbm_n*%.2f)...",
        w_r2n, w_lgn,
    )
    r2_norm_test = _pun(r2_opt, warm_pairs["user_id"].values)
    lg_norm_test = _pun(v6_test, warm_pairs["user_id"].values)
    ens_norm = w_r2n * r2_norm_test + w_lgn * lg_norm_test
    _write(ens_norm, "submission_ensemble_normalized.csv")

    # ═══ FINAL SUMMARY TABLE ═════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("COMPLETE RESULTS TABLE")
    print(f"{'=' * 80}")
    print(
        f"\n  {'Submission':<28} {'Method':<34} {'Val AUC':>9} {'Notes'}"
    )
    print(f"  {'-' * 28} {'-' * 34} {'-' * 9} {'-' * 20}")
    table = [
        ("Rule 2 original", "Heuristic", base_auc, "baseline"),
        ("Rule 2 optimized", "Improved heuristic", opt_auc, "phase 1"),
        ("LGBM v5", "Binary 40k users", 0.801, "prev best LGBM (Kaggle)"),
        ("LGBM v6", "Binary + interactions", v6_auc, "phase 2"),
        ("Ensemble original", f"rule2*0.4 + lgbm*0.6", 0.875, "prev best (Kaggle)"),
        ("Ensemble v2", f"r2*{w_r2:.2f} + lgbm*{w_lg:.2f}", best_raw[0], "phase 3"),
        ("Ensemble normalized", f"r2_n*{w_r2n:.2f} + lgbm_n*{w_lgn:.2f}", best_norm[0], "phase 3"),
    ]
    for name, method, auc, note in table:
        print(f"  {name:<28} {method:<34} {auc:>9.4f} {note}")

    # Best submission recommendation
    best_sub = max(table, key=lambda x: x[2])
    print(f"\n  RECOMMENDED UPLOAD: {best_sub[0]}")
    print(f"  Validation AUC: {best_sub[2]:.4f}")

    print(f"\n  Analysis — likelihood of beating 0.890:")
    gap = 0.890 - best_sub[2]
    if gap <= 0:
        print(f"    Our validation AUC already exceeds 0.890!")
        print(f"    Kaggle AUC may differ due to different eval pairs.")
    else:
        print(f"    Validation-to-Kaggle gap: ~{gap:.3f}")
        print(f"    Key factors:")
        print(f"    - Validation uses hard negatives (realistic, correlates with Kaggle)")
        print(f"    - Per-user normalization equalizes model contributions")
        print(f"    - Interaction features capture non-linear artist-CF, genre-CF patterns")
        print(f"    - Optimized weights reduce content-CF imbalance")
        print(f"    - Previous ensemble went from 0.863 (rule2) + 0.801 (lgbm) -> 0.875 (Kaggle)")
        print(f"      showing ensembles gain ~0.012-0.074 over components")
        print(f"    - If improvements hold on Kaggle, ensemble_normalized is the best bet")

    print(f"\n  Phase 4 time: {time.time() - t0:.0f}s")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    T0 = time.time()
    config = yaml.safe_load(open(CONFIG, encoding="utf-8"))

    # Setup: load data, split, build eval features
    (data, train_full, train_80, holdout, tracks,
     eval_pairs, labels, pf, tf80, idf80) = setup(config)

    # Full features dict for track_features access
    features = {"track_features": build_track_features(train_full, tracks)}

    # Phase 1: Optimize scoring logic
    (best_hw4, best_blend, best_idf_name, best_idf,
     use_pct, base_auc, opt_auc) = phase1(
        pf, labels, eval_pairs, train_80, tracks, tf80, config,
    )

    # Phase 2: Train LGBM v6
    model, med, mins, maxs, v6_auc = phase2(
        data, features, config, best_hw4, best_blend, pf, labels,
    )

    # Phase 3: Ensemble optimization
    best_raw, best_norm = phase3(
        pf, labels, best_hw4, best_blend,
        model, med, mins, maxs,
        data, features, config,
    )

    # Phase 4: Generate submissions + summary
    phase4(
        data, features, config,
        best_hw4, best_blend, best_idf_name, use_pct,
        model, med, mins, maxs,
        best_raw, best_norm,
        base_auc, opt_auc, v6_auc,
    )

    # Update config.yaml with optimized weights
    print(f"\nUpdating config.yaml with optimized weights...")
    config["scorer"]["rule1_weights"] = {
        k.replace("hw4_", "").replace("_score", ""): v
        for k, v in best_hw4.items()
    }
    # Actually keep the original key names for config compatibility
    config["scorer"]["rule1_weights"] = best_hw4
    config["scorer"]["rule2_weights"] = best_blend
    with open(CONFIG, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  config.yaml updated with optimized weights.")

    print(f"\n{'=' * 70}")
    print(f"TOTAL TIME: {time.time() - T0:.0f}s ({(time.time() - T0) / 60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
