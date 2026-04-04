"""
collab_features.py
==================
Collaborative-filtering feature signals that augment the rule-based feature
vector produced by feature_engineering.py.

Three complementary CF signals are computed for each (user_id, track_id) pair:

  cf_user_user_score
      Weighted average interaction rate among the top-K most cosine-similar
      users. Captures "users like you listened to this track."

  cf_item_item_score
      Mean cosine similarity between the candidate track and the tracks the
      target user has already interacted with. Captures "this track is similar
      to what you have listened to."

  cf_svd_score
      Dot product of the user's and track's latent vectors produced by
      TruncatedSVD on the interaction matrix, normalised to [0, 1].

Cold-start handling
-------------------
Users or tracks absent from the training set receive 0.0 for every CF
signal — never NaN — so the values can be used directly as numeric features.

Memory-efficient UU computation
--------------------------------
Rather than computing the full (n_all_users × n_all_users) cosine similarity
matrix (which can exceed 19 GiB for large datasets), this module computes UU
similarities only for the unique *query* users present in ``user_track_pairs``,
processing them in blocks of ``UU_BLOCK`` rows at a time to keep peak memory
bounded to approximately ``UU_BLOCK × n_all_users × 8`` bytes.

Configuration keys (read from ``config['collab_filtering']``)
-------------------------------------------------------------
  cf_top_k      int, default 20  — neighbourhood size for user-user CF.
  n_components  int, default 20  — number of SVD latent factors.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# Rows of query users processed per batch when computing UU similarity.
# Keeps peak memory at roughly UU_BLOCK × n_users × 8 bytes.
UU_BLOCK = 200


# ---------------------------------------------------------------------------
# Interaction matrix
# ---------------------------------------------------------------------------

def build_interaction_matrix(
    train: pd.DataFrame,
) -> tuple[csr_matrix, dict[int, int], dict[int, int]]:
    """Build a sparse user-item interaction count matrix from training data.

    Parameters
    ----------
    train : pd.DataFrame
        Training interactions with at least columns ['user_id', 'track_id'].
        Duplicate (user, track) rows are summed into a single count.

    Returns
    -------
    matrix : scipy.sparse.csr_matrix
        Shape (n_users, n_tracks). Entry (i, j) is the number of times user i
        interacted with track j.
    user_to_idx : dict[int, int]
        Mapping from user_id to row index in the matrix.
    track_to_idx : dict[int, int]
        Mapping from track_id to column index in the matrix.
    """
    if train.empty:
        return csr_matrix((0, 0)), {}, {}

    users = sorted(train["user_id"].unique())
    tracks = sorted(train["track_id"].unique())
    user_to_idx: dict[int, int] = {uid: i for i, uid in enumerate(users)}
    track_to_idx: dict[int, int] = {tid: j for j, tid in enumerate(tracks)}

    counts = (
        train.groupby(["user_id", "track_id"]).size().reset_index(name="count")
    )
    rows = counts["user_id"].map(user_to_idx).values
    cols = counts["track_id"].map(track_to_idx).values
    data = counts["count"].values.astype(float)

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(users), len(tracks)),
        dtype=float,
    )
    return matrix, user_to_idx, track_to_idx


# ---------------------------------------------------------------------------
# SVD helpers
# ---------------------------------------------------------------------------

def _fit_svd(
    matrix: csr_matrix,
    n_components: int,
) -> tuple[np.ndarray | None, np.ndarray | None, float, float]:
    """Fit TruncatedSVD on the interaction matrix.

    Parameters
    ----------
    matrix : csr_matrix
        User-item interaction matrix of shape (n_users, n_tracks).
    n_components : int
        Number of latent factors. Clamped so it never exceeds
        min(n_users - 1, n_tracks - 1).

    Returns
    -------
    user_vecs : ndarray of shape (n_users, n_comp), or None when SVD cannot
        be fitted (matrix too small).
    track_vecs : ndarray of shape (n_tracks, n_comp), or None.
    score_min : float — minimum raw dot-product score across all (user, track) pairs.
    score_max : float — maximum raw dot-product score.
    """
    n_users, n_tracks = matrix.shape
    n_comp = min(n_components, n_users - 1, n_tracks - 1)
    if n_comp < 1:
        return None, None, 0.0, 1.0

    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    user_vecs = svd.fit_transform(matrix)        # (n_users, n_comp)
    track_vecs = svd.components_.T               # (n_tracks, n_comp)

    # Estimate global score range by sampling random (user, track) pairs.
    # This avoids materialising any large dense matrix while still giving a
    # tight enough range for normalisation to [0, 1].
    rng = np.random.default_rng(42)
    sample_n = min(50_000, n_users * n_tracks)
    s_u = rng.integers(0, n_users, size=sample_n)
    s_t = rng.integers(0, n_tracks, size=sample_n)
    sample_raw = np.sum(user_vecs[s_u] * track_vecs[s_t], axis=1)
    s_min = float(sample_raw.min())
    s_max = float(sample_raw.max())
    if s_min >= s_max:
        s_min, s_max = 0.0, 1.0
    return user_vecs, track_vecs, s_min, s_max


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_cf_features(
    user_track_pairs: pd.DataFrame,
    train: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Compute collaborative-filtering scores for each (user_id, track_id) pair.

    Builds the interaction matrix and SVD decomposition exactly once.  UU
    cosine similarities are computed only for query users (those present in
    ``user_track_pairs``) in blocks of ``UU_BLOCK`` rows to keep peak memory
    bounded, rather than materialising the full (n_all_users × n_all_users)
    similarity matrix.

    Parameters
    ----------
    user_track_pairs : pd.DataFrame
        Columns ['user_id', 'track_id']. One row per pair to score.
    train : pd.DataFrame
        Training interactions with columns ['user_id', 'track_id'].
    config : dict
        Parsed config.yaml.  Reads ``config['collab_filtering']['cf_top_k']``
        (default 20) and ``config['collab_filtering']['n_components']``
        (default 20).

    Returns
    -------
    pd.DataFrame
        Same rows as *user_track_pairs* plus three new columns:
          cf_user_user_score  float in [0, 1] — 0.0 for cold users/tracks.
          cf_item_item_score  float in [0, 1] — 0.0 for cold users/tracks.
          cf_svd_score        float in [0, 1] — 0.0 for cold users/tracks.
    """
    cf_cfg = config.get("collab_filtering", {})
    k = int(cf_cfg.get("cf_top_k", 20))
    n_components = int(cf_cfg.get("n_components", 20))

    df = user_track_pairs[["user_id", "track_id"]].copy().reset_index(drop=True)

    # --- Build interaction matrix (once) ------------------------------------
    matrix, user_to_idx, track_to_idx = build_interaction_matrix(train)
    n_users, n_tracks = matrix.shape

    # Initialise all scores to 0.0 (cold-start default)
    uu_scores = np.zeros(len(df))
    ii_scores = np.zeros(len(df))
    svd_scores = np.zeros(len(df))

    if n_users == 0 or n_tracks == 0:
        df["cf_user_user_score"] = uu_scores
        df["cf_item_item_score"] = ii_scores
        df["cf_svd_score"] = svd_scores
        return df

    # --- Pre-normalise matrices (sparse, memory-efficient) ------------------
    # matrix_norm : (n_users, n_tracks) — normalised user vectors
    # matrix_T_norm : (n_tracks, n_users) — normalised track vectors
    matrix_norm = normalize(matrix, norm="l2")
    matrix_T_norm = normalize(matrix.T, norm="l2")

    # Binary interaction matrix for UU neighbour lookup (count → 0/1).
    # Stored in CSC format so that column slicing (binary[:, t_idx]) is O(nnz).
    binary = matrix.copy()
    binary.data = np.ones_like(binary.data)
    binary = binary.tocsc()

    # SVD latent factors
    user_vecs, track_vecs, svd_min, svd_max = _fit_svd(matrix, n_components)

    # --- Map pair rows to matrix indices ------------------------------------
    # _u_idx / _t_idx are NaN for cold users/tracks (not in training).
    df["_u_idx"] = df["user_id"].map(user_to_idx)
    df["_t_idx"] = df["track_id"].map(track_to_idx)
    warm_mask = df["_u_idx"].notna() & df["_t_idx"].notna()
    warm_df = df[warm_mask].copy()
    warm_df["_u_idx"] = warm_df["_u_idx"].astype(int)
    warm_df["_t_idx"] = warm_df["_t_idx"].astype(int)

    if warm_df.empty:
        df["cf_user_user_score"] = uu_scores
        df["cf_item_item_score"] = ii_scores
        df["cf_svd_score"] = svd_scores
        return df.drop(columns=["_u_idx", "_t_idx"])

    # --- Blocked UU + SVD scoring (per unique query user) -------------------
    # Process query users in blocks to keep peak memory ≈ UU_BLOCK × n_users × 8 bytes.
    unique_u_idxs: np.ndarray = warm_df["_u_idx"].unique()
    actual_k = min(k, n_users - 1)

    for block_start in range(0, len(unique_u_idxs), UU_BLOCK):
        block_u_idxs = unique_u_idxs[block_start:block_start + UU_BLOCK]

        # UU similarity: (UU_BLOCK, n_users) dense
        block_sims: np.ndarray = (
            matrix_norm[block_u_idxs] @ matrix_norm.T
        ).toarray()

        for bi, u_idx in enumerate(block_u_idxs):
            sims = block_sims[bi]
            sims[u_idx] = 0.0   # exclude self-similarity

            # Rows in df for this user
            user_row_mask = warm_df["_u_idx"] == u_idx
            user_warm = warm_df[user_row_mask]

            # -- UU score ----------------------------------------------------
            if actual_k > 0:
                top_k_idx = np.argpartition(sims, -actual_k)[-actual_k:]
                top_k_sims = sims[top_k_idx]
                pos = top_k_sims > 0.0
                if pos.any():
                    top_pos_idx = top_k_idx[pos]
                    top_pos_sims = top_k_sims[pos]
                    w_sum = top_pos_sims.sum()

                    for global_i, t_idx in zip(
                        user_warm.index, user_warm["_t_idx"]
                    ):
                        track_col = np.asarray(
                            binary[:, t_idx].todense()
                        ).flatten()
                        neighbor_interactions = track_col[top_pos_idx]
                        if w_sum > 0.0:
                            uu_scores[global_i] = float(
                                np.dot(top_pos_sims, neighbor_interactions) / w_sum
                            )

            # -- SVD score ---------------------------------------------------
            if user_vecs is not None:
                s_range = svd_max - svd_min
                u_vec = user_vecs[u_idx]
                for global_i, t_idx in zip(
                    user_warm.index, user_warm["_t_idx"]
                ):
                    raw = float(u_vec @ track_vecs[t_idx])
                    if s_range > 0.0:
                        svd_scores[global_i] = float(
                            np.clip((raw - svd_min) / s_range, 0.0, 1.0)
                        )

    # --- Item-item scores (vectorised per user) -----------------------------
    # Instead of a Python loop per pair, precompute a mean normalised track
    # vector for each unique query user (average over their history tracks).
    # Then ii_score(u, t) = dot(mean_vec(u), matrix_T_norm[t]), which is a
    # single sparse dot-product per pair.
    #
    # mean_vec(u): (1, n_users) dense — mean of matrix_T_norm rows for u's history.
    user_mean_vec: dict[int, np.ndarray] = {}
    for u_idx in unique_u_idxs:
        history_all = matrix[u_idx].indices   # all tracks user interacted with
        if len(history_all) > 0:
            mean_vec = np.asarray(
                matrix_T_norm[history_all].mean(axis=0)
            ).flatten()                       # (n_users,) dense
            user_mean_vec[u_idx] = mean_vec

    for global_i, row in warm_df.iterrows():
        u_idx = int(row["_u_idx"])
        t_idx = int(row["_t_idx"])
        if u_idx not in user_mean_vec:
            continue
        mean_vec = user_mean_vec[u_idx]
        t_vec = np.asarray(matrix_T_norm[t_idx].todense()).flatten()  # (n_users,)
        raw = float(np.dot(t_vec, mean_vec))
        ii_scores[global_i] = float(np.clip(raw, 0.0, 1.0))

    df["cf_user_user_score"] = uu_scores
    df["cf_item_item_score"] = ii_scores
    df["cf_svd_score"] = svd_scores
    return df.drop(columns=["_u_idx", "_t_idx"])
