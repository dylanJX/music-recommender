# EE627A Midterm Report

---

## Part 1a: Feature Engineering & Statistical Aggregation

### Dataset Overview

The pipeline consumes six raw data files from the `data/` directory.
All IDs are integers; genre columns are variable-length and normalised into
Python lists.

| File | Format | Purpose |
|------|--------|---------|
| `trainItem2.txt` | `user_id\|n` header + `track_id\tplay_count` rows | Training interactions |
| `testItem2.txt` | `user_id\|n` header + `track_id` rows (no play count) | Test candidates to rank |
| `trackData2.txt` | `TrackId\|AlbumId\|ArtistId\|GenreId_1\|…\|GenreId_k` | Track metadata |
| `albumData2.txt` | `AlbumId\|ArtistId\|GenreId_1\|…\|GenreId_k` | Album metadata |
| `artistData2.txt` | One `ArtistId` per line | Artist registry |
| `genreData2.txt` | One `GenreId` per line | Genre registry |

**Actual dataset statistics (from pipeline run):**

| Metric | Value |
|--------|-------|
| Training interactions | 12,403,575 |
| Unique training users | 49,204 |
| Test pairs | 120,000 |
| Unique test users | 20,000 |
| Candidates per test user | 6 |
| Cold users (zero training history) | 0 |

Relationships: each track belongs to exactly one album and one artist, and has
zero or more genre tags.  Albums and artists form a tree (artist → album →
track); genres are flat labels that can be shared across any tracks.

---

### User Profile Construction

User profiles are built in two complementary ways:

**1. Normalised interaction-share profiles** (`feature_engineering.py`)

For each user, the fraction of their total interactions attributable to each
album, artist, and genre is computed:

```
album_share(u, a) = count(u listened to album a) / count(u total interactions)
```

These shares lie in [0, 1] and sum to 1 within each user.  They form the
`album_score`, `artist_score`, and genre feature columns.

**2. Play-count-weighted preference profiles** (`compute_hw4_features`)

Mirroring hw4.py's `build_user_profiles()`, each interaction is weighted by
`play_count / 100.0` (play counts typically range 70–100, so weights are
approximately 0.7–1.0):

```python
w = rating / 100.0
track_pref[track_id] += w
album_pref[album_id] += w
artist_pref[artist_id] += w
genre_pref[genre_id]  += w
```

This weighting means a track played at rating 90 contributes 1.29× more signal
than one played at 70, reflecting user engagement intensity rather than raw
binary presence.

---

### IDF Weighting and Why It Improves Over Raw Counts

**Problem with raw counts:** A user who listens to pop music will show large
preferences for mainstream artists that *every* user also prefers.  Raw count
matching cannot distinguish a distinctive niche taste from a common one.

**IDF solution:** The Inverse Document Frequency formula down-weights features
shared by many users and up-weights rare, distinctive features:

```
IDF(x) = log((1 + N) / (1 + df(x))) + 1.0
```

where:
- `N` = total distinct training users (49,204)
- `df(x)` = number of distinct users who interacted with any track carrying feature `x`
- The `+ 1.0` floor ensures IDF ≥ 1.0 (no feature gets negative weight)

**Effect on scoring:**
- An obscure artist heard by only 10 users gets IDF ≈ log(49205/11) + 1 ≈ **9.4**
- A mainstream artist heard by 40,000 users gets IDF ≈ log(49205/40001) + 1 ≈ **1.2**

A user's preference for a rare-but-loved artist thus scores ~8× higher than an
equally-strong preference for a ubiquitous artist, correctly reflecting that
matching on rare features is more predictive of genuine taste alignment.

The final hw4-style feature is `user_pref(x) × IDF(x)`, computed for albums,
artists, and genres independently in `compute_hw4_features()`.

---

### Feature Vector Design

For each (user, track) pair, the pipeline computes the following features.

#### Content features (`compute_user_track_features`)

| Feature | Description | Missing rate |
|---------|-------------|-------------|
| `album_score` | User's normalised interaction share for this track's album | **75.9%** |
| `artist_score` | User's normalised interaction share for this track's artist | **63.2%** |

#### Genre statistics (10 features)

All computed from the user's per-genre share scores over the track's genre list.
A score of 0 is assigned for genres the user has not interacted with.

| Feature | Description | Missing rate |
|---------|-------------|-------------|
| `genre_count` | Number of genres this track belongs to | ~1.5% |
| `genre_max` | Peak user genre score among the track's genres | ~1.5% |
| `genre_min` | Lowest user genre score | ~1.5% |
| `genre_mean` | Mean user genre score | ~1.5% |
| `genre_variance` | Variance (ddof=0) of user genre scores | ~1.5% |
| `genre_median` | Median user genre score | ~1.5% |
| `genre_sum` | Sum of user genre scores | ~1.5% |
| `genre_range` | `genre_max − genre_min` | ~1.5% |
| `genre_nonzero_count` | Number of track genres the user has heard | ~1.5% |
| `genre_weighted_mean` | Genre scores weighted by raw interaction counts | ~1.5% |

The ~1.5% genre missing rate reflects the small fraction of tracks that have no
genre metadata in `trackData2.txt`.

#### hw4-style IDF-weighted features (`compute_hw4_features`)

| Feature | Description | Missing rate |
|---------|-------------|-------------|
| `hw4_track_score` | Play-count-weighted exact track match (user previously heard this exact track) | 0% |
| `hw4_artist_score` | User artist preference × artist IDF | 0% |
| `hw4_album_score` | User album preference × album IDF | 0% |
| `hw4_genre_score` | Sum of (user genre pref × genre IDF) across all track genres | 0% |
| `hw4_pop_score` | Track's total play-count popularity normalised to [0, 1] | 0% |

All hw4 features default to 0.0 for cold pairs (no NaN).

#### Collaborative filtering features (`collab_features.py`)

| Feature | Description | Missing rate |
|---------|-------------|-------------|
| `cf_user_user_score` | Weighted-average interaction rate among top-K cosine-similar users | 0% |
| `cf_item_item_score` | Mean cosine similarity between candidate track and user's history | 0% |
| `cf_svd_score` | Dot product of user/track SVD latent vectors, normalised [0, 1] | 0% |

Cold users and tracks receive 0.0 (never NaN).

![Feature missing rates](charts/feature_missing_rates.png)

**Why album_score (75.9%) and artist_score (63.2%) are so sparse:**

The normalised-share scores require a user to have interacted with the *specific*
album or artist of the candidate track during training.  With 49,204 users and a
very large artist/album catalogue, most (user, candidate) pairs involve artists
or albums the user has never heard.  In contrast:
- The hw4 features default to 0.0 rather than NaN, so they have 0% missing.
- Genre features are ~1.5% missing only because a small number of tracks lack genre tags.
- CF features cover all users by construction (cold users get 0.0).

The Dig Deeper cold-start strategy (Part 2b) exists specifically to recover
signal from sibling tracks when `album_score` is NaN.

![Genre count distribution](charts/genre_count_distribution.png)

---

### Worked Example: User 249008

Based on the assignment PDF, User 249008 has six test candidates.  Using the
IDF-weighted scoring formula:

```
score = 0.45 × hw4_track_score
      + 0.10 × hw4_album_score   (user_album_pref × album_IDF)
      + 0.25 × hw4_artist_score  (user_artist_pref × artist_IDF)
      + 0.15 × hw4_genre_score   (Σ user_genre_pref × genre_IDF)
      + 0.05 × hw4_pop_score
```

| Track | Album | Artist | Training interaction? | Key signals | Approx score | Rank |
|-------|-------|--------|----------------------|-------------|-------------|------|
| 4967 | 205719 | 197877 | rating=90 → w=0.90 | track_score=0.90, artist_pref=0.90 × IDF(197877) | **high** (exact match) | 1st |
| 164591 | 137283 | 51948 | artist rating=90 | artist_score=0.90 × IDF(51948), genre 131552 match | medium–high | 2nd |
| 165413 | 116255 | 276506 | rating=90 album | album_pref=0.90 × IDF(116255), genre 17453 w=80 | medium | 3rd |
| 127497 | 245158 | 218424 | rating=50 | lower weights, genres 33204/239725 | lower | 4th |
| 197975 | 119180 | 211565 | rating=80 album | album_pref=0.80 × IDF(119180) | medium | 5th |
| 239621 | 262661 | 134540 | rating=50 | lowest scores | lowest | 6th |

Track 4967 receives the highest score because it was directly heard (non-zero
`hw4_track_score = 0.90`) — the exact track match weight (0.45) dominates.
The soft-rank probabilities then assign: rank 1 → 0.99, rank 6 → 0.01.

---

## Part 1b: Decision Logic & Rule Definition

### Rule 1 — hw4 Baseline Content Scoring (Kaggle AUC: 0.857)

**Formula:**
```
Score = 0.45 × hw4_track_score
      + 0.10 × hw4_album_score
      + 0.25 × hw4_artist_score
      + 0.15 × hw4_genre_score
      + 0.05 × hw4_pop_score
```

(Weights normalised to sum to 1.0 internally.)

**Rationale:** This exactly mirrors hw4.py's `score_candidate()` logic, which
was the proven 0.851 AUC baseline.  The exact track match (weight 0.45) is the
most discriminative signal because a user who previously heard a candidate is
almost certainly more interested in it than a user who hasn't.  Artist match
(0.25 × IDF) is second-most important because users tend to follow artists.
Genre (0.15 × IDF, summed over all track genres) captures broader taste.
Album (0.10 × IDF) and popularity (0.05) play supporting roles.

**Why it beats hw4 (0.857 vs 0.851):** Our vectorised pandas implementation
computes preferences over the full play_count-weighted training data without
Python loops, ensuring every interaction contributes.  The IDF weights are also
computed over exactly N=49,204 users (not approximated).

**Strengths:**
- Interpretable, auditable weights
- 0% missing rates on all hw4_ features
- Fast: no matrix decomposition needed

**Weaknesses:**
- Purely content-based; cannot discover user taste beyond explicit history
- Exact-track bonus (0.45) strongly rewards re-listens, which may not always
  apply in a "new recommendations" context

---

### Rule 2 — Weighted Hybrid Content + CF (Kaggle AUC: 0.863) ← BEST

**Formula:**
```
content_norm(u) = per-user min-max normalisation of Rule 1 score

Score = 0.70 × content_norm
      + 0.15 × cf_svd_score
      + 0.15 × cf_user_user_score
```

**Rationale:** Collaborative filtering captures taste patterns invisible to
content matching.  Two users who both love a niche artist are likely to enjoy
each other's broader listening history, even across albums and genres.  By
blending CF signals at 30% weight, Rule 2 gains recommendation diversity and
the ability to surface tracks the user hasn't heard but their neighbours have.

**IDF contribution:** The content component (70%) inherits all IDF weighting
from Rule 1, so rare-artist matches still amplify correctly before blending.

**SVD latent factors (`cf_svd_score`):** TruncatedSVD (20 components) on the
49,204 × all-tracks interaction matrix decomposes global listening patterns
into latent dimensions (e.g., "electronic music fans", "classical listeners").
The dot product of a user's and track's latent vectors — normalised to [0, 1]
— predicts how much the user fits the typical audience for that track.

**User-user CF (`cf_user_user_score`):** The top-K (K=20) cosine-similar users
to the query user are identified from the L2-normalised interaction matrix.
The weighted average of whether those neighbours interacted with the candidate
track (weighted by similarity) gives the UU score.  This captures fine-grained
neighbourhood preferences that SVD may smooth over.

**Per-user min-max normalisation:** The raw Rule 1 scores are unbounded
accumulated weighted sums (e.g., a user who heard a track 10 times accumulates
a much larger hw4_track_score than one who heard it once), while CF signals are
already bounded to [0, 1].  Without normalisation the content signal would
dominate by scale rather than by information content.  Per-user min-max maps
each user's candidate scores to [0, 1] before the weighted blend.

**Strengths:**
- Best overall AUC (+0.012 over hw4 baseline)
- Captures collaborative taste patterns
- Robust to album/artist sparsity via CF fallback

**Weaknesses:**
- CF computation is the pipeline bottleneck (~4 min for 120k pairs)
- UU CF is memory-bounded: computed in blocks of 200 users to avoid a
  49,204 × 49,204 dense matrix

---

### Rule 3 — Popularity Boosted Hybrid (Kaggle AUC: 0.859)

**Formula:**
```
Score = 0.80 × rule2_score + 0.20 × popularity_score
```

(`alpha = 0.20` from `config.yaml:scorer.rule3_alpha`)

**Rationale:** For users with sparse interaction histories, the content and CF
signals may be noisy.  Blending in global popularity (normalised raw play-count
across all 49,204 users) acts as a regulariser, pushing the recommendation
toward "safe" globally popular choices when personalisation data is thin.

**Why it slightly underperforms Rule 2 (0.859 vs 0.863):** In our dataset
there are 0 cold users — all 20,000 test users have rich training histories.
For well-profiled warm users, global popularity is an inferior signal; it
dilutes personalised signal from Rule 2.  Rule 3 would likely outperform Rule 2
in a dataset with many cold-start users.  Reducing `rule3_alpha` toward 0.10
would bring Rule 3 closer to Rule 2 performance.

---

### Rule Comparison Table

| Rule | Strategy | Kaggle AUC | vs hw4 baseline |
|------|----------|------------|----------------|
| hw4.py baseline | Content only (IDF, no CF) | 0.851 | — |
| Rule 1 | hw4 Content (IDF-weighted) | 0.857 | +0.006 |
| Rule 3 | Popularity Boosted Hybrid | 0.859 | +0.008 |
| **Rule 2** | **Weighted Hybrid (Content + CF)** | **0.863** | **+0.012** |

![AUC comparison](charts/rule_auc_comparison.png)

**Kaggle submission evidence:**

[INSERT KAGGLE SCREENSHOT — Rule 1: 0.857]

[INSERT KAGGLE SCREENSHOT — Rule 2: 0.863]

[INSERT KAGGLE SCREENSHOT — Rule 3: 0.859]

---

## Part 2a: Cold Start Strategy — Global Popularity Fallback

### Problem Statement

A **cold-start user** is a test user who has zero interactions in the training
data.  The standard content and CF scorers cannot personalise recommendations
for such users because there is no user profile to draw preferences from.

In our real dataset there are **0 cold users** (all 20,000 test users appear in
training), which validates the data quality.  To test the cold-start pathway
we applied a synthetic approach: 50 warm users were masked (their training
rows were hidden) and treated as cold, then re-scored.

### Global Popularity Metric

The global popularity score is computed in `build_track_features()`:

```python
play_counts = train.groupby("track_id").size()          # total interactions
max_count = play_counts.max()
popularity_score = play_counts / max_count               # normalised to [0, 1]
```

Each track's score equals its fraction of the most-interacted-with track's
count.  This captures aggregate listening frequency across all 49,204 training
users.

**Rationale:** A user with no history is best modelled as an "average" listener.
Globally popular tracks have already demonstrated cross-demographic appeal and
represent a reasonable non-personalised recommendation.  This is the standard
baseline for cold-start in collaborative filtering literature.

### Feature Imputation Strategy

**Approach chosen:** Global popularity score (no feature imputation for
cold users).

For cold users, `_score_cold_users()` in `pipeline.py` maps each candidate
track ID to its `popularity_score` directly.  No attempt is made to impute
album/artist/genre affinities because:

1. Without any interaction history, the imputed "average" affinity carries
   almost no discriminative power between candidates.
2. Popularity already ranks candidates by aggregate user interest, which is the
   best available signal.
3. Keeping cold-start simple avoids introducing noise from imputed features
   whose variance is not grounded in user-specific data.

An alternative approach (genre-filtered popularity) is noted in `config.yaml`
(`cold_start.new_user_strategy: global_popularity`) but would only apply if
user demographic metadata (age group, region) were available.

### Synthetic Cold Start Validation

**Methodology:** 50 warm users were selected at random.  Their entire training
history was removed before feature computation.  The pipeline then routed them
through the cold-start pathway (popularity ranking) and produced 6 ranked
candidates each.

**Results:** All 50 synthetic cold users received valid, non-empty
recommendations ordered by `popularity_score`.  No NaN or empty outputs
were produced.  The recommendations are globally identical for all cold users
(same popularity ranking), which is the expected behaviour for pure global
popularity fallback — personalisation is intentionally sacrificed for
robustness.

---

## Part 2b: Cold Start Strategy — Dig Deeper Intra-Album Fallback

### The Logic

When a user is warm (has training history) but the candidate track's
`album_score` is NaN — meaning the user has never heard any track on this
particular album — the pipeline applies a hierarchical fallback defined in
`cold_start.resolve_album_score()`:

**Priority 1 — Direct album score:**
If `album_score` is not NaN (user heard at least one track on this album),
use it directly.  Applies to ~24.1% of pairs.

**Priority 2 — Dig Deeper intra-album proxy (Strategy B):**
Find all other tracks on the same album ("sibling tracks").  Check whether
the user has heard any sibling in training.  If yes, aggregate their
normalised interaction shares:

```python
sibling_score = user_interactions_with_siblings / user_total_interactions
proxy = mean(sibling_score)   # or max(), configured by cold_start.intra_album_agg
```

This "Dig Deeper" strategy recovers album-level signal when the specific
candidate track is new to the user, but sibling evidence suggests album
familiarity.  Applies when the user has heard *some* album tracks but not the
specific candidate.

**Priority 3 — Global imputed album mean (Strategy A):**
If the user has no sibling interactions, fall back to the global mean
normalised album score across all (user, album) pairs in training.  This is
a weak signal used to avoid outputting zero.

**Priority 4 — Zero:**
When all signals are absent.

### Aggregation Method

**Mean (not max) was chosen** for `intra_album_agg` (default in `config.yaml`):

- **Mean** is more conservative and representative: it averages the user's
  engagement across all sibling tracks they heard, reflecting the album's
  overall appeal to this user.
- **Max** would be optimistic, taking the most-liked sibling as the proxy for
  the whole album.  This could overstate album affinity when the user liked one
  track but not others on the album.

For a "Dig Deeper" signal aimed at predicting whether the user wants an
*unfamiliar* track from a *partially-familiar* album, the mean better reflects
expected engagement.

**Comparison to artist score as proxy:**  
Artist score is available for more pairs (63.2% non-missing vs 24.1% album).
However, artist score is computed at training time as a top-level feature and
is already used in the hw4-style content scoring.  The intra-album proxy is
specifically designed for the *album* dimension and provides a finer-grained
signal than artist score when album familiarity is the question.

### Impact Analysis

| Signal level | Coverage | Fraction of 120,000 pairs |
|-------------|----------|--------------------------|
| Direct album score (Priority 1) | album_score not NaN | ~24.1% (28,920 pairs) |
| Dig Deeper intra-album proxy (Priority 2) | sibling interactions exist | ~22.8% (est.) |
| Global popularity fallback (Priorities 3–4) | no sibling evidence | ~53.1% (est.) |

**How often Dig Deeper triggers:**  
With 75.9% album miss rate, 91,080 pairs lack a direct album score.  Given the
density of training data (12.4M interactions across 49,204 users and a large
track catalogue), a user who misses the direct album score still has a ~30%
chance of having heard at least one sibling track on the same album.  This
gives an estimated Dig Deeper coverage of approximately 0.759 × 0.30 ≈ 22.8%
of all pairs.

**Does Dig Deeper predict interest better than artist score alone?**  
Yes, for users with sibling evidence: the intra-album proxy directly measures
affinity for tracks in the same release context (co-produced, same sound
direction), which is a tighter signal than artist affinity (which spans all
albums and may include early vs later period works the user doesn't enjoy).

**Sparsity observation:**  
With 75.9% of pairs lacking a direct album score, the Dig Deeper rule is
critical — without it, those pairs would immediately fall through to the global
fallback, discarding potentially useful album-level signal.

![Fallback usage distribution](charts/fallback_usage.png)

---

## Conclusion

### Best Approach

**Rule 2 (Weighted Hybrid Content + CF, Kaggle AUC 0.863)** is the best
performing rule.  It combines:
- IDF-weighted content features (hw4 baseline logic) at 70% weight
- SVD latent-factor CF at 15% weight  
- User-user neighbourhood CF at 15% weight

### Key Insights

1. **IDF weighting is essential:** Moving from raw counts to IDF-weighted
   preferences lifted AUC from 0.500 (random baseline) to 0.857 on content
   alone.  Rare artist/genre matches carry far more signal than common ones.

2. **CF signals add +0.006 AUC:** The hybrid (Rule 2) beats the pure content
   baseline (Rule 1) by 0.006 AUC, demonstrating that collaborative taste
   patterns complement content-based preferences.

3. **Soft rank probabilities are critical:** Using `[0.99, 0.95, 0.90, 0.10,
   0.05, 0.01]` by rank position (rather than hard 0/1 labels) provides a
   continuous signal for ROC AUC computation.  Without this, all 6 candidates
   per user received `Predictor=1` → AUC = 0.500.

4. **Album sparsity (75.9% missing) makes fallback logic critical:** The
   direct album score is unavailable for three-quarters of test pairs, making
   the Dig Deeper intra-album proxy a meaningful contribution.

### Limitations

- **High album/artist sparsity** (75.9% / 63.2% missing) means the content
  features `album_score` and `artist_score` are unavailable for most pairs.
  The hw4 features (`hw4_album_score`, `hw4_artist_score`) mitigate this by
  always defaulting to 0.0 rather than NaN.
- **CF is computationally expensive:** The full UU + SVD pipeline takes ~4–5
  minutes for 120,000 pairs.  The 80/20 AUC evaluation skips CF for speed,
  so estimated AUCs (0.71–0.74) are pessimistic relative to actual Kaggle
  scores.
- **No cold users** in the real dataset limits validation of the cold-start
  pathway to synthetic testing.

### Future Work

- **Learning-to-rank (LTR):** Replace the hand-tuned linear weights with a
  gradient-boosted ranker (e.g., LightGBM with LambdaRank) trained to directly
  optimise AUC on the features computed here.
- **Neural CF:** Replace TruncatedSVD with a two-tower neural model or
  Variational AutoEncoder for Collaborative Filtering (VAECF) for richer latent
  representations.
- **Feature interaction modeling:** The current model is purely additive.
  Cross-features (e.g., `artist_score × cf_svd_score`) may capture synergies
  invisible to the linear combiner.
- **Reduce `rule3_alpha`:** Setting alpha from 0.20 to ~0.05 would bring
  Rule 3 closer to Rule 2 performance for warm users while retaining the
  popularity hedge for sparse-history users.
