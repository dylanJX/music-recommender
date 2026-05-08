# Music Recommender — Supervised Classifier Extension (PySpark ML)

This document is the primary reference for Claude Code **for this assignment only**.
The original `CLAUDE.md` describes a separate rule-based heuristic pipeline and is **not**
authoritative for the work described here. `hw4.py` is the professor's reference Python
solution using a heuristic scorer; we will **reuse its parsers and feature ideas**, but
replace the hand-tuned scoring with PySpark ML classifiers.

---

## 1. Task statement

Train **four PySpark ML classifiers** on labeled (user, track) pairs from `data/test2_new.txt`,
using features derived from each user's listening history in `data/trainItem2.txt` and track
metadata. Evaluate with `BinaryClassificationEvaluator` (areaUnderROC). Generate a submission
CSV per classifier in the format expected by `sample_submission.csv`.

The four classifiers (must use exactly these from `pyspark.ml.classification`):

1. `LogisticRegression`
2. `DecisionTreeClassifier`
3. `RandomForestClassifier`
4. `GBTClassifier`

---

## 2. Data file schemas (all verified)

### `data/test2_new.txt` — labeled training set for our classifiers

- Format: `userId|trackId|label`, pipe-delimited, no header.
- 6000 rows, 1000 users, exactly 6 rows per user (3 with label=1, 3 with label=0).
- Perfectly balanced. No class-imbalance handling needed.
- This is the **only labeled data** available for supervised training.

### `data/trainItem2.txt` — user listening histories (BLOCKED format, NOT flat CSV)

- Filename is `trainItem2.txt` (single 'r'). The original `CLAUDE.md` had a typo.
- 117 MB. **Do not** call `.toPandas()` on it.
- Format is grouped per user:
  ```
  userId|num_ratings        ← header line
  trackId<TAB>rating         ← num_ratings rating lines
  trackId<TAB>rating
  ...
  userId|num_ratings        ← next user's header
  ...
  ```
- Ratings are integers on a 0–100 scale (Yahoo! Music KDDCup format).
- **Cannot be read with `spark.read.csv` directly.** Use `hw4.py`'s `read_train_history()`
  to parse, or preprocess to a flat CSV first.

### `data/testItem2.txt` — unlabeled candidate lists (same blocked format)

- Format:
  ```
  userId|num_candidates
  trackId
  trackId
  ...
  ```
- Each user has exactly 6 candidate trackIds.
- 20000 users × 6 candidates = **120000 (user, track) pairs to predict**.
- The submission must produce a prediction for every one of these pairs.
- Use `hw4.py`'s `read_test_items()` to parse.

### `data/sample_submission.csv` — submission template

- Two columns: `TrackID,Predictor`
- 120000 data rows + 1 header.
- `TrackID` column is composite: `"<userId>_<trackId>"` (string, underscore-separated).
- `Predictor` is the prediction (0/1 for hard, probability ∈ [0,1] for soft).
- **Row order must match the sample exactly.** Use `hw4.py`'s `reorder_to_sample()` helper.

### Metadata files

- `trackData2.txt`: `TrackId|AlbumId|ArtistId|GenreId_1|...|GenreId_k`
  - Variable genre count. `"None"` literal string for missing album/artist.
  - Use `hw4.py`'s `read_track_metadata()` to parse.
- `albumData2.txt`, `artistData2.txt`, `genreData2.txt` — likely not needed for ML features;
  `trackData2.txt` already gives us album/artist/genre per track.

---

## 3. Critical clarification: training data vs. submission target

There are **two distinct test files** and they serve different roles. **Do not confuse them.**

| File | Role | Has labels? |
|---|---|---|
| `test2_new.txt` | **Training data for our classifiers.** Supervised (user, track, label) tuples. | Yes |
| `testItem2.txt` | **Submission target.** Unlabeled (user, track) pairs to predict. | No |

**Workflow:**
1. Train classifiers on `test2_new.txt` (with optional internal 70/30 split for AUC reporting).
2. After picking the best model, **retrain on all 6000 rows** of `test2_new.txt`.
3. Use that final model to predict on `testItem2.txt`'s 120000 pairs.
4. Format predictions to match `sample_submission.csv` row order and write the submission.

The 1000 users in `test2_new.txt` are **not** the same as the 20000 users in `testItem2.txt`.
Most submission users are unseen at training time. This means our features must generalize —
they should be functions of `(userId, trackId)` lookups into the training history and metadata,
not user-specific learned embeddings.

---

## 4. Pipeline overview

```
                              test2_new.txt (6000 labeled)
                                       │
                                       ▼
trainItem2.txt ──► parse ──► user_history dict
                                       │
trackData2.txt ──► parse ──► track_meta dict
                                       │
                                       ▼
                          build features for each (userId, trackId)
                                       │
                                       ▼
                          Spark DataFrame: [features Vector, label double]
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
                70/30 split for AUC          retrain on full 6000
                          │                         │
                          ▼                         ▼
                report 4 AUCs               apply best model to
                pick winner                 testItem2.txt (120k pairs)
                                                    │
                                                    ▼
                                       reorder to sample_submission.csv
                                                    │
                                                    ▼
                                       submissions/ml_<name>.csv
```

---

## 5. Feature engineering

For each `(userId, trackId)` pair, build a feature vector. **Do not include `userId` or
`trackId` themselves in the vector.**

### User-level features (computed once per user from `user_history`)

| Feature | Description | Cold-user value |
|---|---|---|
| `user_mean_rating` | mean of all ratings this user gave | global_mean_rating |
| `user_rating_count` | total items this user has rated | 0 |
| `user_rating_std` | std dev of this user's ratings | 0 |
| `user_is_cold` | 1 if user has no history, else 0 | 1 |

### Track-level features (computed once per track from `user_history`)

| Feature | Description | Cold-track value |
|---|---|---|
| `track_mean_rating` | mean rating this track received | global_mean_rating |
| `track_rating_count` | popularity proxy: count of ratings | 0 |
| `track_is_cold` | 1 if track unseen in training, else 0 | 1 |

### Interaction features (per `(userId, trackId)` pair — the high-signal ones)

Requires joining `user_history` with `track_meta` (album/artist/genre per track), then
aggregating per user. These are direct adaptations of `hw4.py`'s `user_profiles` structure.

| Feature | Description |
|---|---|
| `user_artist_mean_rating` | user's mean rating across tracks by this track's artist |
| `user_artist_count` | number of tracks by this artist the user has rated |
| `user_album_mean_rating` | same, for album |
| `user_album_count` | same, for album |
| `user_genre_mean_rating` | user's mean rating across tracks sharing any genre with this track |
| `user_genre_overlap_count` | number of this track's genres the user has any history with |

### Imputation rules for nulls (after the join)

PySpark ML errors on nulls in feature vectors with confusing messages. After the left join:
- Numeric means → impute with `global_mean_rating` (compute once).
- Counts → impute with 0.
- Indicator flags → impute with 1 (i.e. assume cold/missing if data absent).

### Assembly

All features are numeric. Use `VectorAssembler` directly. **No `StringIndexer` or
`OneHotEncoder` needed** — there are no raw categorical columns in the feature set.

---

## 6. Reusing hw4.py (do not rewrite parsers)

`hw4.py` has working, tested parsers for the blocked-format files. Import and reuse them
in our ML code rather than rewriting. Specifically:

- `read_train_history(path)` → `dict[user_id] -> list of (track_id, rating)` tuples
- `read_test_items(path)` → `dict[user_id] -> list of track_id`
- `read_track_metadata(path)` → `dict[track_id] -> {'album', 'artist', 'genres'}`
- `reorder_to_sample(rows, sample_path)` → DataFrame in submission order

Suggested module structure:
```
src/ml/
├── __init__.py
├── parsers.py     ← thin wrapper: re-exports the 4 functions from hw4.py
├── features.py    ← builds (userId, trackId) → feature dict using parsed inputs
├── train.py       ← Spark session, vector assembly, train+eval all 4 classifiers
└── submit.py      ← apply trained model to testItem2.txt, format submission
```

`features.py` should produce a **pandas DataFrame** keyed by `(userId, trackId)` with all
feature columns + label. That's then converted to a Spark DataFrame in `train.py` via
`spark.createDataFrame(pdf)`. Building features in pandas is fine here because:
- We only need features for ~6000 + ~120000 = 126000 pairs total, not the full training set.
- The expensive aggregations (per-user, per-artist, etc.) happen once over `user_history`.

---

## 7. PySpark code reference (copy-paste ready)

### Spark session
```python
from pyspark.sql import SparkSession
spark = (SparkSession.builder
         .appName('ml-recommender')
         .config('spark.driver.memory', '4g')
         .getOrCreate())
```

### Building the labeled Spark DataFrame
```python
import pandas as pd
from pyspark.ml.feature import VectorAssembler

# pdf is a pandas DataFrame with columns:
# [userId, trackId, label, feat_1, feat_2, ..., feat_N]
feature_cols = [c for c in pdf.columns if c not in ('userId', 'trackId', 'label')]

sdf = spark.createDataFrame(pdf)
sdf = sdf.withColumn('label', sdf['label'].cast('double'))

assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
sdf = assembler.transform(sdf).select('userId', 'trackId', 'features', 'label')
```

### Train/test split
```python
train, test = sdf.randomSplit([0.7, 0.3], seed=2026)
```

### The four classifiers
```python
from pyspark.ml.classification import (
    LogisticRegression, DecisionTreeClassifier,
    RandomForestClassifier, GBTClassifier,
)

classifiers = {
    'lr':  LogisticRegression(featuresCol='features', labelCol='label', maxIter=10),
    'dt':  DecisionTreeClassifier(featuresCol='features', labelCol='label', maxDepth=5),
    'rf':  RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=50),
    'gbt': GBTClassifier(featuresCol='features', labelCol='label', maxIter=20),
}
```

### Evaluation
```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(
    rawPredictionCol='rawPrediction', labelCol='label', metricName='areaUnderROC',
)

results = {}
for name, clf in classifiers.items():
    model = clf.fit(train)
    preds = model.transform(test)
    auc = evaluator.evaluate(preds)
    print(f"{name}: AUC = {auc:.4f}")
    results[name] = (model, auc)
```

### Producing the submission
```python
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

# 1. Retrain best model on full labeled data
best_name = max(results, key=lambda k: results[k][1])
best_clf = classifiers[best_name]
final_model = best_clf.fit(sdf)   # full 6000 rows

# 2. Build features for testItem2.txt's 120000 pairs (same feature pipeline)
test_pdf = build_features_for_test(testItem2_pairs, user_history, track_meta)
test_sdf = spark.createDataFrame(test_pdf)
test_sdf = assembler.transform(test_sdf)

# 3. Predict; extract probability of class 1
preds = final_model.transform(test_sdf)
prob_1 = udf(lambda v: float(v[1]), DoubleType())
out_pdf = (preds
           .select('userId', 'trackId', prob_1('probability').alias('prob'))
           .toPandas())

# 4. Format to (TrackID, Predictor) and reorder to sample
out_pdf['TrackID'] = out_pdf['userId'].astype(str) + '_' + out_pdf['trackId'].astype(str)
soft_rows = list(zip(out_pdf['TrackID'], out_pdf['prob']))

from hw4 import reorder_to_sample
submission = reorder_to_sample(soft_rows, 'data/sample_submission.csv')
submission.to_csv(f'submissions/ml_{best_name}_soft.csv', index=False)
```

For a hard-label submission, derive top-3-per-user from the probabilities (matching
`hw4.py`'s convention: rank within each user, top 3 → 1, bottom 3 → 0).

---

## 8. Implementation sequence

1. **Set up `src/ml/` directory.** Add `__init__.py`. Do not modify heuristic modules.
2. **`parsers.py`**: import the 4 functions from `hw4.py`. Verify by running each on real
   files and printing first few entries.
3. **`features.py`**: implement `build_features(pairs, user_history, track_meta) -> DataFrame`.
   `pairs` is a list of `(userId, trackId, label_or_None)` tuples. Returns pandas DataFrame.
   Test on a tiny subset (10 pairs) first, verify no nulls.
4. **`train.py`**: Spark session, load `test2_new.txt`, build features, vector-assemble,
   70/30 split, train **only LogisticRegression first**, print AUC. Get this working
   end-to-end before adding the other 3.
5. Once LR works, add DT, RF, GBT in the same loop. Report all 4 AUCs.
6. **`submit.py`**: retrain best on full labeled set, predict on `testItem2.txt`, reorder
   to sample, write CSV. Verify row count = 120000 and column names = `TrackID,Predictor`.
7. (Optional) Tune the winner with `CrossValidator` per slide 31 of the lecture.

---

## 9. Conventions

- Python 3.10+, PySpark 3.x.
- All features are numeric; cast labels to `DoubleType`.
- Use `seed=2026` for any random split for reproducibility.
- Output filenames: `submissions/ml_<classifier>_<soft|hard>.csv`
  (e.g. `ml_gbt_soft.csv`).
- Do not modify `hw4.py` itself — import from it.
- Do not modify the heuristic pipeline modules from the original `CLAUDE.md`.

---

## 10. Pitfalls to avoid

- **Do not** confuse `test2_new.txt` (labeled, our training data) with `testItem2.txt`
  (unlabeled, the submission target). They are different files with different formats.
- **Do not** try to `spark.read.csv('trainItem2.txt')` — it's a blocked format, not flat.
  Use `hw4.read_train_history`.
- **Do not** include `userId` or `trackId` in the feature vector — they are identifiers,
  not signals. The model would just memorize them from the 70% split.
- **Do not** use `OneHotEncoderEstimator` — deprecated, removed in Spark 3.x.
- **Do not** leave `label` as `IntegerType` — `BinaryClassificationEvaluator` needs `Double`.
- **Do not** assume GBT will win. With sparse aggregated features, LR is often competitive
  and trains in seconds. Benchmark all four honestly.
- **Do not** write the submission CSV in arbitrary order — it must match
  `sample_submission.csv` row order exactly. Use `reorder_to_sample`.
- **Do not** forget cold-user / cold-track imputation. Most submission users are unseen
  in `test2_new.txt`, so cold-user handling matters at submission time.
