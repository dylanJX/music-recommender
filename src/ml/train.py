"""
train.py — Train 4 PySpark ML classifiers on test2_new.txt, report AUC.

Run: py -3 -m src.ml.train
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, IntegerType, DoubleType as SparkDoubleType,
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression, DecisionTreeClassifier,
    RandomForestClassifier, GBTClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from src.ml.parsers import (
    read_train_history, read_test_items, read_track_metadata,
    reorder_to_sample, DATA_DIR,
)
from src.ml.features import build_features, FEATURE_COLS


def load_labeled_pairs(path):
    """Parse test2_new.txt into list of (userId, trackId, label) tuples."""
    pairs = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            pairs.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return pairs


def _pdf_to_spark(spark, pdf):
    """Load pandas DataFrame into Spark via CSV to avoid Python worker issues."""
    tmpdir = tempfile.mkdtemp()
    csv_path = str(Path(tmpdir) / "data.csv")
    pdf.to_csv(csv_path, index=False)

    schema = StructType(
        [StructField("userId", IntegerType()), StructField("trackId", IntegerType())]
        + [StructField("label", SparkDoubleType())]
        + [StructField(c, SparkDoubleType()) for c in FEATURE_COLS]
    )

    sdf = spark.read.csv(csv_path, header=True, schema=schema)
    return sdf


def main():
    # --- 1. Parse data ---
    print("Parsing data files...")
    user_history = read_train_history(str(DATA_DIR / "trainItem2.txt"))
    track_meta = read_track_metadata(str(DATA_DIR / "trackData2.txt"))
    labeled_pairs = load_labeled_pairs(str(DATA_DIR / "test2_new.txt"))
    print(f"  {len(user_history)} users in history, "
          f"{len(track_meta)} tracks in metadata, "
          f"{len(labeled_pairs)} labeled pairs")

    # --- 2. Build features ---
    print("Building features...")
    pdf = build_features(labeled_pairs, user_history, track_meta)
    # Reorder columns: userId, trackId, label, then features
    pdf = pdf[["userId", "trackId", "label"] + FEATURE_COLS]
    pdf["label"] = pdf["label"].astype(float)
    print(f"  Feature matrix: {pdf.shape}")

    # --- 3. Spark session ---
    spark = (SparkSession.builder
             .master("local[1]")
             .appName("ml-recommender")
             .config("spark.driver.memory", "4g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    sdf = _pdf_to_spark(spark, pdf)

    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
    sdf = assembler.transform(sdf).select("userId", "trackId", "features", "label")

    # --- 4. Train/test split ---
    train, test = sdf.randomSplit([0.7, 0.3], seed=2026)
    print(f"  Train: {train.count()}, Test: {test.count()}")

    # --- 5. Train and evaluate all 4 classifiers ---
    classifiers = {
        "lr":  LogisticRegression(featuresCol="features", labelCol="label", maxIter=10),
        "dt":  DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=5),
        "rf":  RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50),
        "gbt": GBTClassifier(featuresCol="features", labelCol="label", maxIter=20),
    }

    evaluator = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC",
    )

    results = {}
    for name, clf in classifiers.items():
        print(f"  Training {name}...")
        model = clf.fit(train)
        preds = model.transform(test)
        auc = evaluator.evaluate(preds)
        print(f"    {name}: AUC = {auc:.4f}")
        results[name] = (model, auc)

    best_name = max(results, key=lambda k: results[k][1])
    print(f"\nBest classifier: {best_name} (AUC = {results[best_name][1]:.4f})")

    # --- 6. Retrain best on full labeled data ---
    print(f"Retraining {best_name} on all {sdf.count()} labeled rows...")
    final_model = classifiers[best_name].fit(sdf)

    # --- 7. Build features for submission (testItem2.txt) ---
    print("Building submission features for 120k pairs...")
    test_users = read_test_items(str(DATA_DIR / "testItem2.txt"))
    test_pairs = []
    for uid, tids in test_users.items():
        for tid in tids:
            test_pairs.append((uid, tid, None))

    test_pdf = build_features(test_pairs, user_history, track_meta)
    test_pdf["label"] = 0.0
    test_pdf = test_pdf[["userId", "trackId", "label"] + FEATURE_COLS]

    test_sdf = _pdf_to_spark(spark, test_pdf)
    test_sdf = assembler.transform(test_sdf).select("userId", "trackId", "features", "label")

    # --- 8. Predict ---
    print("Predicting...")
    preds = final_model.transform(test_sdf)

    # Collect to pandas and extract probability of class 1 from Vector
    out_pdf = preds.select("userId", "trackId", "probability").toPandas()
    out_pdf["prob"] = out_pdf["probability"].apply(lambda v: float(v[1]))
    out_pdf["TrackID"] = out_pdf["userId"].astype(str) + "_" + out_pdf["trackId"].astype(str)

    # --- 9. Soft submission ---
    soft_rows = list(zip(out_pdf["TrackID"], out_pdf["prob"]))
    sub_soft = reorder_to_sample(soft_rows, str(DATA_DIR / "sample_submission.csv"))

    submissions_dir = Path(__file__).resolve().parent.parent.parent / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    soft_path = submissions_dir / f"ml_{best_name}_soft.csv"
    sub_soft.to_csv(soft_path, index=False)
    print(f"Soft submission: {soft_path} ({len(sub_soft)} rows)")

    # --- 10. Hard submission (top-3-per-user → 1, bottom-3 → 0) ---
    out_pdf_sorted = out_pdf.sort_values(["userId", "prob"], ascending=[True, False])
    hard_labels = []
    for uid, group in out_pdf_sorted.groupby("userId"):
        for rank, (_, row) in enumerate(group.iterrows()):
            hard_labels.append((row["TrackID"], 1 if rank < 3 else 0))

    sub_hard = reorder_to_sample(hard_labels, str(DATA_DIR / "sample_submission.csv"))
    hard_path = submissions_dir / f"ml_{best_name}_hard.csv"
    sub_hard.to_csv(hard_path, index=False)
    print(f"Hard submission: {hard_path} ({len(sub_hard)} rows)")

    spark.stop()
    print("Done.")


if __name__ == "__main__":
    main()
