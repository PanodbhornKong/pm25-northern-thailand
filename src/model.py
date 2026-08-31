"""
model.py
--------
Part C (machine learning) for HW4 (DS-270702).

Run this from the REPO ROOT:

    python3 src/model.py

What it does:
  1. Loads data/processed/daily_combined.csv.
  2. Builds two framings, per Section 3 of the lab sheet:
       - Classification: will tomorrow exceed the PCD Moderate
         threshold (37.5 ug/m3)?  [priority framing, per Section 2]
       - Regression: what will tomorrow's daily mean PM2.5 be?
  3. Uses a chronological train/test split (the most recent ~20% of
     calendar days, for BOTH locations together, held out as test) --
     never a random split, because this is time-series data.
  4. Cross-validates on the training portion only, using a date-based
     expanding-window split (the same idea as sklearn's
     TimeSeriesSplit, applied to calendar dates so both locations'
     rows for a given date always land in the same fold).
  5. Reports two baselines per Section 6 Rule 3:
       - persistence (tomorrow = today) -- the primary baseline for
         time-series data
       - majority-class / training-mean -- a secondary reference
  6. Reports recall on the exceedance class (not just accuracy) for
     classification, per Section 6 Rule 6, and MAE/RMSE for regression.
  7. Saves metrics to outputs/results/metrics.json and writes
     Checkpoint C5's answer to outputs/results/checkpoints_c5.txt.

Feature availability (Section 6 Rule 2): every feature used here
describes TODAY (the day the forecast is made from), which has
already finished by the time a real warning would be issued for
TOMORROW. None of the features require knowing anything about the
future, so all of them are legitimately available at prediction time.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
)


def rmse(y_true, y_pred):
    """sqrt(MSE), computed manually so this works across sklearn versions
    (newer sklearn removed the `squared=` argument from mean_squared_error)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# --------------------------------------------------------------------------

PROCESSED_PATH = os.path.join("data", "processed", "daily_combined.csv")
RESULTS_DIR = os.path.join("outputs", "results")

TEST_FRACTION = 0.20   # most recent 20% of calendar days held out as test
N_CV_SPLITS = 5
RANDOM_STATE = 42

FEATURE_COLS = [
    "pm2_5_mean", "pm10_mean", "carbon_monoxide_mean", "dust_mean",
    "temperature_mean", "humidity_mean", "wind_speed_mean",
    "wind_direction_mean_deg", "precipitation_sum", "pressure_mean",
    "exceeds_pcd_moderate", "consecutive_bad_days",
]
CLASSIFICATION_TARGET = "target_next_day_exceeds"
REGRESSION_TARGET = "target_next_day_pm2_5"

# --------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    df = pd.get_dummies(df, columns=["location"], prefix="loc")
    location_cols = [c for c in df.columns if c.startswith("loc_")]
    return df, location_cols


def chronological_split(df: pd.DataFrame):
    """Hold out the most recent TEST_FRACTION of calendar days as test,
    for both locations together, so no location's future data leaks
    into another location's training rows for an earlier date."""
    unique_dates = np.sort(df["date"].unique())
    cutoff_idx = int(len(unique_dates) * (1 - TEST_FRACTION))
    cutoff_date = unique_dates[cutoff_idx]
    train_df = df[df["date"] < cutoff_date].copy()
    test_df = df[df["date"] >= cutoff_date].copy()
    return train_df, test_df, cutoff_date


def date_based_cv_folds(train_df: pd.DataFrame, n_splits: int):
    """Expanding-window folds over calendar dates (like TimeSeriesSplit,
    but applied to unique dates so both locations stay together)."""
    unique_dates = np.sort(train_df["date"].unique())
    fold_edges = np.array_split(np.arange(len(unique_dates)), n_splits + 1)
    folds = []
    for i in range(n_splits):
        train_dates = unique_dates[np.concatenate(fold_edges[: i + 1])]
        val_dates = unique_dates[fold_edges[i + 1]]
        train_idx = train_df.index[train_df["date"].isin(train_dates)]
        val_idx = train_df.index[train_df["date"].isin(val_dates)]
        folds.append((train_idx, val_idx))
    return folds


def run_classification(train_df, test_df, feature_cols, log_lines, metrics):
    train = train_df.dropna(subset=[CLASSIFICATION_TARGET])
    test = test_df.dropna(subset=[CLASSIFICATION_TARGET])

    X_train, y_train = train[feature_cols], train[CLASSIFICATION_TARGET].astype(bool)
    X_test, y_test = test[feature_cols], test[CLASSIFICATION_TARGET].astype(bool)

    log_lines.append("\n" + "=" * 60)
    log_lines.append("CLASSIFICATION -- will tomorrow exceed 37.5 ug/m3?")
    log_lines.append("=" * 60)
    log_lines.append(f"Train rows: {len(train)}, Test rows: {len(test)}")
    log_lines.append(f"Positive class rate (exceeds) -- train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

    # ---- Baselines (Section 6, Rule 3) -------------------------------
    # Each row's "today" value is in the SAME row (features describe today,
    # the target column describes tomorrow) -- no lookup into train_df needed.
    persistence_pred = test["exceeds_pcd_moderate"].astype(bool)
    persistence_recall = recall_score(y_test, persistence_pred, zero_division=0)
    persistence_acc = accuracy_score(y_test, persistence_pred)

    majority_class = y_train.mode()[0]
    majority_pred = pd.Series(majority_class, index=y_test.index)
    majority_recall = recall_score(y_test, majority_pred, zero_division=0)
    majority_acc = accuracy_score(y_test, majority_pred)

    log_lines.append("\nBaselines:")
    log_lines.append(f"  Persistence (today's class = tomorrow's class): "
                      f"accuracy={persistence_acc:.3f}, recall(exceeds)={persistence_recall:.3f}")
    log_lines.append(f"  Majority class ('{majority_class}' always): "
                      f"accuracy={majority_acc:.3f}, recall(exceeds)={majority_recall:.3f}")

    # ---- Cross-validation on training data only ------------------------
    cv_recalls = []
    for fold_train_idx, fold_val_idx in date_based_cv_folds(train, N_CV_SPLITS):
        fold_train_idx = fold_train_idx.intersection(train.index)
        fold_val_idx = fold_val_idx.intersection(train.index)
        if len(fold_train_idx) == 0 or len(fold_val_idx) == 0:
            continue
        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced")
        clf.fit(train.loc[fold_train_idx, feature_cols], train.loc[fold_train_idx, CLASSIFICATION_TARGET].astype(bool))
        val_pred = clf.predict(train.loc[fold_val_idx, feature_cols])
        cv_recalls.append(recall_score(train.loc[fold_val_idx, CLASSIFICATION_TARGET].astype(bool), val_pred, zero_division=0))

    mean_cv_recall = float(np.mean(cv_recalls)) if cv_recalls else float("nan")
    log_lines.append(f"\nCross-validation (date-based, {len(cv_recalls)} folds): mean recall(exceeds) = {mean_cv_recall:.3f}")

    # ---- Final model on the full training set, evaluated on test -------
    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced")
    clf.fit(X_train, y_train)
    test_pred = clf.predict(X_test)

    test_acc = accuracy_score(y_test, test_pred)
    test_recall = recall_score(y_test, test_pred, zero_division=0)
    test_precision = precision_score(y_test, test_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[False, True]).ravel()

    log_lines.append("\nFinal model (Random Forest) on held-out test set:")
    log_lines.append(f"  accuracy={test_acc:.3f}, precision(exceeds)={test_precision:.3f}, recall(exceeds)={test_recall:.3f}")
    log_lines.append(f"  confusion matrix -- TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    log_lines.append(
        f"  {fn} missed warnings (false negatives) out of {tp + fn} true exceedance days in the test set: "
        f"these are the days the model would have failed to warn a hospital about."
    )

    if abs(mean_cv_recall - test_recall) > 0.10:
        log_lines.append(
            "  NOTE: cross-validation recall and test recall differ by more than 0.10 -- "
            "investigate whether the test period (most recent dry season) behaves differently "
            "from earlier training folds before trusting the test score alone."
        )

    metrics["classification"] = {
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "baseline_persistence": {"accuracy": persistence_acc, "recall_exceeds": persistence_recall},
        "baseline_majority_class": {"accuracy": majority_acc, "recall_exceeds": majority_recall},
        "cv_mean_recall_exceeds": mean_cv_recall,
        "test": {
            "accuracy": test_acc, "precision_exceeds": test_precision, "recall_exceeds": test_recall,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
    }


def run_regression(train_df, test_df, feature_cols, log_lines, metrics):
    train = train_df.dropna(subset=[REGRESSION_TARGET])
    test = test_df.dropna(subset=[REGRESSION_TARGET])

    X_train, y_train = train[feature_cols], train[REGRESSION_TARGET]
    X_test, y_test = test[feature_cols], test[REGRESSION_TARGET]

    log_lines.append("\n" + "=" * 60)
    log_lines.append("REGRESSION -- tomorrow's daily mean PM2.5 (ug/m3)")
    log_lines.append("=" * 60)
    log_lines.append(f"Train rows: {len(train)}, Test rows: {len(test)}")

    # ---- Baselines -------------------------------------------------------
    persistence_pred = test["pm2_5_mean"]
    persistence_mae = mean_absolute_error(y_test, persistence_pred)
    persistence_rmse = rmse(y_test, persistence_pred)

    mean_pred = pd.Series(y_train.mean(), index=y_test.index)
    mean_mae = mean_absolute_error(y_test, mean_pred)
    mean_rmse = rmse(y_test, mean_pred)

    log_lines.append("\nBaselines:")
    log_lines.append(f"  Persistence (today's PM2.5 = tomorrow's): MAE={persistence_mae:.2f}, RMSE={persistence_rmse:.2f}")
    log_lines.append(f"  Training mean (always predict {y_train.mean():.2f}): MAE={mean_mae:.2f}, RMSE={mean_rmse:.2f}")

    # ---- Final model -------------------------------------------------------
    reg = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)
    reg.fit(X_train, y_train)
    test_pred = reg.predict(X_test)

    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = rmse(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)

    log_lines.append("\nFinal model (Random Forest) on held-out test set:")
    log_lines.append(f"  MAE={test_mae:.2f}, RMSE={test_rmse:.2f}, R2={test_r2:.3f}")

    beats_persistence = test_mae < persistence_mae
    log_lines.append(
        f"  Model {'beats' if beats_persistence else 'does NOT beat'} the persistence baseline on MAE. "
        + ("" if beats_persistence else "This is reported honestly, not hidden -- see Section 6 note on baselines.")
    )

    metrics["regression"] = {
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "baseline_persistence": {"mae": persistence_mae, "rmse": persistence_rmse},
        "baseline_mean": {"mae": mean_mae, "rmse": mean_rmse},
        "test": {"mae": test_mae, "rmse": test_rmse, "r2": test_r2},
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df, location_cols = load_data()
    feature_cols = FEATURE_COLS + location_cols

    train_df, test_df, cutoff_date = chronological_split(df)

    log_lines = ["Checkpoint C5 and model results -- generated by model.py"]
    log_lines.append(f"Chronological split: train = before {cutoff_date}, test = {cutoff_date} onward")
    log_lines.append(f"(most recent {TEST_FRACTION:.0%} of calendar days, both locations, held out as test)")

    metrics = {}
    run_classification(train_df, test_df, feature_cols, log_lines, metrics)
    run_regression(train_df, test_df, feature_cols, log_lines, metrics)

    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)
    log_lines.append(f"\nMetrics saved -> {metrics_path}")

    report_path = os.path.join(RESULTS_DIR, "checkpoints_c5.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print("\n".join(log_lines))
    print(f"\nC5 checkpoint notes written to {report_path}")


if __name__ == "__main__":
    main()
