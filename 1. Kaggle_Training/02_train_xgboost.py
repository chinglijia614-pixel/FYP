from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from shared.training_report_utils import (
    build_classification_report_df,
    build_confusion_matrix_df,
    cleanup_legacy_training_csvs,
    update_training_workbook,
)

# =============================================================================
# CONFIG
# =============================================================================
RANDOM_STATE = 42
REQUESTED_N_SPLITS = 5
N_BINS_ECE = 10

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_INPUT_DIR = PROJECT_ROOT / "input"
ARTIFACTS_DIR = PROJECT_ROOT / "Artifacts_kaggle"
OUTPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# UTILS
# =============================================================================
def timer(label: str, start: float) -> None:
    print(f"[TIMER] {label}: {time.perf_counter() - start:.2f} sec")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def multiclass_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        if not np.any(mask):
            continue

        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


def top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int = 3) -> float:
    topk = np.argsort(y_proba, axis=1)[:, -k:]
    hits = [int(y in preds) for y, preds in zip(y_true, topk)]
    return float(np.mean(hits))


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "top1_accuracy": float(accuracy_score(y_true, y_pred)),
        "top3_accuracy": float(top_k_accuracy(y_true, y_proba, 3)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "ece": float(multiclass_ece(y_true, y_proba, N_BINS_ECE)),
    }


def save_confusion_matrix_csv(y_true: np.ndarray, y_pred: np.ndarray, class_names: np.ndarray, path: Path) -> None:
    labels_used = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    used_names = [class_names[i] for i in labels_used]
    cm = confusion_matrix(y_true, y_pred, labels=labels_used)
    df_cm = pd.DataFrame(cm, index=used_names, columns=used_names).reset_index().rename(columns={"index": "true_class"})
    save_csv(df_cm, path)


def save_classification_report_csv(y_true: np.ndarray, y_pred: np.ndarray, class_names: np.ndarray, path: Path) -> None:
    labels_used = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    used_names = [class_names[i] for i in labels_used]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_used,
        target_names=used_names,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label_name, metrics in report.items():
        if isinstance(metrics, dict):
            rows.append({"label": label_name, **metrics})
    save_csv(pd.DataFrame(rows), path)


# =============================================================================
# DATA LOADING
# =============================================================================
def load_prepared_data():
    X_train = pd.read_csv(MODEL_INPUT_DIR / "X_train.csv")
    X_test = pd.read_csv(MODEL_INPUT_DIR / "X_test.csv")
    X_cv = pd.read_csv(MODEL_INPUT_DIR / "X_cv.csv")

    y_train = pd.read_csv(MODEL_INPUT_DIR / "y_train.csv")["target"].to_numpy()
    y_test = pd.read_csv(MODEL_INPUT_DIR / "y_test.csv")["target"].to_numpy()
    y_cv = pd.read_csv(MODEL_INPUT_DIR / "y_cv.csv")["target"].to_numpy()

    label_encoder: LabelEncoder = joblib.load(ARTIFACTS_DIR / "label_encoder.pkl")
    feature_names: list[str] = joblib.load(ARTIFACTS_DIR / "feature_names.pkl")

    # safety: reorder columns if needed
    X_train = X_train[feature_names]
    X_test = X_test[feature_names]
    X_cv = X_cv[feature_names]

    return X_train, X_test, X_cv, y_train, y_test, y_cv, label_encoder, feature_names


# =============================================================================
# MODEL
# =============================================================================
def build_xgboost_model(num_classes: int) -> Any:
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=RANDOM_STATE,
    )


def get_safe_n_splits(y_train: np.ndarray, requested_n_splits: int) -> tuple[int, int]:
    min_class_count = int(pd.Series(y_train).value_counts().min())
    safe = max(2, min(requested_n_splits, min_class_count))
    return safe, min_class_count


def evaluate_xgboost_fold(
    model: Any,
    X_train_fold: pd.DataFrame,
    y_train_fold: np.ndarray,
    X_eval_fold: pd.DataFrame,
    y_eval_fold: np.ndarray,
) -> dict[str, float]:
    # XGBoost needs contiguous local class ids in each fold fit
    fold_encoder = LabelEncoder()
    y_train_local = fold_encoder.fit_transform(y_train_fold)
    y_eval_local = fold_encoder.transform(y_eval_fold)

    model.fit(X_train_fold, y_train_local)
    y_pred_local = model.predict(X_eval_fold)
    y_proba_local = model.predict_proba(X_eval_fold)

    return calculate_metrics(y_eval_local, y_pred_local, y_proba_local)


def run_cross_validation(model: Any, X_cv: pd.DataFrame, y_cv: np.ndarray) -> pd.DataFrame:
    safe_n_splits, min_class_count = get_safe_n_splits(y_cv, REQUESTED_N_SPLITS)
    print(
        f"Using StratifiedKFold with n_splits={safe_n_splits} "
        f"(requested={REQUESTED_N_SPLITS}, min CV class count={min_class_count})"
    )

    skf = StratifiedKFold(n_splits=safe_n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_cv, y_cv), start=1):
        fold_model = clone(model)

        X_train_fold = X_cv.iloc[tr_idx]
        y_train_fold = y_cv[tr_idx]
        X_eval_fold = X_cv.iloc[va_idx]
        y_eval_fold = y_cv[va_idx]

        metrics = evaluate_xgboost_fold(fold_model, X_train_fold, y_train_fold, X_eval_fold, y_eval_fold)

        rows.append({
            "model": "XGBoost",
            "fold": fold_idx,
            "n_splits_requested": REQUESTED_N_SPLITS,
            "n_splits_used": safe_n_splits,
            "min_class_count_in_cv": min_class_count,
            **metrics,
        })

        print(
            f"  Fold {fold_idx}: "
            f"Top-1={metrics['top1_accuracy']:.4f}, "
            f"Top-3={metrics['top3_accuracy']:.4f}, "
            f"Macro-F1={metrics['macro_f1']:.4f}, "
            f"ECE={metrics['ece']:.4f}"
        )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    script_start = time.perf_counter()

    try:
        section_start = time.perf_counter()
        try:
            X_train, X_test, X_cv, y_train, y_test, y_cv, label_encoder, feature_names = load_prepared_data()
            class_names = label_encoder.classes_
            print("=" * 72)
            print("TRAINING ENTRY - XGBOOST")
            print("=" * 72)
            print(f"Train rows: {len(X_train)}")
            print(f"Test rows: {len(X_test)}")
            print(f"CV rows: {len(X_cv)}")
            print(f"Features: {len(feature_names)}")
            print(f"Classes: {len(class_names)}")
        finally:
            timer("load_prepared_data", section_start)

        section_start = time.perf_counter()
        try:
            model = build_xgboost_model(len(class_names))
            cv_df = run_cross_validation(model, X_cv, y_cv)

            cv_summary = (
                cv_df.groupby("model", as_index=False)
                .agg(
                    cv_top1_mean=("top1_accuracy", "mean"),
                    cv_top1_std=("top1_accuracy", "std"),
                    cv_top3_mean=("top3_accuracy", "mean"),
                    cv_top3_std=("top3_accuracy", "std"),
                    cv_macro_f1_mean=("macro_f1", "mean"),
                    cv_macro_f1_std=("macro_f1", "std"),
                    cv_ece_mean=("ece", "mean"),
                    cv_ece_std=("ece", "std"),
                    cv_folds_used=("n_splits_used", "max"),
                    min_class_count_in_cv=("min_class_count_in_cv", "max"),
                )
            )
        finally:
            timer("cross_validation", section_start)

        section_start = time.perf_counter()
        try:
            print("Training final XGBoost model on full training split...")
            model.fit(X_train, y_train)

            joblib.dump(model, ARTIFACTS_DIR / "xgboost_model.pkl")

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            holdout_metrics = {
                "model": "XGBoost",
                "split": "kaggle_holdout",
                **calculate_metrics(y_test, y_pred, y_proba),
            }

            holdout_df = pd.DataFrame([holdout_metrics])
            confusion_matrix_df = build_confusion_matrix_df(y_test, y_pred, class_names)
            classification_report_df = build_classification_report_df(y_test, y_pred, class_names)
            workbook_path = update_training_workbook(
                output_dir=OUTPUT_DIR,
                model_name="XGBoost",
                cv_df=cv_df,
                cv_summary_df=cv_summary,
                holdout_df=holdout_df,
                classification_report_df=classification_report_df,
                confusion_matrix_df=confusion_matrix_df,
            )
            cleanup_legacy_training_csvs(OUTPUT_DIR)

            print(
                f"Holdout -> Top-1={holdout_metrics['top1_accuracy']:.4f}, "
                f"Top-3={holdout_metrics['top3_accuracy']:.4f}, "
                f"Macro-F1={holdout_metrics['macro_f1']:.4f}, "
                f"ECE={holdout_metrics['ece']:.4f}"
            )
            print(f"Training report workbook updated: {workbook_path.name}")
        finally:
            timer("final_train_and_holdout", section_start)

    finally:
        timer("total_script", script_start)


if __name__ == "__main__":
    main()
