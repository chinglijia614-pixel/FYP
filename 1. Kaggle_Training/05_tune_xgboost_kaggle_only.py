from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# PURPOSE
# =============================================================================
# Kaggle-only XGBoost tuning script.
#
# This file does NOT change the prepared dataset and does NOT use external
# validation data. It only uses the existing files produced by:
#   0. Data Preparation/01_prepare_kaggle_data.py
#
# Workflow:
#   1. Load prepared Kaggle-only train / test / CV files.
#   2. Evaluate several XGBoost parameter candidates using CV on X_cv/y_cv.
#   3. Select the candidate with the best CV top-1 accuracy.
#   4. Train the best candidate on the full Kaggle training split.
#   5. Evaluate once on the Kaggle holdout test split.
#   6. Save a tuning workbook and tuned model artifact.
# =============================================================================

RANDOM_STATE = 42
N_BINS_ECE = 10

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_INPUT_DIR = PROJECT_ROOT / "input"
ARTIFACTS_DIR = PROJECT_ROOT / "Artifacts_kaggle"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TUNING_WORKBOOK_PATH = OUTPUT_DIR / "xgboost_kaggle_only_tuning_report.xlsx"
TUNED_MODEL_PATH = ARTIFACTS_DIR / "xgboost_model_tuned.pkl"
TUNED_PARAMS_PATH = ARTIFACTS_DIR / "xgboost_tuned_params.json"

# Candidate order matters. The first candidate is the current baseline from
# 02_train_xgboost.py so the report can compare improvement directly.
CANDIDATES: list[dict[str, Any]] = [
    {
        "candidate": "baseline_current",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 1,
        "gamma": 0.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    },
    {
        "candidate": "balanced_600_d6_lr004",
        "n_estimators": 600,
        "learning_rate": 0.04,
        "max_depth": 6,
        "min_child_weight": 2,
        "gamma": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "reg_lambda": 3.0,
        "reg_alpha": 0.1,
    },
    {
        "candidate": "regularized_850_d6_lr003",
        "n_estimators": 850,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 2,
        "gamma": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "reg_lambda": 4.0,
        "reg_alpha": 0.1,
    },
    {
        "candidate": "shallow_900_d5_lr0035",
        "n_estimators": 900,
        "learning_rate": 0.035,
        "max_depth": 5,
        "min_child_weight": 2,
        "gamma": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 2.0,
        "reg_alpha": 0.05,
    },
    {
        "candidate": "depth7_700_lr0035",
        "n_estimators": 700,
        "learning_rate": 0.035,
        "max_depth": 7,
        "min_child_weight": 2,
        "gamma": 0.1,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 4.0,
        "reg_alpha": 0.1,
    },
    {
        "candidate": "conservative_1000_d6_lr0025",
        "n_estimators": 1000,
        "learning_rate": 0.025,
        "max_depth": 6,
        "min_child_weight": 3,
        "gamma": 0.2,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_lambda": 6.0,
        "reg_alpha": 0.2,
    },
]


def timer(label: str, start: float) -> None:
    print(f"[TIMER] {label}: {time.perf_counter() - start:.2f} sec")


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
    k = min(k, y_proba.shape[1])
    topk = np.argsort(y_proba, axis=1)[:, -k:]
    hits = [int(y in preds) for y, preds in zip(y_true, topk)]
    return float(np.mean(hits))


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "top1_accuracy": float(accuracy_score(y_true, y_pred)),
        "top3_accuracy": float(top_k_accuracy(y_true, y_proba, 3)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ece": float(multiclass_ece(y_true, y_proba, N_BINS_ECE)),
    }


def load_prepared_data():
    required_files = [
        MODEL_INPUT_DIR / "X_train.csv",
        MODEL_INPUT_DIR / "X_test.csv",
        MODEL_INPUT_DIR / "X_cv.csv",
        MODEL_INPUT_DIR / "y_train.csv",
        MODEL_INPUT_DIR / "y_test.csv",
        MODEL_INPUT_DIR / "y_cv.csv",
        ARTIFACTS_DIR / "label_encoder.pkl",
        ARTIFACTS_DIR / "feature_names.pkl",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Prepared Kaggle files are missing. Run this first from repo root:\n"
            "python \"0. Data Preparation/01_prepare_kaggle_data.py\"\n\n"
            "Missing files:\n - " + "\n - ".join(missing)
        )

    X_train = pd.read_csv(MODEL_INPUT_DIR / "X_train.csv")
    X_test = pd.read_csv(MODEL_INPUT_DIR / "X_test.csv")
    X_cv = pd.read_csv(MODEL_INPUT_DIR / "X_cv.csv")

    y_train = pd.read_csv(MODEL_INPUT_DIR / "y_train.csv")["target"].to_numpy()
    y_test = pd.read_csv(MODEL_INPUT_DIR / "y_test.csv")["target"].to_numpy()
    y_cv = pd.read_csv(MODEL_INPUT_DIR / "y_cv.csv")["target"].to_numpy()

    label_encoder: LabelEncoder = joblib.load(ARTIFACTS_DIR / "label_encoder.pkl")
    feature_names: list[str] = joblib.load(ARTIFACTS_DIR / "feature_names.pkl")

    X_train = X_train[feature_names].astype(np.uint8, copy=False)
    X_test = X_test[feature_names].astype(np.uint8, copy=False)
    X_cv = X_cv[feature_names].astype(np.uint8, copy=False)

    return X_train, X_test, X_cv, y_train, y_test, y_cv, label_encoder, feature_names


def candidate_to_xgb_params(candidate: dict[str, Any], num_classes: int, n_jobs: int) -> dict[str, Any]:
    return {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": n_jobs,
        "verbosity": 1,
        "n_estimators": candidate["n_estimators"],
        "learning_rate": candidate["learning_rate"],
        "max_depth": candidate["max_depth"],
        "min_child_weight": candidate["min_child_weight"],
        "gamma": candidate["gamma"],
        "subsample": candidate["subsample"],
        "colsample_bytree": candidate["colsample_bytree"],
        "reg_lambda": candidate["reg_lambda"],
        "reg_alpha": candidate["reg_alpha"],
    }


def get_safe_cv_folds(y: np.ndarray, requested_folds: int) -> tuple[int, int]:
    min_class_count = int(pd.Series(y).value_counts().min())
    safe_folds = max(2, min(requested_folds, min_class_count))
    return safe_folds, min_class_count


def evaluate_candidate_cv(
    candidate: dict[str, Any],
    X_cv: pd.DataFrame,
    y_cv: np.ndarray,
    cv_folds: int,
    n_jobs: int,
) -> list[dict[str, Any]]:
    safe_folds, min_class_count = get_safe_cv_folds(y_cv, cv_folds)
    skf = StratifiedKFold(n_splits=safe_folds, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, Any]] = []

    print("=" * 72)
    print(f"Candidate: {candidate['candidate']}")
    print(f"CV folds used: {safe_folds} (requested={cv_folds}, min_class_count={min_class_count})")
    print(json.dumps(candidate, indent=2))
    print("=" * 72)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv), start=1):
        fold_start = time.perf_counter()

        X_train_fold = X_cv.iloc[train_idx]
        y_train_fold = y_cv[train_idx]
        X_val_fold = X_cv.iloc[val_idx]
        y_val_fold = y_cv[val_idx]

        # XGBoost fold training is safer with contiguous local label IDs.
        fold_encoder = LabelEncoder()
        y_train_local = fold_encoder.fit_transform(y_train_fold)
        y_val_local = fold_encoder.transform(y_val_fold)

        model = xgb.XGBClassifier(
            **candidate_to_xgb_params(
                candidate,
                num_classes=len(fold_encoder.classes_),
                n_jobs=n_jobs,
            )
        )
        model.fit(X_train_fold, y_train_local)

        y_pred = model.predict(X_val_fold)
        y_proba = model.predict_proba(X_val_fold)
        metrics = calculate_metrics(y_val_local, y_pred, y_proba)

        row = {
            "candidate": candidate["candidate"],
            "fold": fold_idx,
            "cv_folds_used": safe_folds,
            "min_class_count_in_cv": min_class_count,
            **{k: candidate[k] for k in candidate if k != "candidate"},
            **metrics,
            "fit_eval_seconds": round(time.perf_counter() - fold_start, 2),
        }
        rows.append(row)

        print(
            f"Fold {fold_idx}: "
            f"Top-1={metrics['top1_accuracy']:.4f}, "
            f"Top-3={metrics['top3_accuracy']:.4f}, "
            f"Macro-F1={metrics['macro_f1']:.4f}, "
            f"ECE={metrics['ece']:.4f}, "
            f"Time={row['fit_eval_seconds']:.2f}s"
        )

    return rows


def summarise_cv_results(cv_results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        cv_results.groupby("candidate", as_index=False)
        .agg(
            cv_top1_mean=("top1_accuracy", "mean"),
            cv_top1_std=("top1_accuracy", "std"),
            cv_top3_mean=("top3_accuracy", "mean"),
            cv_top3_std=("top3_accuracy", "std"),
            cv_macro_f1_mean=("macro_f1", "mean"),
            cv_macro_f1_std=("macro_f1", "std"),
            cv_ece_mean=("ece", "mean"),
            cv_ece_std=("ece", "std"),
            avg_fit_eval_seconds=("fit_eval_seconds", "mean"),
        )
        .sort_values(
            ["cv_top1_mean", "cv_macro_f1_mean", "cv_top3_mean"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )
    return summary


def train_best_and_evaluate_holdout(
    best_candidate: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    class_names: np.ndarray,
    n_jobs: int,
) -> tuple[pd.DataFrame, Any]:
    start = time.perf_counter()
    print("=" * 72)
    print("Training best candidate on full Kaggle training split")
    print(json.dumps(best_candidate, indent=2))
    print("=" * 72)

    model = xgb.XGBClassifier(
        **candidate_to_xgb_params(
            best_candidate,
            num_classes=len(class_names),
            n_jobs=n_jobs,
        )
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = calculate_metrics(y_test, y_pred, y_proba)

    holdout_df = pd.DataFrame([
        {
            "candidate": best_candidate["candidate"],
            "split": "kaggle_holdout_test_once_after_cv_selection",
            **{k: best_candidate[k] for k in best_candidate if k != "candidate"},
            **metrics,
            "train_eval_seconds": round(time.perf_counter() - start, 2),
        }
    ])

    print(
        f"Holdout result -> "
        f"Top-1={metrics['top1_accuracy']:.4f}, "
        f"Top-3={metrics['top3_accuracy']:.4f}, "
        f"Macro-F1={metrics['macro_f1']:.4f}, "
        f"ECE={metrics['ece']:.4f}"
    )
    return holdout_df, model


def save_report(
    cv_results: pd.DataFrame,
    cv_summary: pd.DataFrame,
    holdout_df: pd.DataFrame,
    best_candidate: dict[str, Any],
) -> None:
    params_df = pd.DataFrame([best_candidate])
    notes_df = pd.DataFrame([
        {
            "note": "This report is Kaggle-only. Candidate selection used CV on X_cv/y_cv. Holdout test was evaluated once after selecting the best CV candidate."
        },
        {
            "note": "The tuned model is saved as Artifacts_kaggle/xgboost_model_tuned.pkl and does not overwrite xgboost_model.pkl."
        },
    ])

    with pd.ExcelWriter(TUNING_WORKBOOK_PATH, engine="openpyxl") as writer:
        cv_summary.to_excel(writer, sheet_name="cv_summary", index=False)
        cv_results.to_excel(writer, sheet_name="cv_fold_results", index=False)
        holdout_df.to_excel(writer, sheet_name="holdout_result", index=False)
        params_df.to_excel(writer, sheet_name="best_params", index=False)
        notes_df.to_excel(writer, sheet_name="notes", index=False)

    print(f"Saved tuning workbook: {TUNING_WORKBOOK_PATH}")


def print_recommended_function(best_candidate: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("RECOMMENDED build_xgboost_model() TO COPY INTO 02_train_xgboost.py")
    print("=" * 72)
    print("def build_xgboost_model(num_classes: int) -> Any:")
    print("    return xgb.XGBClassifier(")
    print('        objective="multi:softprob",')
    print("        num_class=num_classes,")
    print('        eval_metric="mlogloss",')
    print(f"        n_estimators={best_candidate['n_estimators']},")
    print(f"        max_depth={best_candidate['max_depth']},")
    print(f"        learning_rate={best_candidate['learning_rate']},")
    print(f"        min_child_weight={best_candidate['min_child_weight']},")
    print(f"        gamma={best_candidate['gamma']},")
    print(f"        subsample={best_candidate['subsample']},")
    print(f"        colsample_bytree={best_candidate['colsample_bytree']},")
    print(f"        reg_lambda={best_candidate['reg_lambda']},")
    print(f"        reg_alpha={best_candidate['reg_alpha']},")
    print('        tree_method="hist",')
    print("        random_state=RANDOM_STATE,")
    print("    )")
    print("=" * 72 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Kaggle-only XGBoost candidates.")
    parser.add_argument(
        "--candidate_limit",
        type=int,
        default=4,
        help="Number of candidates to run from the ordered candidate list. Use 0 to run all candidates. Default: 4.",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=2,
        help="Requested StratifiedKFold count for candidate selection. Default: 2 for speed.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="XGBoost CPU threads. Use -1 for all available cores. Default: -1.",
    )
    return parser.parse_args()


def main() -> None:
    script_start = time.perf_counter()
    args = parse_args()

    X_train, X_test, X_cv, y_train, y_test, y_cv, label_encoder, feature_names = load_prepared_data()
    class_names = label_encoder.classes_

    print("=" * 72)
    print("KAGGLE-ONLY XGBOOST TUNING")
    print("=" * 72)
    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"CV rows: {len(X_cv):,}")
    print(f"Features: {len(feature_names):,}")
    print(f"Classes: {len(class_names):,}")
    print(f"candidate_limit: {args.candidate_limit}")
    print(f"cv_folds requested: {args.cv_folds}")
    print(f"n_jobs: {args.n_jobs}")

    candidates = CANDIDATES if args.candidate_limit == 0 else CANDIDATES[: args.candidate_limit]
    all_rows: list[dict[str, Any]] = []

    for candidate in candidates:
        candidate_rows = evaluate_candidate_cv(
            candidate=candidate,
            X_cv=X_cv,
            y_cv=y_cv,
            cv_folds=args.cv_folds,
            n_jobs=args.n_jobs,
        )
        all_rows.extend(candidate_rows)

    cv_results = pd.DataFrame(all_rows)
    cv_summary = summarise_cv_results(cv_results)

    print("\n" + "=" * 72)
    print("CV SUMMARY SORTED BY TOP-1")
    print("=" * 72)
    print(cv_summary.to_string(index=False))

    best_name = str(cv_summary.iloc[0]["candidate"])
    best_candidate = next(c for c in candidates if c["candidate"] == best_name)

    holdout_df, tuned_model = train_best_and_evaluate_holdout(
        best_candidate=best_candidate,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
        n_jobs=args.n_jobs,
    )

    joblib.dump(tuned_model, TUNED_MODEL_PATH)
    TUNED_PARAMS_PATH.write_text(json.dumps(best_candidate, indent=2), encoding="utf-8")

    save_report(
        cv_results=cv_results,
        cv_summary=cv_summary,
        holdout_df=holdout_df,
        best_candidate=best_candidate,
    )

    print(f"Saved tuned model: {TUNED_MODEL_PATH}")
    print(f"Saved tuned params JSON: {TUNED_PARAMS_PATH}")
    print_recommended_function(best_candidate)
    timer("total_script", script_start)


if __name__ == "__main__":
    main()
