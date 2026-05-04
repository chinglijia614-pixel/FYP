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
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# PURPOSE
# =============================================================================
# Kaggle-only XGBoost tuning script.
#
# This version is designed for a large Kaggle-only dataset where one full
# XGBoost train/test run can take many hours.
#
# Default behaviour is FAST SCREENING:
#   1. Load the existing prepared Kaggle files.
#   2. Take a class-safe screening sample from X_cv/y_cv.
#   3. Split that screening sample into train/validation while keeping at least
#      one sample per disease class in both screening train and validation.
#   4. Compare candidate parameter sets on the smaller screening split.
#   5. Save the screening report and print the best candidate.
#
# It does NOT touch X_test by default.
# To do the expensive final full training + X_test evaluation, add:
#   --run_final_train
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


def select_candidates(candidate_limit: int, candidate_name: str | None) -> list[dict[str, Any]]:
    if candidate_name:
        selected = [c for c in CANDIDATES if c["candidate"] == candidate_name]
        if not selected:
            available = ", ".join(c["candidate"] for c in CANDIDATES)
            raise ValueError(f"Unknown candidate_name={candidate_name}. Available: {available}")
        return selected
    return CANDIDATES if candidate_limit == 0 else CANDIDATES[:candidate_limit]


def make_screening_pool(
    X_cv: pd.DataFrame,
    y_cv: np.ndarray,
    screening_rows: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    if screening_rows <= 0 or screening_rows >= len(X_cv):
        print(f"Using full X_cv for screening: {len(X_cv):,} rows")
        return X_cv.reset_index(drop=True), y_cv

    classes = np.unique(y_cv)
    min_required = len(classes) * 2
    if screening_rows < min_required:
        raise ValueError(
            "screening_rows is too small for the number of classes. "
            f"Use at least {min_required} rows."
        )

    rng = np.random.default_rng(RANDOM_STATE)
    selected_idx: list[int] = []
    remaining_idx: list[int] = []
    skipped_classes = 0

    # Keep at least 2 rows per class in the screening pool so the later
    # train/validation split can place at least 1 row in each side.
    for label in classes:
        class_idx = np.flatnonzero(y_cv == label)
        rng.shuffle(class_idx)
        if len(class_idx) < 2:
            skipped_classes += 1
            continue
        selected_idx.extend(class_idx[:2].tolist())
        remaining_idx.extend(class_idx[2:].tolist())

    remaining_budget = screening_rows - len(selected_idx)
    if remaining_budget > 0 and remaining_idx:
        if remaining_budget >= len(remaining_idx):
            selected_idx.extend(remaining_idx)
        else:
            selected_idx.extend(rng.choice(remaining_idx, size=remaining_budget, replace=False).tolist())

    selected_idx = sorted(selected_idx)
    X_pool = X_cv.iloc[selected_idx].reset_index(drop=True)
    y_pool = y_cv[selected_idx]

    pool_min_count = int(pd.Series(y_pool).value_counts().min())
    print(f"Using class-safe screening pool: {len(X_pool):,} rows from X_cv")
    print(f"Screening pool classes: {len(np.unique(y_pool)):,}")
    print(f"Screening pool minimum class count: {pool_min_count}")
    if skipped_classes:
        print(f"[WARN] Skipped classes with fewer than 2 rows in X_cv: {skipped_classes}")

    return X_pool, y_pool


def make_screening_split(
    X_pool: pd.DataFrame,
    y_pool: np.ndarray,
    validation_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    classes = np.unique(y_pool)
    desired_val_size = int(round(len(y_pool) * validation_size))
    desired_val_size = max(desired_val_size, len(classes))
    desired_val_size = min(desired_val_size, len(y_pool) - len(classes))

    val_idx: list[int] = []
    extra_candidates: list[int] = []

    # Put 1 row per class into validation, reserve 1 row per class for train,
    # then use the remaining rows to fill validation to the requested size.
    for label in classes:
        class_idx = np.flatnonzero(y_pool == label)
        rng.shuffle(class_idx)
        if len(class_idx) < 2:
            raise ValueError(
                f"Class {label} has only {len(class_idx)} row(s) in screening pool. "
                "Increase --screening_rows."
            )
        val_idx.append(int(class_idx[0]))
        extra_candidates.extend(class_idx[2:].tolist())

    extra_needed = desired_val_size - len(val_idx)
    if extra_needed > 0 and extra_candidates:
        extra_needed = min(extra_needed, len(extra_candidates))
        val_idx.extend(rng.choice(extra_candidates, size=extra_needed, replace=False).tolist())

    val_idx = np.array(sorted(set(val_idx)), dtype=int)
    train_mask = np.ones(len(y_pool), dtype=bool)
    train_mask[val_idx] = False
    train_idx = np.flatnonzero(train_mask)

    y_train_screen = y_pool[train_idx]
    y_val_screen = y_pool[val_idx]
    print(f"Screening split train min class count: {int(pd.Series(y_train_screen).value_counts().min())}")
    print(f"Screening split validation min class count: {int(pd.Series(y_val_screen).value_counts().min())}")

    return (
        X_pool.iloc[train_idx].reset_index(drop=True),
        X_pool.iloc[val_idx].reset_index(drop=True),
        y_train_screen,
        y_val_screen,
    )


def evaluate_candidate_screening(
    candidate: dict[str, Any],
    X_screen_train: pd.DataFrame,
    y_screen_train: np.ndarray,
    X_screen_val: pd.DataFrame,
    y_screen_val: np.ndarray,
    n_jobs: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    print("=" * 72)
    print(f"Screening candidate: {candidate['candidate']}")
    print(json.dumps(candidate, indent=2))
    print("=" * 72)

    local_encoder = LabelEncoder()
    y_train_local = local_encoder.fit_transform(y_screen_train)
    y_val_local = local_encoder.transform(y_screen_val)

    model = xgb.XGBClassifier(
        **candidate_to_xgb_params(
            candidate,
            num_classes=len(local_encoder.classes_),
            n_jobs=n_jobs,
        )
    )
    model.fit(X_screen_train, y_train_local)

    y_pred = model.predict(X_screen_val)
    y_proba = model.predict_proba(X_screen_val)
    metrics = calculate_metrics(y_val_local, y_pred, y_proba)

    row = {
        "candidate": candidate["candidate"],
        **{k: candidate[k] for k in candidate if k != "candidate"},
        **metrics,
        "screening_seconds": round(time.perf_counter() - start, 2),
    }

    print(
        f"Screening result -> "
        f"Top-1={metrics['top1_accuracy']:.4f}, "
        f"Top-3={metrics['top3_accuracy']:.4f}, "
        f"Macro-F1={metrics['macro_f1']:.4f}, "
        f"ECE={metrics['ece']:.4f}, "
        f"Time={row['screening_seconds']:.2f}s"
    )
    return row


def summarise_screening_results(results_df: pd.DataFrame) -> pd.DataFrame:
    return results_df.sort_values(
        ["top1_accuracy", "macro_f1", "top3_accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


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
    print("FINAL FULL TRAINING on X_train, then one-time X_test evaluation")
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
            "split": "kaggle_holdout_test_once_after_screening_selection",
            **{k: best_candidate[k] for k in best_candidate if k != "candidate"},
            **metrics,
            "full_train_eval_seconds": round(time.perf_counter() - start, 2),
        }
    ])

    print(
        f"Final holdout result -> "
        f"Top-1={metrics['top1_accuracy']:.4f}, "
        f"Top-3={metrics['top3_accuracy']:.4f}, "
        f"Macro-F1={metrics['macro_f1']:.4f}, "
        f"ECE={metrics['ece']:.4f}"
    )
    return holdout_df, model


def save_report(
    screening_results: pd.DataFrame,
    screening_summary: pd.DataFrame,
    holdout_df: pd.DataFrame,
    best_candidate: dict[str, Any],
    notes: list[str],
) -> None:
    params_df = pd.DataFrame([best_candidate])
    notes_df = pd.DataFrame({"note": notes})

    with pd.ExcelWriter(TUNING_WORKBOOK_PATH, engine="openpyxl") as writer:
        screening_summary.to_excel(writer, sheet_name="screening_summary", index=False)
        screening_results.to_excel(writer, sheet_name="screening_results", index=False)
        if not holdout_df.empty:
            holdout_df.to_excel(writer, sheet_name="final_holdout_result", index=False)
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
    parser = argparse.ArgumentParser(description="Fast Kaggle-only XGBoost parameter screening.")
    parser.add_argument(
        "--candidate_limit",
        type=int,
        default=4,
        help="Number of candidates to screen from the ordered list. Use 0 to screen all. Default: 4.",
    )
    parser.add_argument(
        "--candidate_name",
        type=str,
        default=None,
        help="Run only one candidate by name, e.g. balanced_600_d6_lr004.",
    )
    parser.add_argument(
        "--screening_rows",
        type=int,
        default=40000,
        help="Rows sampled from X_cv/y_cv for fast screening. Use 0 for full X_cv. Default: 40000.",
    )
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.2,
        help="Validation fraction inside the screening pool. Default: 0.2.",
    )
    parser.add_argument(
        "--run_final_train",
        action="store_true",
        help="After screening, train the selected best candidate on full X_train and evaluate X_test once.",
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
    print("FAST KAGGLE-ONLY XGBOOST PARAMETER SCREENING")
    print("=" * 72)
    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"CV rows: {len(X_cv):,}")
    print(f"Features: {len(feature_names):,}")
    print(f"Classes: {len(class_names):,}")
    print(f"candidate_limit: {args.candidate_limit}")
    print(f"candidate_name: {args.candidate_name}")
    print(f"screening_rows: {args.screening_rows}")
    print(f"validation_size: {args.validation_size}")
    print(f"run_final_train: {args.run_final_train}")
    print(f"n_jobs: {args.n_jobs}")

    candidates = select_candidates(args.candidate_limit, args.candidate_name)
    X_pool, y_pool = make_screening_pool(X_cv, y_cv, args.screening_rows)
    X_screen_train, X_screen_val, y_screen_train, y_screen_val = make_screening_split(
        X_pool,
        y_pool,
        args.validation_size,
    )
    print(f"Screening train rows: {len(X_screen_train):,}")
    print(f"Screening validation rows: {len(X_screen_val):,}")

    rows = []
    for candidate in candidates:
        rows.append(
            evaluate_candidate_screening(
                candidate=candidate,
                X_screen_train=X_screen_train,
                y_screen_train=y_screen_train,
                X_screen_val=X_screen_val,
                y_screen_val=y_screen_val,
                n_jobs=args.n_jobs,
            )
        )

    screening_results = pd.DataFrame(rows)
    screening_summary = summarise_screening_results(screening_results)

    print("\n" + "=" * 72)
    print("SCREENING SUMMARY SORTED BY TOP-1")
    print("=" * 72)
    print(screening_summary.to_string(index=False))

    best_name = str(screening_summary.iloc[0]["candidate"])
    best_candidate = next(c for c in candidates if c["candidate"] == best_name)
    TUNED_PARAMS_PATH.write_text(json.dumps(best_candidate, indent=2), encoding="utf-8")

    holdout_df = pd.DataFrame()
    if args.run_final_train:
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
        print(f"Saved tuned model: {TUNED_MODEL_PATH}")
    else:
        print("\nFinal full training was skipped.")
        print("Add --run_final_train only when you are ready to spend one full training run on the selected candidate.")

    notes = [
        "Default mode is fast screening only. It does not evaluate X_test unless --run_final_train is used.",
        "screening_summary is for parameter selection only and should not be reported as final test accuracy.",
        "Final reportable Kaggle holdout accuracy is only in final_holdout_result when --run_final_train is used.",
        "The tuned params JSON is saved even when final full training is skipped.",
    ]
    save_report(
        screening_results=screening_results,
        screening_summary=screening_summary,
        holdout_df=holdout_df,
        best_candidate=best_candidate,
        notes=notes,
    )

    print(f"Saved best params JSON: {TUNED_PARAMS_PATH}")
    print_recommended_function(best_candidate)
    timer("total_script", script_start)


if __name__ == "__main__":
    main()
