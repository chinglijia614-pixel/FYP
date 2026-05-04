from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from openpyxl import load_workbook
from sklearn.metrics import accuracy_score, f1_score


RANDOM_STATE = 42
N_BINS_ECE = 10
EARLY_STOPPING_ROUNDS = 20
DEFAULT_MAX_SAMPLE_WEIGHT = 4.0

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_INPUT_DIR = PROJECT_ROOT / "input"
ARTIFACTS_DIR = PROJECT_ROOT / "Artifacts_kaggle"
OUTPUT_DIR = PROJECT_ROOT / "output"
BACKUP_DIR = ARTIFACTS_DIR / "backups"
MODEL_PATH = ARTIFACTS_DIR / "xgboost_model.pkl"
WORKBOOK_PATH = OUTPUT_DIR / "kaggle_model_training_report.xlsx"
RESULTS_CSV_PATH = OUTPUT_DIR / "xgboost_kaggle_tuning_candidates.csv"
PROMOTED_METRICS_PATH = OUTPUT_DIR / "xgboost_kaggle_tuning_promoted_metrics.csv"


def multiclass_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for idx in range(n_bins):
        lo, hi = bins[idx], bins[idx + 1]
        if idx == n_bins - 1:
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


def compute_sample_weights(y: np.ndarray, max_weight: float = DEFAULT_MAX_SAMPLE_WEIGHT) -> np.ndarray:
    counts = pd.Series(y).value_counts()
    class_weights = 1.0 / np.sqrt(counts)
    sample_weights = pd.Series(y).map(class_weights).to_numpy(dtype=np.float32)
    sample_weights = sample_weights / float(sample_weights.mean())
    sample_weights = np.clip(sample_weights, 1.0, max_weight).astype(np.float32)
    sample_weights = sample_weights / float(sample_weights.mean())
    return sample_weights


def load_prepared_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    feature_names: list[str] = joblib.load(ARTIFACTS_DIR / "feature_names.pkl")
    X_train = pd.read_csv(MODEL_INPUT_DIR / "X_train.csv")[feature_names]
    X_cv = pd.read_csv(MODEL_INPUT_DIR / "X_cv.csv")[feature_names]
    X_test = pd.read_csv(MODEL_INPUT_DIR / "X_test.csv")[feature_names]
    y_train = pd.read_csv(MODEL_INPUT_DIR / "y_train.csv")["target"].to_numpy()
    y_cv = pd.read_csv(MODEL_INPUT_DIR / "y_cv.csv")["target"].to_numpy()
    y_test = pd.read_csv(MODEL_INPUT_DIR / "y_test.csv")["target"].to_numpy()
    return X_train, X_cv, X_test, y_train, y_cv, y_test


def get_candidate_configs() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": "k01_baseline_es",
            "use_sample_weight": False,
            "n_estimators": 500,
            "max_depth": 8,
            "learning_rate": 0.05,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        {
            "candidate_id": "k02_shallow_regularized",
            "use_sample_weight": False,
            "n_estimators": 600,
            "max_depth": 6,
            "learning_rate": 0.04,
            "min_child_weight": 2.0,
            "gamma": 0.05,
            "reg_alpha": 0.05,
            "reg_lambda": 1.2,
        },
        {
            "candidate_id": "k03_lower_lr_deeper",
            "use_sample_weight": False,
            "n_estimators": 700,
            "max_depth": 7,
            "learning_rate": 0.03,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        {
            "candidate_id": "k04_weighted_shallow",
            "use_sample_weight": True,
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 2.0,
            "gamma": 0.05,
            "reg_alpha": 0.05,
            "reg_lambda": 1.2,
        },
        {
            "candidate_id": "k05_weighted_baseline",
            "use_sample_weight": True,
            "n_estimators": 500,
            "max_depth": 8,
            "learning_rate": 0.05,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
    ]


def build_model(num_classes: int, candidate: dict[str, Any], n_estimators_override: int | None = None) -> Any:
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        n_estimators=n_estimators_override or candidate["n_estimators"],
        max_depth=candidate["max_depth"],
        learning_rate=candidate["learning_rate"],
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=candidate["min_child_weight"],
        gamma=candidate["gamma"],
        reg_alpha=candidate["reg_alpha"],
        reg_lambda=candidate["reg_lambda"],
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )


def evaluate_model(model: Any, X_eval: pd.DataFrame, y_eval: np.ndarray) -> dict[str, float]:
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)
    return calculate_metrics(y_eval, y_pred, y_proba)


def get_best_iteration(model: Any) -> int:
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None or best_iteration < 0:
        return int(model.get_params()["n_estimators"])
    return int(best_iteration) + 1


def run_candidate(
    candidate: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_cv: pd.DataFrame,
    y_cv: np.ndarray,
) -> dict[str, Any]:
    model = build_model(len(np.unique(y_train)), candidate)
    fit_kwargs: dict[str, Any] = {
        "eval_set": [(X_cv, y_cv)],
        "verbose": False,
    }
    if candidate["use_sample_weight"]:
        fit_kwargs["sample_weight"] = compute_sample_weights(y_train)

    fit_start = time.perf_counter()
    model.fit(X_train, y_train, **fit_kwargs)
    fit_seconds = time.perf_counter() - fit_start
    metrics = evaluate_model(model, X_cv, y_cv)
    best_trees = get_best_iteration(model)

    return {
        **candidate,
        "fit_seconds": round(fit_seconds, 2),
        "best_trees": best_trees,
        **{f"cv_{key}": value for key, value in metrics.items()},
    }


def rank_candidates(results_df: pd.DataFrame) -> pd.DataFrame:
    return results_df.sort_values(
        by=["cv_top1_accuracy", "cv_macro_f1", "cv_top3_accuracy", "cv_ece"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def backup_current_assets() -> tuple[Path, Path]:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_backup = BACKUP_DIR / f"xgboost_model_before_kaggle_tune_{timestamp}.pkl"
    workbook_backup = BACKUP_DIR / f"kaggle_model_training_report_before_kaggle_tune_{timestamp}.xlsx"
    shutil.copy2(MODEL_PATH, model_backup)
    shutil.copy2(WORKBOOK_PATH, workbook_backup)
    return model_backup, workbook_backup


def update_workbook_with_metrics(metrics: dict[str, float], split_name: str) -> None:
    model_summary_df = pd.read_excel(WORKBOOK_PATH, sheet_name="model_summary")
    mask = model_summary_df["model"].astype(str).str.lower() == "xgboost"
    model_summary_df.loc[mask, "holdout_top1_accuracy"] = float(metrics["top1_accuracy"])
    model_summary_df.loc[mask, "holdout_top3_accuracy"] = float(metrics["top3_accuracy"])
    model_summary_df.loc[mask, "holdout_macro_f1"] = float(metrics["macro_f1"])
    model_summary_df.loc[mask, "holdout_ece"] = float(metrics["ece"])

    with pd.ExcelWriter(WORKBOOK_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        model_summary_df.to_excel(writer, sheet_name="model_summary", index=False)

    workbook = load_workbook(WORKBOOK_PATH)
    worksheet = workbook["xgboost"]
    worksheet["A3"] = "XGBoost"
    worksheet["B3"] = split_name
    worksheet["C3"] = float(metrics["top1_accuracy"])
    worksheet["D3"] = float(metrics["top3_accuracy"])
    worksheet["E3"] = float(metrics["macro_f1"])
    worksheet["F3"] = float(metrics["ece"])
    workbook.save(WORKBOOK_PATH)


def promote_best_candidate(
    best_candidate: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_cv: pd.DataFrame,
    y_cv: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict[str, Any]:
    X_dev = pd.concat([X_train, X_cv], ignore_index=True)
    y_dev = np.concatenate([y_train, y_cv])

    final_model = build_model(
        num_classes=len(np.unique(y_dev)),
        candidate=best_candidate,
        n_estimators_override=int(best_candidate["best_trees"]),
    )
    fit_kwargs: dict[str, Any] = {"verbose": False}
    if best_candidate["use_sample_weight"]:
        fit_kwargs["sample_weight"] = compute_sample_weights(y_dev)

    fit_start = time.perf_counter()
    final_model.fit(X_dev, y_dev, **fit_kwargs)
    fit_seconds = time.perf_counter() - fit_start
    holdout_metrics = evaluate_model(final_model, X_test, y_test)

    model_backup, workbook_backup = backup_current_assets()
    joblib.dump(final_model, MODEL_PATH)
    update_workbook_with_metrics(holdout_metrics, split_name="kaggle_holdout_tuned")

    promotion_record = {
        "promoted_candidate_id": best_candidate["candidate_id"],
        "model_backup_path": str(model_backup),
        "workbook_backup_path": str(workbook_backup),
        "fit_seconds_dev": round(fit_seconds, 2),
        **holdout_metrics,
    }
    pd.DataFrame([promotion_record]).to_csv(PROMOTED_METRICS_PATH, index=False)
    return promotion_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Kaggle-only XGBoost against the Kaggle CV split and optionally promote the best candidate.")
    parser.add_argument(
        "--promote-best",
        action="store_true",
        help="After candidate screening, train the top-ranked candidate on X_train + X_cv and replace the main XGBoost artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X_train, X_cv, X_test, y_train, y_cv, y_test = load_prepared_data()
    base_model = joblib.load(MODEL_PATH)

    print("=" * 72)
    print("KAGGLE XGBOOST TUNING")
    print("=" * 72)
    print(f"Target: improve Kaggle-only holdout model at {MODEL_PATH}")
    print(f"Train rows: {len(X_train)}")
    print(f"CV rows: {len(X_cv)}")
    print(f"Test rows: {len(X_test)}")

    baseline_metrics = evaluate_model(base_model, X_cv, y_cv)
    print(
        "Baseline on Kaggle CV -> "
        f"Top-1={baseline_metrics['top1_accuracy']:.4f}, "
        f"Top-3={baseline_metrics['top3_accuracy']:.4f}, "
        f"Macro-F1={baseline_metrics['macro_f1']:.4f}, "
        f"ECE={baseline_metrics['ece']:.4f}"
    )

    rows: list[dict[str, Any]] = []
    for candidate in get_candidate_configs():
        print(f"Running {candidate['candidate_id']} ...")
        result = run_candidate(candidate, X_train, y_train, X_cv, y_cv)
        print(
            f"  cv -> Top-1={result['cv_top1_accuracy']:.4f}, "
            f"Top-3={result['cv_top3_accuracy']:.4f}, "
            f"Macro-F1={result['cv_macro_f1']:.4f}, "
            f"ECE={result['cv_ece']:.4f}, "
            f"best_trees={result['best_trees']}, "
            f"fit={result['fit_seconds']:.2f}s"
        )
        rows.append(result)

    results_df = rank_candidates(pd.DataFrame(rows))
    results_df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"Saved tuning results: {RESULTS_CSV_PATH}")
    print(results_df.to_string(index=False))

    if not args.promote_best:
        return

    best_candidate = results_df.iloc[0].to_dict()
    print(f"Promoting best candidate: {best_candidate['candidate_id']}")
    promotion_record = promote_best_candidate(best_candidate, X_train, y_train, X_cv, y_cv, X_test, y_test)
    print("Promotion complete.")
    print(json.dumps(promotion_record, indent=2))


if __name__ == "__main__":
    main()
