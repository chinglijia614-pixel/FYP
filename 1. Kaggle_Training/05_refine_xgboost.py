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
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
N_BINS_ECE = 10
DEFAULT_REFINE_MAX_ROWS = 60000
DEFAULT_REFINE_EVAL_FRACTION = 0.20
DEFAULT_MAX_SAMPLE_WEIGHT = 4.0

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_INPUT_DIR = PROJECT_ROOT / "input"
ARTIFACTS_DIR = PROJECT_ROOT / "Artifacts_kaggle"
OUTPUT_DIR = PROJECT_ROOT / "output"
BACKUP_DIR = ARTIFACTS_DIR / "backups"
MODEL_PATH = ARTIFACTS_DIR / "xgboost_model.pkl"
WORKBOOK_PATH = OUTPUT_DIR / "kaggle_model_training_report.xlsx"
RESULTS_CSV_PATH = OUTPUT_DIR / "xgboost_refinement_candidates.csv"
PROMOTED_METRICS_PATH = OUTPUT_DIR / "xgboost_refinement_promoted_metrics.csv"


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


def load_prepared_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    feature_names: list[str] = joblib.load(ARTIFACTS_DIR / "feature_names.pkl")
    X_cv = pd.read_csv(MODEL_INPUT_DIR / "X_cv.csv")[feature_names]
    X_test = pd.read_csv(MODEL_INPUT_DIR / "X_test.csv")[feature_names]
    y_cv = pd.read_csv(MODEL_INPUT_DIR / "y_cv.csv")["target"].to_numpy()
    y_test = pd.read_csv(MODEL_INPUT_DIR / "y_test.csv")["target"].to_numpy()
    return X_cv, X_test, y_cv, y_test, feature_names


def get_candidate_configs() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": "c01_unweighted_fast",
            "use_sample_weight": False,
            "n_estimators": 10,
            "max_depth": 5,
            "learning_rate": 0.03,
            "min_child_weight": 2.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        {
            "candidate_id": "c02_unweighted_regularized",
            "use_sample_weight": False,
            "n_estimators": 15,
            "max_depth": 6,
            "learning_rate": 0.02,
            "min_child_weight": 2.0,
            "reg_alpha": 0.05,
            "reg_lambda": 1.1,
        },
        {
            "candidate_id": "c03_weighted_fast",
            "use_sample_weight": True,
            "n_estimators": 10,
            "max_depth": 5,
            "learning_rate": 0.03,
            "min_child_weight": 2.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        {
            "candidate_id": "c04_weighted_regularized",
            "use_sample_weight": True,
            "n_estimators": 15,
            "max_depth": 6,
            "learning_rate": 0.02,
            "min_child_weight": 2.0,
            "reg_alpha": 0.05,
            "reg_lambda": 1.2,
        },
    ]


def sample_refine_pool(
    X_cv: pd.DataFrame,
    y_cv: np.ndarray,
    refine_max_rows: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(X_cv) <= refine_max_rows:
        return X_cv.reset_index(drop=True), y_cv.copy()

    y_series = pd.Series(y_cv, name="target")
    class_to_seed_index = y_series.groupby(y_series).head(1).index.to_numpy()
    seeded_X = X_cv.loc[class_to_seed_index].copy()
    seeded_y = y_cv[class_to_seed_index]

    if len(seeded_X) > refine_max_rows:
        raise ValueError(
            f"refine_max_rows={refine_max_rows} is smaller than the number of classes ({len(seeded_X)}). "
            "Increase refine_max_rows so every class can be represented in the refinement pool."
        )

    remaining_slots = refine_max_rows - len(seeded_X)
    if remaining_slots == 0:
        return seeded_X.reset_index(drop=True), seeded_y.copy()

    remaining_mask = np.ones(len(X_cv), dtype=bool)
    remaining_mask[class_to_seed_index] = False
    X_remaining = X_cv.loc[remaining_mask].reset_index(drop=True)
    y_remaining = y_cv[remaining_mask]

    remaining_counts = pd.Series(y_remaining).value_counts()
    remaining_singleton_classes = set(remaining_counts[remaining_counts < 2].index.tolist())
    if remaining_singleton_classes:
        singleton_mask = pd.Series(y_remaining).isin(remaining_singleton_classes).to_numpy()
        X_forced = X_remaining.loc[singleton_mask].reset_index(drop=True)
        y_forced = y_remaining[singleton_mask]
        X_eligible = X_remaining.loc[~singleton_mask].reset_index(drop=True)
        y_eligible = y_remaining[~singleton_mask]
    else:
        X_forced = X_remaining.iloc[0:0].copy()
        y_forced = np.array([], dtype=y_remaining.dtype)
        X_eligible = X_remaining
        y_eligible = y_remaining

    if len(X_forced) > remaining_slots:
        X_forced = X_forced.iloc[:remaining_slots].reset_index(drop=True)
        y_forced = y_forced[:remaining_slots]
        X_sampled = X_eligible.iloc[0:0].copy()
        y_sampled = np.array([], dtype=y_eligible.dtype)
    else:
        remaining_slots_after_forced = remaining_slots - len(X_forced)
        if remaining_slots_after_forced == 0:
            X_sampled = X_eligible.iloc[0:0].copy()
            y_sampled = np.array([], dtype=y_eligible.dtype)
        else:
            X_sampled, _, y_sampled, _ = train_test_split(
                X_eligible,
                y_eligible,
                train_size=remaining_slots_after_forced,
                stratify=y_eligible,
                random_state=RANDOM_STATE,
            )

    X_subset = pd.concat(
        [seeded_X.reset_index(drop=True), X_forced.reset_index(drop=True), X_sampled.reset_index(drop=True)],
        ignore_index=True,
    )
    y_subset = np.concatenate([seeded_y, y_forced, y_sampled])
    return X_subset.reset_index(drop=True), y_subset


def split_refine_pool(
    X_pool: pd.DataFrame,
    y_pool: np.ndarray,
    refine_eval_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    class_counts = pd.Series(y_pool).value_counts()
    singleton_classes = set(class_counts[class_counts < 2].index.tolist())

    if singleton_classes:
        singleton_mask = pd.Series(y_pool).isin(singleton_classes).to_numpy()
        X_singletons = X_pool.loc[singleton_mask].reset_index(drop=True)
        y_singletons = y_pool[singleton_mask]
        X_split_pool = X_pool.loc[~singleton_mask].reset_index(drop=True)
        y_split_pool = y_pool[~singleton_mask]
    else:
        X_singletons = X_pool.iloc[0:0].copy()
        y_singletons = np.array([], dtype=y_pool.dtype)
        X_split_pool = X_pool.reset_index(drop=True)
        y_split_pool = y_pool

    X_train_base, X_eval_refine, y_train_base, y_eval_refine = train_test_split(
        X_split_pool,
        y_split_pool,
        test_size=refine_eval_fraction,
        stratify=y_split_pool,
        random_state=RANDOM_STATE,
    )

    if len(X_singletons) == 0:
        return X_train_base, X_eval_refine, y_train_base, y_eval_refine

    X_train_refine = pd.concat([X_train_base.reset_index(drop=True), X_singletons], ignore_index=True)
    y_train_refine = np.concatenate([y_train_base, y_singletons])
    return X_train_refine, X_eval_refine.reset_index(drop=True), y_train_refine, y_eval_refine


def build_refinement_model(num_classes: int, candidate: dict[str, Any]) -> Any:
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        n_estimators=candidate["n_estimators"],
        max_depth=candidate["max_depth"],
        learning_rate=candidate["learning_rate"],
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=candidate["min_child_weight"],
        reg_alpha=candidate["reg_alpha"],
        reg_lambda=candidate["reg_lambda"],
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def evaluate_model(model: Any, X_eval: pd.DataFrame, y_eval: np.ndarray) -> dict[str, float]:
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)
    return calculate_metrics(y_eval, y_pred, y_proba)


def run_candidate(
    base_model: Any,
    candidate: dict[str, Any],
    X_train_refine: pd.DataFrame,
    y_train_refine: np.ndarray,
    X_eval_refine: pd.DataFrame,
    y_eval_refine: np.ndarray,
) -> dict[str, Any]:
    model = build_refinement_model(len(np.unique(y_train_refine)), candidate)
    fit_kwargs: dict[str, Any] = {"xgb_model": base_model.get_booster(), "verbose": False}
    if candidate["use_sample_weight"]:
        fit_kwargs["sample_weight"] = compute_sample_weights(y_train_refine)

    fit_start = time.perf_counter()
    model.fit(X_train_refine, y_train_refine, **fit_kwargs)
    fit_seconds = time.perf_counter() - fit_start
    metrics = evaluate_model(model, X_eval_refine, y_eval_refine)

    return {
        **candidate,
        "refine_train_rows": int(len(X_train_refine)),
        "refine_eval_rows": int(len(X_eval_refine)),
        "fit_seconds": round(fit_seconds, 2),
        **{f"refine_eval_{key}": value for key, value in metrics.items()},
    }


def rank_candidates(results_df: pd.DataFrame) -> pd.DataFrame:
    return results_df.sort_values(
        by=[
            "refine_eval_top1_accuracy",
            "refine_eval_macro_f1",
            "refine_eval_top3_accuracy",
            "refine_eval_ece",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def backup_current_assets() -> tuple[Path, Path]:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_backup = BACKUP_DIR / f"xgboost_model_before_promotion_{timestamp}.pkl"
    workbook_backup = BACKUP_DIR / f"kaggle_model_training_report_before_promotion_{timestamp}.xlsx"
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
    base_model: Any,
    best_candidate: dict[str, Any],
    X_cv: pd.DataFrame,
    y_cv: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict[str, Any]:
    promotion_model = build_refinement_model(len(np.unique(y_cv)), best_candidate)
    fit_kwargs: dict[str, Any] = {"xgb_model": base_model.get_booster(), "verbose": False}
    if best_candidate["use_sample_weight"]:
        fit_kwargs["sample_weight"] = compute_sample_weights(y_cv)

    fit_start = time.perf_counter()
    promotion_model.fit(X_cv, y_cv, **fit_kwargs)
    fit_seconds = time.perf_counter() - fit_start
    holdout_metrics = evaluate_model(promotion_model, X_test, y_test)

    model_backup, workbook_backup = backup_current_assets()
    joblib.dump(promotion_model, MODEL_PATH)
    update_workbook_with_metrics(holdout_metrics, split_name="kaggle_holdout_refined")

    promotion_record = {
        "promoted_candidate_id": best_candidate["candidate_id"],
        "model_backup_path": str(model_backup),
        "workbook_backup_path": str(workbook_backup),
        "fit_seconds_full_cv": round(fit_seconds, 2),
        **holdout_metrics,
    }
    pd.DataFrame([promotion_record]).to_csv(PROMOTED_METRICS_PATH, index=False)
    return promotion_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen and optionally promote refined XGBoost continuation candidates.")
    parser.add_argument(
        "--refine-max-rows",
        type=int,
        default=DEFAULT_REFINE_MAX_ROWS,
        help="Maximum X_cv rows to use during candidate screening. Default: 60000.",
    )
    parser.add_argument(
        "--refine-eval-fraction",
        type=float,
        default=DEFAULT_REFINE_EVAL_FRACTION,
        help="Fraction of the refine pool reserved for candidate ranking. Default: 0.20.",
    )
    parser.add_argument(
        "--promote-best",
        action="store_true",
        help="After screening, retrain the top-ranked candidate on full X_cv and promote it to xgboost_model.pkl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X_cv, X_test, y_cv, y_test, _ = load_prepared_data()
    base_model = joblib.load(MODEL_PATH)

    print("=" * 72)
    print("XGBOOST REFINEMENT SCREEN")
    print("=" * 72)
    print(f"Base model path: {MODEL_PATH}")
    print(f"Screen refine_max_rows: {args.refine_max_rows}")
    print(f"Screen refine_eval_fraction: {args.refine_eval_fraction:.2f}")

    X_pool, y_pool = sample_refine_pool(X_cv, y_cv, args.refine_max_rows)
    pool_class_counts = pd.Series(y_pool).value_counts()
    singleton_count = int((pool_class_counts < 2).sum())
    X_train_refine, X_eval_refine, y_train_refine, y_eval_refine = split_refine_pool(
        X_pool, y_pool, args.refine_eval_fraction
    )
    print(f"Refine pool rows: {len(X_pool)}")
    print(f"Refine pool singleton classes moved to train-only: {singleton_count}")
    print(f"Refine train rows: {len(X_train_refine)}")
    print(f"Refine eval rows: {len(X_eval_refine)}")

    baseline_metrics = evaluate_model(base_model, X_eval_refine, y_eval_refine)
    print(
        "Baseline on refine eval -> "
        f"Top-1={baseline_metrics['top1_accuracy']:.4f}, "
        f"Top-3={baseline_metrics['top3_accuracy']:.4f}, "
        f"Macro-F1={baseline_metrics['macro_f1']:.4f}, "
        f"ECE={baseline_metrics['ece']:.4f}"
    )

    rows: list[dict[str, Any]] = []
    for candidate in get_candidate_configs():
        print(f"Running {candidate['candidate_id']} ...")
        result = run_candidate(
            base_model=base_model,
            candidate=candidate,
            X_train_refine=X_train_refine,
            y_train_refine=y_train_refine,
            X_eval_refine=X_eval_refine,
            y_eval_refine=y_eval_refine,
        )
        print(
            f"  refine eval -> Top-1={result['refine_eval_top1_accuracy']:.4f}, "
            f"Top-3={result['refine_eval_top3_accuracy']:.4f}, "
            f"Macro-F1={result['refine_eval_macro_f1']:.4f}, "
            f"ECE={result['refine_eval_ece']:.4f}, "
            f"fit={result['fit_seconds']:.2f}s"
        )
        rows.append(result)

    results_df = rank_candidates(pd.DataFrame(rows))
    results_df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"Saved screening results: {RESULTS_CSV_PATH}")
    print(results_df.to_string(index=False))

    if not args.promote_best:
        return

    best_candidate = results_df.iloc[0].to_dict()
    print(f"Promoting best candidate: {best_candidate['candidate_id']}")
    promotion_record = promote_best_candidate(
        base_model=base_model,
        best_candidate=best_candidate,
        X_cv=X_cv,
        y_cv=y_cv,
        X_test=X_test,
        y_test=y_test,
    )
    print("Promotion complete.")
    print(json.dumps(promotion_record, indent=2))


if __name__ == "__main__":
    main()
