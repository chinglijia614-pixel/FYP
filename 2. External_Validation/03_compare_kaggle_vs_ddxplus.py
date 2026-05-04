from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder


RANDOM_STATE = 42
N_BINS_ECE = 10

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
KAGGLE_DIR = PROJECT_ROOT / "1. Kaggle_Training"
KAGGLE_INPUT_DIR = KAGGLE_DIR / "input"
KAGGLE_ARTIFACTS_DIR = KAGGLE_DIR / "Artifacts_kaggle"
OUTPUT_DIR = BASE_DIR / "output"

ALIGNMENT_REPORT_PATH = OUTPUT_DIR / "ddxplus_alignment_report.xlsx"
ALIGNMENT_REPORT_FALLBACK_PATH = OUTPUT_DIR / "ddxplus_alignment_report_latest.xlsx"
REPORT_WORKBOOK_PATH = OUTPUT_DIR / "objective1_ddxplus_single_vs_multidataset_report.xlsx"
SELECTED_MODEL_NAME = "XGBoost"
MAPPING_SCOPE = "exact+alias"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def build_confusion_matrix_df(y_true: np.ndarray, y_pred: np.ndarray, class_names: np.ndarray) -> pd.DataFrame:
    labels_used = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    used_names = [class_names[i] for i in labels_used]
    matrix = confusion_matrix(y_true, y_pred, labels=labels_used)
    return pd.DataFrame(matrix, index=used_names, columns=used_names).reset_index().rename(
        columns={"index": "true_class"}
    )


def build_classification_report_df(y_true: np.ndarray, y_pred: np.ndarray, class_names: np.ndarray) -> pd.DataFrame:
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
    return pd.DataFrame(rows)


def write_section(
    writer: pd.ExcelWriter,
    sheet_name: str,
    title: str,
    df: pd.DataFrame,
    startrow: int,
) -> int:
    pd.DataFrame({title: [title]}).to_excel(
        writer,
        sheet_name=sheet_name,
        startrow=startrow,
        index=False,
        header=False,
    )
    df.to_excel(writer, sheet_name=sheet_name, startrow=startrow + 1, index=False)
    return startrow + len(df) + 4


def load_kaggle_resources() -> tuple[pd.DataFrame, np.ndarray, list[str], LabelEncoder]:
    X_train = pd.read_csv(KAGGLE_INPUT_DIR / "X_train.csv")
    y_train = pd.read_csv(KAGGLE_INPUT_DIR / "y_train.csv")["target"].to_numpy()
    feature_names: list[str] = joblib.load(KAGGLE_ARTIFACTS_DIR / "feature_names.pkl")
    label_encoder: LabelEncoder = joblib.load(KAGGLE_ARTIFACTS_DIR / "label_encoder.pkl")
    X_train = X_train[feature_names]
    return X_train, y_train, feature_names, label_encoder


def resolve_alignment_report_path() -> Path:
    if ALIGNMENT_REPORT_FALLBACK_PATH.exists():
        if (not ALIGNMENT_REPORT_PATH.exists()) or (
            ALIGNMENT_REPORT_FALLBACK_PATH.stat().st_mtime > ALIGNMENT_REPORT_PATH.stat().st_mtime
        ):
            return ALIGNMENT_REPORT_FALLBACK_PATH
    return ALIGNMENT_REPORT_PATH


def load_ddxplus_alignment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alignment_report_path = resolve_alignment_report_path()
    if not alignment_report_path.exists():
        raise FileNotFoundError("Alignment outputs not found. Run align_ddxplus.py first.")

    try:
        row_alignment_df = pd.read_excel(alignment_report_path, sheet_name="row_alignment_internal")
    except ValueError:
        row_alignment_df = pd.read_excel(alignment_report_path, sheet_name="row_alignment")

    disease_mapping_df = pd.read_excel(alignment_report_path, sheet_name="disease_mapping")
    feature_mapping_df = pd.read_excel(alignment_report_path, sheet_name="feature_mapping")
    split_coverage_df = pd.read_excel(alignment_report_path, sheet_name="split_coverage")
    return row_alignment_df, disease_mapping_df, feature_mapping_df, split_coverage_df


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


def build_predictions_df(
    evaluation_name: str,
    meta_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: np.ndarray,
) -> pd.DataFrame:
    top3_idx = np.argsort(y_proba, axis=1)[:, -3:][:, ::-1]
    rows = []
    for idx, row in meta_test.reset_index(drop=True).iterrows():
        pred_idx = int(y_pred[idx])
        true_idx = int(y_true[idx])
        top3 = top3_idx[idx]
        rows.append(
            {
                "evaluation_name": evaluation_name,
                "row_id": row["row_id"],
                "raw_pathology": row["raw_pathology"],
                "mapped_disease": row["mapped_disease"],
                "predicted_disease": class_names[pred_idx],
                "top1_correct": int(pred_idx == true_idx),
                "top3_correct": int(true_idx in top3.tolist()),
                "top1_confidence": float(np.max(y_proba[idx])),
                "top3_pred_1": class_names[top3[0]],
                "top3_pred_2": class_names[top3[1]] if len(top3) > 1 else None,
                "top3_pred_3": class_names[top3[2]] if len(top3) > 2 else None,
                "mapped_feature_count": int(row["mapped_feature_count"]),
                "mapped_features": row["mapped_features"],
                "source_split": row["source_split"],
            }
        )
    return pd.DataFrame(rows)


def evaluate_regime(
    evaluation_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    meta_test: pd.DataFrame,
    class_names: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = build_xgboost_model(len(class_names))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = calculate_metrics(y_test, y_pred, y_proba)

    summary_df = pd.DataFrame(
        [
            {
                "evaluation_name": evaluation_name,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                **metrics,
            }
        ]
    )
    predictions_df = build_predictions_df(
        evaluation_name=evaluation_name,
        meta_test=meta_test,
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=class_names,
    )
    classification_report_df = build_classification_report_df(y_test, y_pred, class_names)
    classification_report_df.insert(0, "evaluation_name", evaluation_name)
    confusion_matrix_df = build_confusion_matrix_df(y_test, y_pred, class_names)
    confusion_matrix_df.insert(0, "evaluation_name", evaluation_name)
    return summary_df, predictions_df, classification_report_df, confusion_matrix_df


def build_shared_space() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder, pd.DataFrame, pd.DataFrame]:
    X_kaggle, y_kaggle, feature_names, kaggle_encoder = load_kaggle_resources()
    row_alignment_df, disease_mapping_df, feature_mapping_df, split_coverage_df = load_ddxplus_alignment()

    train_df = row_alignment_df.loc[row_alignment_df["source_split"].isin(["train", "validate"])].copy().reset_index(drop=True)
    train_df = train_df.drop_duplicates(subset=["mapped_disease", "mapped_features"], keep="first").reset_index(drop=True)
    test_df = row_alignment_df.loc[row_alignment_df["source_split"] == "test"].copy().reset_index(drop=True)

    shared_diseases = sorted(set(train_df["mapped_disease"]).intersection(set(test_df["mapped_disease"])))
    train_df = train_df.loc[train_df["mapped_disease"].isin(shared_diseases)].reset_index(drop=True)
    test_df = test_df.loc[test_df["mapped_disease"].isin(shared_diseases)].reset_index(drop=True)

    local_encoder = LabelEncoder()
    local_encoder.fit(shared_diseases)

    kaggle_disease_text = kaggle_encoder.inverse_transform(y_kaggle.astype(int))
    kaggle_df = X_kaggle.copy()
    kaggle_df["mapped_disease"] = kaggle_disease_text
    kaggle_df = kaggle_df.loc[kaggle_df["mapped_disease"].isin(shared_diseases)].reset_index(drop=True)

    def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
        matrix = pd.DataFrame(0, index=df.index, columns=feature_names, dtype=np.uint8)
        for idx, feature_text in enumerate(df["mapped_features"]):
            for feature in str(feature_text).split(" | "):
                if feature:
                    matrix.at[idx, feature] = 1
        return matrix

    X_ddx_train = build_feature_matrix(train_df)
    X_ddx_test = build_feature_matrix(test_df)
    y_ddx_train = local_encoder.transform(train_df["mapped_disease"].astype(str))
    y_ddx_test = local_encoder.transform(test_df["mapped_disease"].astype(str))

    X_kaggle_shared = kaggle_df[feature_names].reset_index(drop=True)
    y_kaggle_shared = local_encoder.transform(kaggle_df["mapped_disease"].astype(str))
    meta_test = test_df[
        ["source_split", "row_id", "raw_pathology", "mapped_disease", "mapped_feature_count", "mapped_features"]
    ].copy()

    kaggle_comp_df = (
        kaggle_df["mapped_disease"].value_counts().rename_axis("mapped_disease").reset_index(name="kaggle_train_rows")
    )
    ddx_train_comp_df = (
        train_df["mapped_disease"].value_counts().rename_axis("mapped_disease").reset_index(name="ddxplus_train_rows")
    )
    ddx_test_comp_df = (
        test_df["mapped_disease"].value_counts().rename_axis("mapped_disease").reset_index(name="ddxplus_test_rows")
    )
    disease_coverage_df = (
        pd.DataFrame({"mapped_disease": shared_diseases})
        .merge(kaggle_comp_df, on="mapped_disease", how="left")
        .merge(ddx_train_comp_df, on="mapped_disease", how="left")
        .merge(ddx_test_comp_df, on="mapped_disease", how="left")
        .fillna(0)
    )
    disease_coverage_df[["kaggle_train_rows", "ddxplus_train_rows", "ddxplus_test_rows"]] = disease_coverage_df[
        ["kaggle_train_rows", "ddxplus_train_rows", "ddxplus_test_rows"]
    ].astype(int)

    per_class_sample_size_df = disease_coverage_df.copy()
    per_class_sample_size_df["kaggle_only_train_rows"] = per_class_sample_size_df["kaggle_train_rows"]
    per_class_sample_size_df["kaggle_plus_ddxplus_train_rows"] = (
        per_class_sample_size_df["kaggle_train_rows"] + per_class_sample_size_df["ddxplus_train_rows"]
    )
    per_class_sample_size_df["external_ground_truth_test_rows"] = per_class_sample_size_df["ddxplus_test_rows"]
    per_class_sample_size_df["appears_in_external_test_ground_truth"] = (
        per_class_sample_size_df["ddxplus_test_rows"] > 0
    ).astype(int)
    per_class_sample_size_df = per_class_sample_size_df[
        [
            "mapped_disease",
            "kaggle_train_rows",
            "ddxplus_train_rows",
            "ddxplus_test_rows",
            "kaggle_only_train_rows",
            "kaggle_plus_ddxplus_train_rows",
            "external_ground_truth_test_rows",
            "appears_in_external_test_ground_truth",
        ]
    ].sort_values("mapped_disease").reset_index(drop=True)

    raw_rows_by_split = dict(zip(split_coverage_df["source_split"], split_coverage_df["raw_rows"]))
    mapped_rows_before_dedup_by_split = dict(
        zip(split_coverage_df["source_split"], split_coverage_df["mapped_rows_before_dedup"])
    )
    dedup_rows_by_split = dict(
        zip(split_coverage_df["source_split"], split_coverage_df["rows_after_dedup_within_split"])
    )

    validation_scope_df = pd.DataFrame(
        [
            {"metric": "validation_dataset", "value": "DDXPlus English"},
            {"metric": "selected_model_name", "value": SELECTED_MODEL_NAME},
            {"metric": "mapping_scope", "value": MAPPING_SCOPE},
            {"metric": "raw_disease_classes", "value": len(disease_mapping_df)},
            {"metric": "mapped_disease_classes", "value": int(disease_mapping_df["mapped_disease"].dropna().nunique())},
            {"metric": "shared_disease_classes_used_for_training", "value": len(shared_diseases)},
            {"metric": "kaggle_train_rows_in_shared_space", "value": len(X_kaggle_shared)},
            {"metric": "ddxplus_train_rows_raw", "value": int(raw_rows_by_split.get("train", 0))},
            {"metric": "ddxplus_validate_rows_raw", "value": int(raw_rows_by_split.get("validate", 0))},
            {"metric": "ddxplus_test_rows_raw", "value": int(raw_rows_by_split.get("test", 0))},
            {"metric": "ddxplus_train_rows_mapped_before_dedup", "value": int(mapped_rows_before_dedup_by_split.get("train", 0))},
            {"metric": "ddxplus_validate_rows_mapped_before_dedup", "value": int(mapped_rows_before_dedup_by_split.get("validate", 0))},
            {"metric": "ddxplus_test_rows_mapped_before_dedup", "value": int(mapped_rows_before_dedup_by_split.get("test", 0))},
            {"metric": "ddxplus_train_rows_dedup", "value": int(dedup_rows_by_split.get("train", 0))},
            {"metric": "ddxplus_validate_rows_dedup", "value": int(dedup_rows_by_split.get("validate", 0))},
            {"metric": "ddxplus_test_rows_dedup", "value": int(dedup_rows_by_split.get("test", 0))},
            {"metric": "ddxplus_train_plus_validate_rows_used", "value": len(X_ddx_train)},
            {"metric": "ddxplus_test_rows_used", "value": len(X_ddx_test)},
            {"metric": "aligned_feature_space_size", "value": len(feature_names)},
            {"metric": "aligned_label_space_size", "value": len(kaggle_encoder.classes_)},
        ]
    )

    dataset_summary_df = pd.DataFrame(
        [
            {"metric": "mapping_scope", "value": MAPPING_SCOPE},
            {"metric": "shared_disease_count", "value": len(shared_diseases)},
            {"metric": "kaggle_train_rows_shared", "value": len(X_kaggle_shared)},
            {"metric": "ddxplus_train_rows_shared", "value": len(X_ddx_train)},
            {"metric": "ddxplus_test_rows_shared", "value": len(X_ddx_test)},
            {"metric": "kaggle_feature_count", "value": len(feature_names)},
        ]
    )

    shared_bundle_df = pd.DataFrame(
        {
            "X_kaggle_shared": [X_kaggle_shared],
            "y_kaggle_shared": [y_kaggle_shared],
            "X_ddx_train": [X_ddx_train],
            "y_ddx_train": [y_ddx_train],
            "X_ddx_test": [X_ddx_test],
            "y_ddx_test": [y_ddx_test],
            "meta_test": [meta_test.reset_index(drop=True)],
        }
    )

    return (
        shared_bundle_df,
        dataset_summary_df,
        validation_scope_df,
        per_class_sample_size_df,
        disease_coverage_df,
        local_encoder,
        disease_mapping_df,
        feature_mapping_df,
    )


def main() -> None:
    print("--- Objective 1: Kaggle vs DDXPlus External Comparison ---")
    script_start = time.perf_counter()

    try:
        section_start = time.perf_counter()
        try:
            (
                shared_bundle_df,
                dataset_summary_df,
                validation_scope_df,
                per_class_sample_size_df,
                disease_coverage_df,
                local_encoder,
                disease_mapping_df,
                feature_mapping_df,
            ) = build_shared_space()

            shared_bundle = shared_bundle_df.iloc[0]
            X_kaggle_shared = shared_bundle["X_kaggle_shared"]
            y_kaggle_shared = shared_bundle["y_kaggle_shared"]
            X_ddx_train = shared_bundle["X_ddx_train"]
            y_ddx_train = shared_bundle["y_ddx_train"]
            X_ddx_test = shared_bundle["X_ddx_test"]
            y_ddx_test = shared_bundle["y_ddx_test"]
            meta_test = shared_bundle["meta_test"]

            class_names = local_encoder.classes_
            print(f"Selected algorithm: {SELECTED_MODEL_NAME}")
            print(f"Mapping scope: {MAPPING_SCOPE}")
            print(f"Shared diseases: {len(class_names)}")
            print(f"Kaggle train rows in shared space: {len(X_kaggle_shared)}")
            print(f"DDXPlus train rows: {len(X_ddx_train)}")
            print(f"DDXPlus test rows: {len(X_ddx_test)}")
        finally:
            timer("build_shared_space", section_start)

        section_start = time.perf_counter()
        try:
            kaggle_only_summary_df, kaggle_only_predictions_df, kaggle_only_report_df, kaggle_only_cm_df = evaluate_regime(
                evaluation_name="kaggle_only",
                X_train=X_kaggle_shared,
                y_train=y_kaggle_shared,
                X_test=X_ddx_test,
                y_test=y_ddx_test,
                meta_test=meta_test,
                class_names=class_names,
            )

            X_multi = pd.concat([X_kaggle_shared, X_ddx_train], ignore_index=True)
            y_multi = np.concatenate([y_kaggle_shared, y_ddx_train])
            multi_summary_df, multi_predictions_df, multi_report_df, multi_cm_df = evaluate_regime(
                evaluation_name="kaggle_plus_ddxplus",
                X_train=X_multi,
                y_train=y_multi,
                X_test=X_ddx_test,
                y_test=y_ddx_test,
                meta_test=meta_test,
                class_names=class_names,
            )

            comparison_df = pd.concat([kaggle_only_summary_df, multi_summary_df], ignore_index=True)
            comparison_df["top1_gain_vs_kaggle_only"] = comparison_df["top1_accuracy"] - float(
                comparison_df.loc[comparison_df["evaluation_name"] == "kaggle_only", "top1_accuracy"].iloc[0]
            )
            comparison_df["top3_gain_vs_kaggle_only"] = comparison_df["top3_accuracy"] - float(
                comparison_df.loc[comparison_df["evaluation_name"] == "kaggle_only", "top3_accuracy"].iloc[0]
            )
            comparison_df["macro_f1_gain_vs_kaggle_only"] = comparison_df["macro_f1"] - float(
                comparison_df.loc[comparison_df["evaluation_name"] == "kaggle_only", "macro_f1"].iloc[0]
            )
        finally:
            timer("train_and_compare", section_start)

        section_start = time.perf_counter()
        try:
            predictions_df = pd.concat([kaggle_only_predictions_df, multi_predictions_df], ignore_index=True)
            classification_report_df = pd.concat([kaggle_only_report_df, multi_report_df], ignore_index=True)
            confusion_matrix_df = pd.concat([kaggle_only_cm_df, multi_cm_df], ignore_index=True)

            with pd.ExcelWriter(REPORT_WORKBOOK_PATH, engine="openpyxl") as writer:
                current_row = 0
                current_row = write_section(writer, "summary", "dataset_scope", dataset_summary_df, current_row)
                write_section(writer, "summary", "comparison_metrics", comparison_df, current_row)
                validation_scope_df.to_excel(writer, sheet_name="validation_scope", index=False)
                per_class_sample_size_df.to_excel(writer, sheet_name="per_class_sample_size", index=False)
                disease_coverage_df.to_excel(writer, sheet_name="shared_disease_coverage", index=False)
                disease_mapping_df.to_excel(writer, sheet_name="disease_mapping", index=False)
                feature_mapping_df.to_excel(writer, sheet_name="feature_mapping", index=False)
                predictions_df.to_excel(writer, sheet_name="predictions", index=False)
                classification_report_df.to_excel(writer, sheet_name="classification_report", index=False)
                confusion_matrix_df.to_excel(writer, sheet_name="confusion_matrix", index=False)

            print("=" * 72)
            print("DDXPLUS EXTERNAL COMPARISON COMPLETE")
            print("=" * 72)
            kaggle_only_row = comparison_df.loc[comparison_df["evaluation_name"] == "kaggle_only"].iloc[0]
            multi_row = comparison_df.loc[comparison_df["evaluation_name"] == "kaggle_plus_ddxplus"].iloc[0]
            print(f"Kaggle-only Top-1: {float(kaggle_only_row['top1_accuracy']):.4f}")
            print(f"Kaggle+DDXPlus Top-1: {float(multi_row['top1_accuracy']):.4f}")
            print(f"Top-1 improvement: {float(multi_row['top1_accuracy'] - kaggle_only_row['top1_accuracy']):.4f}")
            print(f"Kaggle-only Macro-F1: {float(kaggle_only_row['macro_f1']):.4f}")
            print(f"Kaggle+DDXPlus Macro-F1: {float(multi_row['macro_f1']):.4f}")
            print(f"Workbook saved: {REPORT_WORKBOOK_PATH.name}")
        finally:
            timer("save_outputs", section_start)
    finally:
        timer("total_script", script_start)


if __name__ == "__main__":
    main()
