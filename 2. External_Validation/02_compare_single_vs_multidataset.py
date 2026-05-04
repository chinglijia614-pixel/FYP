from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# =============================================================================
# CONFIG
# =============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_BINS_ECE = 10

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
KAGGLE_DIR = PROJECT_ROOT / "1. Kaggle_Training"
KAGGLE_INPUT_DIR = KAGGLE_DIR / "input"
KAGGLE_ARTIFACTS_DIR = KAGGLE_DIR / "Artifacts_kaggle"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

ALIGNMENT_REPORT_PATH = OUTPUT_DIR / "symbipredict_alignment_report.xlsx"
ALIGNMENT_REPORT_FALLBACK_PATH = OUTPUT_DIR / "symbipredict_alignment_report_latest.xlsx"
SELECTED_MODEL_NAME = "XGBoost"

REPORT_WORKBOOK_PATH = OUTPUT_DIR / "objective1_single_vs_multidataset_report.xlsx"

MODEL_INPUT_FILES = {
    "X_train": KAGGLE_INPUT_DIR / "X_train.csv",
    "y_train": KAGGLE_INPUT_DIR / "y_train.csv",
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# UTILS
# =============================================================================
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


def build_confusion_matrix_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: np.ndarray,
) -> pd.DataFrame:
    labels_used = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    used_names = [class_names[i] for i in labels_used]
    matrix = confusion_matrix(y_true, y_pred, labels=labels_used)
    return pd.DataFrame(matrix, index=used_names, columns=used_names).reset_index().rename(
        columns={"index": "true_class"}
    )


def build_classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: np.ndarray,
) -> pd.DataFrame:
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


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("&", " and ")
    text = re.sub(r"[()/]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_retained_raw_feature_name(raw_column: str) -> str:
    normalized = normalize_text(raw_column).replace(" ", "_")
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return f"symbipredict_only__{normalized}"


def split_pipe_items(value: object) -> list[str]:
    return [x.strip() for x in str(value).split("|") if x.strip()]


def resolve_alignment_report_path() -> Path:
    if ALIGNMENT_REPORT_FALLBACK_PATH.exists():
        if (not ALIGNMENT_REPORT_PATH.exists()) or (
            ALIGNMENT_REPORT_FALLBACK_PATH.stat().st_mtime > ALIGNMENT_REPORT_PATH.stat().st_mtime
        ):
            return ALIGNMENT_REPORT_FALLBACK_PATH
    return ALIGNMENT_REPORT_PATH


def load_kaggle_resources() -> tuple[pd.DataFrame, np.ndarray, list[str], LabelEncoder]:
    X_train = pd.read_csv(MODEL_INPUT_FILES["X_train"])
    y_train = pd.read_csv(MODEL_INPUT_FILES["y_train"])["target"].to_numpy()
    feature_names: list[str] = joblib.load(KAGGLE_ARTIFACTS_DIR / "feature_names.pkl")
    label_encoder: LabelEncoder = joblib.load(KAGGLE_ARTIFACTS_DIR / "label_encoder.pkl")

    X_train = X_train[feature_names]
    return X_train, y_train, feature_names, label_encoder


def load_symbipredict_alignment() -> pd.DataFrame:
    alignment_report_path = resolve_alignment_report_path()
    if not alignment_report_path.exists():
        raise FileNotFoundError(
            "Alignment outputs not found. Run align_symbipredict_2022.py first."
        )

    try:
        row_alignment_df = pd.read_excel(alignment_report_path, sheet_name="row_alignment_internal")
    except ValueError:
        row_alignment_df = pd.read_excel(alignment_report_path, sheet_name="row_alignment")

    row_alignment_df["mapping_status"] = row_alignment_df["disease_mapping_status"]
    row_alignment_df["aligned_index"] = np.arange(len(row_alignment_df))
    return row_alignment_df


def build_validation_scope_df(
    row_alignment_df: pd.DataFrame,
    shared_diseases: list[str],
    X_kaggle_shared: pd.DataFrame,
    X_sym_train: pd.DataFrame,
    X_sym_test: pd.DataFrame,
    y_sym_text_train: pd.Series,
    y_sym_text_test: pd.Series,
    feature_names: list[str],
    symbipredict_only_features: list[str],
    union_feature_names: list[str],
) -> pd.DataFrame:
    mapped_mask = row_alignment_df["mapping_status"].isin({"exact_match", "alias_map"})
    retained_mask = row_alignment_df["retained_for_alignment"] == 1
    train_ground_truth_classes = sorted(pd.Series(y_sym_text_train).dropna().astype(str).unique().tolist())
    test_ground_truth_classes = sorted(pd.Series(y_sym_text_test).dropna().astype(str).unique().tolist())

    return pd.DataFrame(
        [
            {"metric": "validation_dataset", "value": "SymbiPredict 2022"},
            {"metric": "selected_model_name", "value": SELECTED_MODEL_NAME},
            {"metric": "mapping_scope", "value": "exact+alias"},
            {"metric": "raw_external_rows", "value": len(row_alignment_df)},
            {"metric": "retained_external_rows_before_mapping_scope", "value": int(retained_mask.sum())},
            {"metric": "mapped_external_rows_exact_or_alias", "value": int(mapped_mask.sum())},
            {"metric": "dropped_external_rows", "value": int(len(row_alignment_df) - mapped_mask.sum())},
            {"metric": "raw_external_disease_classes", "value": int(row_alignment_df["raw_prognosis"].nunique())},
            {"metric": "mapped_external_disease_classes", "value": int(row_alignment_df.loc[mapped_mask, "mapped_disease"].dropna().nunique())},
            {"metric": "shared_disease_classes_used_for_training", "value": len(shared_diseases)},
            {"metric": "external_ground_truth_classes_in_train_split", "value": len(train_ground_truth_classes)},
            {"metric": "external_ground_truth_classes_in_test_split", "value": len(test_ground_truth_classes)},
            {"metric": "kaggle_train_rows_in_shared_space", "value": len(X_kaggle_shared)},
            {"metric": "symbipredict_train_rows_in_shared_space", "value": len(X_sym_train)},
            {"metric": "symbipredict_test_rows_in_shared_space", "value": len(X_sym_test)},
            {"metric": "kaggle_feature_count", "value": len(feature_names)},
            {"metric": "symbipredict_only_feature_count", "value": len(symbipredict_only_features)},
            {"metric": "union_feature_count", "value": len(union_feature_names)},
        ]
    )


def build_per_class_sample_size_df(disease_coverage_df: pd.DataFrame) -> pd.DataFrame:
    df = disease_coverage_df.copy()
    count_columns = ["kaggle_train_rows", "symbipredict_train_rows", "symbipredict_test_rows"]
    df[count_columns] = df[count_columns].fillna(0).astype(int)
    df["kaggle_only_train_rows"] = df["kaggle_train_rows"]
    df["kaggle_plus_symbipredict_train_rows"] = df["kaggle_train_rows"] + df["symbipredict_train_rows"]
    df["external_ground_truth_test_rows"] = df["symbipredict_test_rows"]
    df["appears_in_external_test_ground_truth"] = (df["symbipredict_test_rows"] > 0).astype(int)
    ordered_columns = [
        "mapped_disease",
        "kaggle_train_rows",
        "symbipredict_train_rows",
        "symbipredict_test_rows",
        "kaggle_only_train_rows",
        "kaggle_plus_symbipredict_train_rows",
        "external_ground_truth_test_rows",
        "appears_in_external_test_ground_truth",
    ]
    return df[ordered_columns].sort_values("mapped_disease").reset_index(drop=True)


def load_symbipredict_alignment_retained() -> pd.DataFrame:
    row_alignment_df = load_symbipredict_alignment()
    retained_df = row_alignment_df.loc[row_alignment_df["retained_for_alignment"] == 1].reset_index(drop=True)
    return retained_df


# =============================================================================
# DATA PREP
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


def build_shared_space(
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder, pd.DataFrame]:
    X_kaggle, y_kaggle, feature_names, kaggle_encoder = load_kaggle_resources()
    row_alignment_df = load_symbipredict_alignment()
    retained_df = row_alignment_df.loc[row_alignment_df["retained_for_alignment"] == 1].reset_index(drop=True)

    retained_filtered = retained_df.loc[retained_df["mapping_status"].isin({"exact_match", "alias_map"})].copy().reset_index(drop=True)

    shared_diseases = sorted(retained_filtered["mapped_disease"].dropna().unique().tolist())
    if not shared_diseases:
        raise ValueError("No shared diseases remain after applying the chosen mapping scope.")

    symbipredict_only_features = sorted(
        {
            build_retained_raw_feature_name(raw_feature)
            for raw_text in retained_filtered["unmapped_active_features"].tolist()
            for raw_feature in split_pipe_items(raw_text)
        }
    )
    union_feature_names = feature_names + symbipredict_only_features

    union_active_feature_rows: list[list[str]] = []
    for _, row in retained_filtered.iterrows():
        active_features = set(split_pipe_items(row["mapped_features"]))
        active_features.update(
            build_retained_raw_feature_name(raw_feature)
            for raw_feature in split_pipe_items(row["unmapped_active_features"])
        )
        union_active_feature_rows.append(sorted(active_features))

    retained_filtered["union_active_features"] = [" | ".join(items) for items in union_active_feature_rows]

    # SymbiPredict contains many repeated disease + symptom rows.
    # Deduplicate on the full union feature signature before the train/test split.
    retained_filtered = retained_filtered.loc[
        ~retained_filtered.duplicated(subset=["mapped_disease", "union_active_features"], keep="first")
    ].copy().reset_index(drop=True)

    X_sym_union = pd.DataFrame(0, index=retained_filtered.index, columns=union_feature_names, dtype=np.uint8)
    for idx, features in enumerate(retained_filtered["union_active_features"]):
        for feat in split_pipe_items(features):
            if feat in X_sym_union.columns:
                X_sym_union.at[idx, feat] = 1

    kaggle_disease_text = kaggle_encoder.inverse_transform(y_kaggle.astype(int))
    kaggle_df = X_kaggle.copy()
    kaggle_df["mapped_disease"] = kaggle_disease_text
    kaggle_df = kaggle_df.loc[kaggle_df["mapped_disease"].isin(shared_diseases)].reset_index(drop=True)
    for sym_feature in symbipredict_only_features:
        kaggle_df[sym_feature] = 0

    retained_filtered["mapped_disease"] = retained_filtered["mapped_disease"].astype(str)

    local_encoder = LabelEncoder()
    local_encoder.fit(shared_diseases)

    X_sym_train, X_sym_test, y_sym_text_train, y_sym_text_test, meta_train, meta_test = train_test_split(
        X_sym_union,
        retained_filtered["mapped_disease"],
        retained_filtered,
        test_size=TEST_SIZE,
        stratify=retained_filtered["mapped_disease"],
        random_state=RANDOM_STATE,
    )

    kaggle_comp_df = (
        kaggle_df["mapped_disease"].value_counts()
        .rename_axis("mapped_disease")
        .reset_index(name="kaggle_train_rows")
    )
    sym_train_comp_df = (
        y_sym_text_train.value_counts()
        .rename_axis("mapped_disease")
        .reset_index(name="symbipredict_train_rows")
    )
    sym_test_comp_df = (
        y_sym_text_test.value_counts()
        .rename_axis("mapped_disease")
        .reset_index(name="symbipredict_test_rows")
    )
    disease_coverage_df = (
        pd.DataFrame({"mapped_disease": shared_diseases})
        .merge(kaggle_comp_df, on="mapped_disease", how="left")
        .merge(sym_train_comp_df, on="mapped_disease", how="left")
        .merge(sym_test_comp_df, on="mapped_disease", how="left")
        .fillna(0)
    )
    disease_coverage_df[["kaggle_train_rows", "symbipredict_train_rows", "symbipredict_test_rows"]] = disease_coverage_df[
        ["kaggle_train_rows", "symbipredict_train_rows", "symbipredict_test_rows"]
    ].astype(int)

    X_kaggle_shared = kaggle_df[union_feature_names].reset_index(drop=True)
    y_kaggle_shared = local_encoder.transform(kaggle_df["mapped_disease"])

    X_sym_train = X_sym_train[union_feature_names].reset_index(drop=True)
    X_sym_test = X_sym_test[union_feature_names].reset_index(drop=True)
    y_sym_train = local_encoder.transform(y_sym_text_train.astype(str))
    y_sym_test = local_encoder.transform(y_sym_text_test.astype(str))

    dataset_summary_df = pd.DataFrame(
        [
            {"metric": "mapping_scope", "value": "exact+alias"},
            {"metric": "shared_disease_count", "value": len(shared_diseases)},
            {"metric": "kaggle_train_rows_shared", "value": len(X_kaggle_shared)},
            {"metric": "symbipredict_unique_patterns_shared", "value": len(retained_filtered)},
            {"metric": "symbipredict_train_rows_shared", "value": len(X_sym_train)},
            {"metric": "symbipredict_test_rows_shared", "value": len(X_sym_test)},
            {"metric": "kaggle_feature_count", "value": len(feature_names)},
            {"metric": "symbipredict_only_feature_count", "value": len(symbipredict_only_features)},
            {"metric": "union_feature_count", "value": len(union_feature_names)},
        ]
    )

    shared_bundle = pd.DataFrame(
        {
            "split": ["kaggle_train", "symbipredict_train", "symbipredict_test"],
            "rows": [len(X_kaggle_shared), len(X_sym_train), len(X_sym_test)],
        }
    )
    dataset_summary_df = pd.concat([dataset_summary_df, shared_bundle.rename(columns={"split": "metric", "rows": "value"})], ignore_index=True)
    validation_scope_df = build_validation_scope_df(
        row_alignment_df=row_alignment_df,
        shared_diseases=shared_diseases,
        X_kaggle_shared=X_kaggle_shared,
        X_sym_train=X_sym_train,
        X_sym_test=X_sym_test,
        y_sym_text_train=y_sym_text_train,
        y_sym_text_test=y_sym_text_test,
        feature_names=feature_names,
        symbipredict_only_features=symbipredict_only_features,
        union_feature_names=union_feature_names,
    )
    per_class_sample_size_df = build_per_class_sample_size_df(disease_coverage_df)

    return (
        pd.DataFrame(
            {
                "X_kaggle_shared": [X_kaggle_shared],
                "y_kaggle_shared": [y_kaggle_shared],
                "X_sym_train": [X_sym_train],
                "y_sym_train": [y_sym_train],
                "X_sym_test": [X_sym_test],
                "y_sym_test": [y_sym_test],
                "meta_test": [meta_test.reset_index(drop=True)],
                "union_feature_names": [union_feature_names],
            }
        ),
        dataset_summary_df,
        validation_scope_df,
        per_class_sample_size_df,
        disease_coverage_df,
        local_encoder,
        retained_filtered,
    )


# =============================================================================
# TRAIN / EVAL
# =============================================================================
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
                "raw_prognosis": row["raw_prognosis"],
                "mapped_disease": row["mapped_disease"],
                "mapping_status": row["mapping_status"],
                "predicted_disease": class_names[pred_idx],
                "top1_correct": int(pred_idx == true_idx),
                "top3_correct": int(true_idx in top3.tolist()),
                "top1_confidence": float(np.max(y_proba[idx])),
                "top3_pred_1": class_names[top3[0]],
                "top3_pred_2": class_names[top3[1]] if len(top3) > 1 else None,
                "top3_pred_3": class_names[top3[2]] if len(top3) > 2 else None,
                "active_raw_feature_count": row["active_raw_feature_count"],
                "mapped_feature_count": row["mapped_feature_count"],
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


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    print("--- Objective 1: Single vs Multi-dataset Comparison ---")
    script_start = time.perf_counter()

    try:
        section_start = time.perf_counter()
        try:
            shared_bundle_df, dataset_summary_df, validation_scope_df, per_class_sample_size_df, disease_coverage_df, local_encoder, retained_filtered = build_shared_space()
            shared_bundle = shared_bundle_df.iloc[0]
            X_kaggle_shared = shared_bundle["X_kaggle_shared"]
            y_kaggle_shared = shared_bundle["y_kaggle_shared"]
            X_sym_train = shared_bundle["X_sym_train"]
            y_sym_train = shared_bundle["y_sym_train"]
            X_sym_test = shared_bundle["X_sym_test"]
            y_sym_test = shared_bundle["y_sym_test"]
            meta_test = shared_bundle["meta_test"]
            union_feature_names = shared_bundle["union_feature_names"]

            class_names = local_encoder.classes_
            print(f"Selected algorithm from Objective 2: {SELECTED_MODEL_NAME}")
            print(f"Mapping scope: {dataset_summary_df.loc[dataset_summary_df['metric'] == 'mapping_scope', 'value'].iloc[0]}")
            print(f"Shared diseases: {len(class_names)}")
            print(f"Kaggle train rows in shared space: {len(X_kaggle_shared)}")
            print(f"SymbiPredict train rows: {len(X_sym_train)}")
            print(f"SymbiPredict test rows: {len(X_sym_test)}")
            print(f"Union feature count: {len(union_feature_names)}")
        finally:
            timer("build_shared_space", section_start)

        section_start = time.perf_counter()
        try:
            kaggle_only_summary_df, kaggle_only_predictions_df, kaggle_only_report_df, kaggle_only_cm_df = evaluate_regime(
                evaluation_name="kaggle_only",
                X_train=X_kaggle_shared,
                y_train=y_kaggle_shared,
                X_test=X_sym_test,
                y_test=y_sym_test,
                meta_test=meta_test,
                class_names=class_names,
            )

            X_multi = pd.concat([X_kaggle_shared, X_sym_train], ignore_index=True)
            y_multi = np.concatenate([y_kaggle_shared, y_sym_train])
            multi_summary_df, multi_predictions_df, multi_report_df, multi_cm_df = evaluate_regime(
                evaluation_name="kaggle_plus_symbipredict",
                X_train=X_multi,
                y_train=y_multi,
                X_test=X_sym_test,
                y_test=y_sym_test,
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
                predictions_df.to_excel(writer, sheet_name="predictions", index=False)
                classification_report_df.to_excel(writer, sheet_name="classification_report", index=False)
                confusion_matrix_df.to_excel(writer, sheet_name="confusion_matrix", index=False)

            print("=" * 72)
            print("OBJECTIVE 1 COMPARISON COMPLETE")
            print("=" * 72)
            kaggle_only_row = comparison_df.loc[comparison_df["evaluation_name"] == "kaggle_only"].iloc[0]
            multi_row = comparison_df.loc[comparison_df["evaluation_name"] == "kaggle_plus_symbipredict"].iloc[0]
            print(f"Kaggle-only Top-1: {float(kaggle_only_row['top1_accuracy']):.4f}")
            print(f"Kaggle+SymbiPredict Top-1: {float(multi_row['top1_accuracy']):.4f}")
            print(
                "Top-1 improvement: "
                f"{float(multi_row['top1_accuracy'] - kaggle_only_row['top1_accuracy']):.4f}"
            )
            print(f"Kaggle-only Macro-F1: {float(kaggle_only_row['macro_f1']):.4f}")
            print(f"Kaggle+SymbiPredict Macro-F1: {float(multi_row['macro_f1']):.4f}")
            print(f"Workbook saved: {REPORT_WORKBOOK_PATH.name}")
        finally:
            timer("save_outputs", section_start)
    finally:
        timer("total_script", script_start)


if __name__ == "__main__":
    main()
