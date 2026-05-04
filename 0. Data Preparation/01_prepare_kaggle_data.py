from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# CONFIG
# =============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
KAGGLE_ROOT = REPO_ROOT / "1. Kaggle_Training"

INPUT_CANDIDATES = (
    PROJECT_ROOT / "Kaggle.csv",
    PROJECT_ROOT / "input" / "Kaggle.csv",
)

ARTIFACTS_DIR = KAGGLE_ROOT / "Artifacts_kaggle"
OUTPUT_DIR = PROJECT_ROOT / "output"
NEXT_STEP_INPUT_DIR = KAGGLE_ROOT / "input"
REPORT_WORKBOOK_PATH = OUTPUT_DIR / "kaggle_data_preparation_report.xlsx"

for d in [ARTIFACTS_DIR, OUTPUT_DIR, NEXT_STEP_INPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# UTILS
# =============================================================================
def timer(label: str, start: float) -> None:
    print(f"[TIMER] {label}: {time.perf_counter() - start:.2f} sec")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def cleanup_legacy_output_files() -> None:
    legacy_filenames = [
        "cleaning_summary.csv",
        "feature_name_audit.csv",
        "removed_constant_columns.csv",
        "singleton_classes_report.csv",
        "label_mapping.csv",
        "final_feature_names.csv",
        "split_info.csv",
    ]
    for filename in legacy_filenames:
        path = OUTPUT_DIR / filename
        if path.exists():
            try:
                path.unlink()
            except PermissionError:
                print(f"[WARN] Could not remove locked legacy file: {path.name}")


def resolve_input_file() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    searched = "\n - ".join(str(p) for p in INPUT_CANDIDATES)
    raise FileNotFoundError(f"Missing input file. Checked:\n - {searched}")


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("&", " and ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_column_name(name: str) -> str:
    text = normalize_text(name)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_target_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def merge_duplicate_columns_by_max(df: pd.DataFrame) -> pd.DataFrame:
    # after renaming, duplicate columns may appear
    # for binary symptom columns, max() is equivalent to OR
    return df.T.groupby(level=0).max().T


# =============================================================================
# CORE CLEANING
# =============================================================================
def detect_target_column(columns: list[str]) -> str:
    for c in ["diseases", "disease", "prognosis"]:
        if c in columns:
            return c
    raise KeyError("Could not find target column among: diseases / disease / prognosis")


def clean_kaggle_dataframe(
    csv_path: Path,
) -> Tuple[pd.DataFrame, str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        df_cleaned      -> cleaned dataframe with cleaned features + target
        target_col      -> detected target column name
        feature_audit   -> original vs cleaned feature column names
        cleaning_log    -> one-row summary of cleaning counts
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input file: {csv_path}")

    print("=" * 72)
    print("STEP 0 - KAGGLE DATA CLEANING / PREPARATION")
    print("=" * 72)

    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {csv_path.name}")
    print(f"Original shape: {df.shape}")

    original_rows = len(df)
    original_columns = list(df.columns)

    # -------------------------------------------------------------------------
    # 1. detect target
    # -------------------------------------------------------------------------
    target_col = detect_target_column(original_columns)
    print(f"Detected target column: {target_col}")

    # -------------------------------------------------------------------------
    # 2. clean target text
    # -------------------------------------------------------------------------
    df[target_col] = df[target_col].map(clean_target_text)

    # -------------------------------------------------------------------------
    # 3. remove empty target rows
    # -------------------------------------------------------------------------
    before_blank_target = len(df)
    df = df[df[target_col].astype(str).str.strip() != ""].copy()
    removed_blank_target = before_blank_target - len(df)

    # -------------------------------------------------------------------------
    # 4. clean feature column names
    # -------------------------------------------------------------------------
    original_feature_cols = [c for c in df.columns if c != target_col]
    rename_map = {col: clean_column_name(col) for col in original_feature_cols}

    feature_audit = pd.DataFrame({
        "original_column": original_feature_cols,
        "cleaned_column": [rename_map[c] for c in original_feature_cols],
    })

    df = df.rename(columns=rename_map)

    # -------------------------------------------------------------------------
    # 5. merge duplicate cleaned columns
    # -------------------------------------------------------------------------
    n_feature_cols_before_merge = len([c for c in df.columns if c != target_col])

    # keep target out of merge logic
    df_features = df.drop(columns=[target_col]).copy()
    df_target = df[[target_col]].copy()

    df_features = merge_duplicate_columns_by_max(df_features)

    n_feature_cols_after_merge = len(df_features.columns)
    merged_duplicate_feature_count = n_feature_cols_before_merge - n_feature_cols_after_merge

    df = pd.concat([df_features, df_target], axis=1)

    # -------------------------------------------------------------------------
    # 6. numeric coercion + binary conversion
    # -------------------------------------------------------------------------
    feature_cols = [c for c in df.columns if c != target_col]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df[col] = (df[col] > 0).astype(int)

    # -------------------------------------------------------------------------
    # 7. remove exact duplicate rows
    # -------------------------------------------------------------------------
    before_duplicates = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed_exact_duplicates = before_duplicates - len(df)

    # -------------------------------------------------------------------------
    # 8. remove constant columns (all 0 or all 1)
    # -------------------------------------------------------------------------
    feature_cols = [c for c in df.columns if c != target_col]
    nunique = df[feature_cols].nunique(dropna=False)

    constant_cols = nunique[nunique <= 1].index.tolist()
    zero_only_cols = [c for c in constant_cols if df[c].iloc[0] == 0] if constant_cols else []
    one_only_cols = [c for c in constant_cols if df[c].iloc[0] == 1] if constant_cols else []

    if constant_cols:
        df = df.drop(columns=constant_cols)

    # final feature cols after all cleaning
    final_feature_cols = [c for c in df.columns if c != target_col]

    cleaning_log = pd.DataFrame([{
        "input_file": csv_path.name,
        "original_rows": original_rows,
        "original_total_columns": len(original_columns),
        "original_feature_columns": len(original_feature_cols),
        "removed_blank_target_rows": removed_blank_target,
        "merged_duplicate_feature_columns": merged_duplicate_feature_count,
        "removed_exact_duplicate_rows": removed_exact_duplicates,
        "removed_constant_columns_total": len(constant_cols),
        "removed_all_zero_columns": len(zero_only_cols),
        "removed_all_one_columns": len(one_only_cols),
        "final_rows": len(df),
        "final_feature_columns": len(final_feature_cols),
        "target_column": target_col,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }])

    print(f"Removed blank-target rows: {removed_blank_target}")
    print(f"Merged duplicate cleaned feature columns: {merged_duplicate_feature_count}")
    print(f"Removed exact duplicate rows: {removed_exact_duplicates}")
    print(f"Removed constant columns: {len(constant_cols)}")
    print(f"Final shape: {df.shape}")

    if constant_cols:
        removed_constant_columns = pd.DataFrame({
                "feature_name": constant_cols,
                "constant_type": [
                    "all_zero" if c in zero_only_cols else "all_one"
                    for c in constant_cols
                ],
            })
    else:
        removed_constant_columns = pd.DataFrame(columns=["feature_name", "constant_type"])

    return df, target_col, feature_audit, cleaning_log, removed_constant_columns


# =============================================================================
# SPLIT LOGIC
# =============================================================================
def drop_singleton_classes(
    X: pd.DataFrame,
    y: pd.Series,
):
    counts = y.value_counts()
    singleton_labels = counts[counts < 2].index.tolist()
    singleton_mask = y.isin(singleton_labels)

    singleton_report = pd.DataFrame({
        "disease_name": singleton_labels,
        "full_dataset_count": [int(counts[label]) for label in singleton_labels],
        "handling_action": ["removed_before_split"] * len(singleton_labels),
    }).sort_values(["full_dataset_count", "disease_name"], ascending=[True, True])

    X_filtered = X.loc[~singleton_mask].reset_index(drop=True)
    y_filtered = y.loc[~singleton_mask].reset_index(drop=True)

    if y_filtered.empty:
        raise ValueError("All classes are singletons; no rows remain after removing singleton classes.")

    return X_filtered, y_filtered, singleton_report


def split_filtered_data(
    X: pd.DataFrame,
    y_encoded: np.ndarray,
):
    if len(np.unique(y_encoded)) == 0:
        raise ValueError("No classes remain after singleton removal; cannot create a holdout split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train, y_test


def get_cv_subset(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    class_names: np.ndarray,
):
    counts = pd.Series(y_train).value_counts().sort_index()
    eligible_ids = counts[counts >= 2].index.tolist()
    mask = pd.Series(y_train).isin(eligible_ids).values

    X_cv = X_train.loc[mask].reset_index(drop=True)
    y_cv = y_train[mask]

    excluded_ids = counts[counts < 2].index.tolist()

    cv_excluded_report = pd.DataFrame({
        "label_id": excluded_ids,
        "disease_name": [class_names[i] for i in excluded_ids],
        "train_count": [int(counts[i]) for i in excluded_ids],
        "excluded_reason": ["count_lt_2_in_train"] * len(excluded_ids),
    }).sort_values(["train_count", "disease_name"], ascending=[True, True])

    return X_cv, y_cv, cv_excluded_report


# =============================================================================
# SAVE PREPARED OUTPUTS
# =============================================================================
def build_summary_sheet(cleaning_summary: pd.DataFrame, split_info: pd.DataFrame) -> pd.DataFrame:
    cleaning_row = cleaning_summary.iloc[0]
    split_row = split_info.iloc[0]
    return pd.DataFrame(
        [
            {
                "input_file": cleaning_row["input_file"],
                "dataset": split_row["dataset"],
                "target_column": split_row["target_column"],
                "original_rows": cleaning_row["original_rows"],
                "rows_after_cleaning": cleaning_row["final_rows"],
                "original_feature_columns": cleaning_row["original_feature_columns"],
                "final_feature_columns": split_row["n_features_final"],
                "removed_blank_target_rows": cleaning_row["removed_blank_target_rows"],
                "merged_duplicate_feature_columns": cleaning_row["merged_duplicate_feature_columns"],
                "removed_exact_duplicate_rows": cleaning_row["removed_exact_duplicate_rows"],
                "removed_constant_columns_total": cleaning_row["removed_constant_columns_total"],
                "removed_all_zero_columns": cleaning_row["removed_all_zero_columns"],
                "removed_all_one_columns": cleaning_row["removed_all_one_columns"],
                "singleton_classes_removed_before_split": split_row["singleton_classes_removed_before_split"],
                "final_class_count": split_row["n_classes_final"],
                "train_rows": split_row["train_rows"],
                "test_rows": split_row["test_rows"],
                "cv_rows": split_row["cv_rows"],
                "test_size_requested": split_row["test_size_requested"],
                "random_state": split_row["random_state"],
            }
        ]
    )


def build_feature_reference_sheet(
    feature_audit: pd.DataFrame,
    feature_names: list[str],
    removed_constant_columns: pd.DataFrame,
) -> pd.DataFrame:
    final_feature_set = set(feature_names)
    removed_constant_map = {}
    if not removed_constant_columns.empty:
        removed_constant_map = dict(
            zip(removed_constant_columns["feature_name"], removed_constant_columns["constant_type"])
        )

    feature_reference = (
        feature_audit.groupby("cleaned_column", as_index=False)
        .agg(
            original_columns=("original_column", lambda values: " | ".join(values)),
            source_column_count=("original_column", "count"),
        )
        .sort_values("cleaned_column", kind="stable")
        .reset_index(drop=True)
    )
    feature_reference["is_final_feature"] = feature_reference["cleaned_column"].isin(final_feature_set)
    feature_reference["feature_status"] = np.where(
        feature_reference["is_final_feature"],
        "final_feature",
        "removed_constant",
    )
    feature_reference["constant_type"] = feature_reference["cleaned_column"].map(removed_constant_map).fillna("")

    return feature_reference[
        [
            "cleaned_column",
            "original_columns",
            "source_column_count",
            "is_final_feature",
            "feature_status",
            "constant_type",
        ]
    ]


def save_excel_report(
    cleaning_summary: pd.DataFrame,
    split_info: pd.DataFrame,
    feature_audit: pd.DataFrame,
    removed_constant_columns: pd.DataFrame,
    singleton_report: pd.DataFrame,
    feature_names: list[str],
    label_encoder: LabelEncoder,
) -> None:
    summary_sheet = build_summary_sheet(cleaning_summary, split_info)
    feature_reference_sheet = build_feature_reference_sheet(
        feature_audit,
        feature_names,
        removed_constant_columns,
    )
    label_mapping_sheet = pd.DataFrame(
        {
            "label_id": np.arange(len(label_encoder.classes_)),
            "disease_name": label_encoder.classes_,
        }
    )

    with pd.ExcelWriter(REPORT_WORKBOOK_PATH, engine="openpyxl") as writer:
        summary_sheet.to_excel(writer, sheet_name="summary", index=False)
        singleton_report.to_excel(writer, sheet_name="singleton_removed", index=False)
        feature_reference_sheet.to_excel(writer, sheet_name="features", index=False)
        label_mapping_sheet.to_excel(writer, sheet_name="label_mapping", index=False)

    cleanup_legacy_output_files()


def save_prepared_outputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_cv: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_cv: np.ndarray,
    label_encoder: LabelEncoder,
    singleton_report: pd.DataFrame,
    cv_excluded_report: pd.DataFrame,
    feature_names: list[str],
    target_col: str,
    feature_audit: pd.DataFrame,
    cleaning_summary: pd.DataFrame,
    removed_constant_columns: pd.DataFrame,
) -> None:
    # main prepared matrices
    save_csv(X_train, NEXT_STEP_INPUT_DIR / "X_train.csv")
    save_csv(X_test, NEXT_STEP_INPUT_DIR / "X_test.csv")
    save_csv(X_cv, NEXT_STEP_INPUT_DIR / "X_cv.csv")

    save_csv(pd.DataFrame({"target": y_train}), NEXT_STEP_INPUT_DIR / "y_train.csv")
    save_csv(pd.DataFrame({"target": y_test}), NEXT_STEP_INPUT_DIR / "y_test.csv")
    save_csv(pd.DataFrame({"target": y_cv}), NEXT_STEP_INPUT_DIR / "y_cv.csv")

    # artifacts
    joblib.dump(label_encoder, ARTIFACTS_DIR / "label_encoder.pkl")
    joblib.dump(feature_names, ARTIFACTS_DIR / "feature_names.pkl")

    split_info = pd.DataFrame([{
        "dataset": "Kaggle cleaned",
        "target_column": target_col,
        "n_features_final": len(feature_names),
        "n_classes_final": len(label_encoder.classes_),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "cv_rows": len(X_cv),
        "singleton_classes_removed_before_split": len(singleton_report),
        "cv_excluded_classes_lt2_in_train": len(cv_excluded_report),
        "test_size_requested": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }])

    save_excel_report(
        cleaning_summary=cleaning_summary,
        split_info=split_info,
        feature_audit=feature_audit,
        removed_constant_columns=removed_constant_columns,
        singleton_report=singleton_report,
        feature_names=feature_names,
        label_encoder=label_encoder,
    )


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    script_start = time.perf_counter()

    try:
        section_start = time.perf_counter()
        try:
            input_file = resolve_input_file()
            df_cleaned, target_col, feature_audit, cleaning_summary, removed_constant_columns = clean_kaggle_dataframe(
                input_file
            )
        finally:
            timer("clean_kaggle_dataframe", section_start)

        section_start = time.perf_counter()
        try:
            X = df_cleaned.drop(columns=[target_col]).copy().astype(int)
            y = df_cleaned[target_col].astype(str).copy()

            X_filtered, y_filtered, singleton_report = drop_singleton_classes(X, y)

            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y_filtered)
            class_names = label_encoder.classes_
            feature_names = X_filtered.columns.tolist()

            X_train, X_test, y_train, y_test = split_filtered_data(X_filtered, y_encoded)

            X_cv, y_cv, cv_excluded_report = get_cv_subset(
                X_train, y_train, class_names
            )

            save_prepared_outputs(
                X_train=X_train,
                X_test=X_test,
                X_cv=X_cv,
                y_train=y_train,
                y_test=y_test,
                y_cv=y_cv,
                label_encoder=label_encoder,
                singleton_report=singleton_report,
                cv_excluded_report=cv_excluded_report,
                feature_names=feature_names,
                target_col=target_col,
                feature_audit=feature_audit,
                cleaning_summary=cleaning_summary,
                removed_constant_columns=removed_constant_columns,
            )

            print("=" * 72)
            print("PREPARATION COMPLETE")
            print("=" * 72)
            print(f"Features: {len(feature_names)}")
            print(f"Classes: {len(class_names)}")
            print(f"Train rows: {len(X_train)}")
            print(f"Test rows: {len(X_test)}")
            print(f"CV rows: {len(X_cv)}")
            print(f"Singleton classes removed before split: {len(singleton_report)}")
            if len(cv_excluded_report) > 0:
                print(f"CV-excluded classes (<2 in train): {len(cv_excluded_report)}")
            print(f"Next-step input saved to: {NEXT_STEP_INPUT_DIR}")
            print(f"Artifacts saved to: {ARTIFACTS_DIR}")
            print(f"Excel report saved to: {REPORT_WORKBOOK_PATH}")
        finally:
            timer("encode_split_and_save", section_start)

    finally:
        timer("total_script", script_start)


if __name__ == "__main__":
    main()
