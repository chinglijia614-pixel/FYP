from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from openai import OpenAI
from sklearn.preprocessing import LabelEncoder


RUNTIME_START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "1. Kaggle_Training" / "Artifacts_kaggle"
KAGGLE_INPUT_DIR = ROOT / "1. Kaggle_Training" / "input"
DDXPLUS_ALIGNMENT_REPORT = ROOT / "2. External_Validation" / "output" / "ddxplus_alignment_report.xlsx"
KAGGLE_REPORT = ROOT / "1. Kaggle_Training" / "output" / "kaggle_model_training_report.xlsx"
DDXPLUS_COMPARISON_REPORT = (
    ROOT / "2. External_Validation" / "output" / "objective1_ddxplus_single_vs_multidataset_report.xlsx"
)
CACHE_DIR = ROOT / "3. Prototype" / "cache"
DDXPLUS_MODEL_CACHE = CACHE_DIR / "kaggle_ddxplus_xgboost_bundle.pkl"
TOP_K = 3

FEATURE_FILE = ARTIFACTS_DIR / "feature_names.pkl"
ENCODER_FILE = ARTIFACTS_DIR / "label_encoder.pkl"
KAGGLE_XGBOOST_FILE = ARTIFACTS_DIR / "xgboost_model.pkl"

MODEL_OPTIONS = {
    "kaggle (xgboost)": "kaggle_xgboost",
    "kaggle + ddxplus (xgboost)": "kaggle_ddxplus_xgboost",
}
DEFAULT_MODEL_LABEL = "kaggle (xgboost)"


def normalize_text(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("&", " and ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
        random_state=42,
    )


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


@st.cache_resource
def load_base_artifacts() -> tuple[list[str], LabelEncoder, dict[str, str]]:
    if not FEATURE_FILE.exists() or not ENCODER_FILE.exists():
        raise FileNotFoundError(
            "Missing feature_names.pkl or label_encoder.pkl. Run the Kaggle training pipeline first."
        )

    features: list[str] = joblib.load(FEATURE_FILE)
    encoder: LabelEncoder = joblib.load(ENCODER_FILE)
    feature_lookup = {normalize_text(name): name for name in features}
    return features, encoder, feature_lookup


@st.cache_data
def load_ddxplus_alignment_rows() -> pd.DataFrame:
    if not DDXPLUS_ALIGNMENT_REPORT.exists():
        raise FileNotFoundError(
            "Missing DDXPlus alignment workbook. Run 2. External_Validation/align_ddxplus.py first."
        )
    return pd.read_excel(DDXPLUS_ALIGNMENT_REPORT, sheet_name="row_alignment_internal")


@st.cache_resource
def load_kaggle_xgboost_resources() -> dict[str, Any]:
    features, encoder, feature_lookup = load_base_artifacts()
    if not KAGGLE_XGBOOST_FILE.exists():
        raise FileNotFoundError(f"Missing model file: {KAGGLE_XGBOOST_FILE}")

    model = joblib.load(KAGGLE_XGBOOST_FILE)
    return {
        "model": model,
        "encoder": encoder,
        "features": features,
        "feature_lookup": feature_lookup,
        "shared_diseases": None,
        "mode_note": "Full Kaggle disease space",
    }


def build_feature_matrix_from_texts(feature_names: list[str], feature_texts: pd.Series) -> pd.DataFrame:
    matrix = pd.DataFrame(0, index=feature_texts.index, columns=feature_names, dtype=np.uint8)
    for idx, feature_text in feature_texts.items():
        for feature in str(feature_text).split(" | "):
            if feature:
                matrix.at[idx, feature] = 1
    return matrix


@st.cache_data
def load_kaggle_xgboost_metrics() -> dict[str, Any]:
    if not KAGGLE_REPORT.exists():
        raise FileNotFoundError(
            "Missing Kaggle model training report. Run 1. Kaggle_Training before using the prototype metrics."
        )

    report_df = pd.read_excel(KAGGLE_REPORT, sheet_name="model_summary")
    row = report_df.loc[report_df["model"].astype(str).str.lower() == "xgboost"]
    if row.empty:
        raise ValueError("XGBoost metrics were not found in kaggle_model_training_report.xlsx.")

    metrics = row.iloc[0]
    return {
        "scope_label": "Kaggle holdout evaluation",
        "disease_classes": 727,
        "top1_accuracy": float(metrics["holdout_top1_accuracy"]),
        "top3_accuracy": float(metrics["holdout_top3_accuracy"]),
        "macro_f1": float(metrics["holdout_macro_f1"]),
        "ece": float(metrics["holdout_ece"]),
    }


@st.cache_data
def load_ddxplus_comparison_metrics() -> dict[str, Any]:
    if not DDXPLUS_COMPARISON_REPORT.exists():
        raise FileNotFoundError(
            "Missing DDXPlus comparison report. Run 2. External_Validation/03_compare_kaggle_vs_ddxplus.py first."
        )

    summary_df = pd.read_excel(DDXPLUS_COMPARISON_REPORT, sheet_name="summary", header=None).fillna("")

    shared_disease_count = 0
    ddxplus_test_rows = 0
    for _, row in summary_df.iterrows():
        key = str(row.iloc[0]).strip()
        value = row.iloc[1] if len(row) > 1 else ""
        if key == "shared_disease_count":
            shared_disease_count = int(value)
        elif key == "ddxplus_test_rows_shared":
            ddxplus_test_rows = int(value)

    header_row = None
    for idx, row in summary_df.iterrows():
        if str(row.iloc[0]).strip() == "evaluation_name":
            header_row = idx
            break

    if header_row is None:
        raise ValueError("Comparison metrics section was not found in the DDXPlus report.")

    header = [str(item).strip() for item in summary_df.iloc[header_row].tolist() if str(item).strip()]
    metric_rows = []
    for idx in range(header_row + 1, len(summary_df)):
        first_cell = str(summary_df.iloc[idx, 0]).strip()
        if not first_cell:
            break
        metric_rows.append(summary_df.iloc[idx, : len(header)].tolist())

    metrics_df = pd.DataFrame(metric_rows, columns=header)
    for column in metrics_df.columns[1:]:
        try:
            metrics_df[column] = pd.to_numeric(metrics_df[column])
        except (TypeError, ValueError):
            continue

    kaggle_only = metrics_df.loc[metrics_df["evaluation_name"] == "kaggle_only"].iloc[0].to_dict()
    kaggle_plus = metrics_df.loc[metrics_df["evaluation_name"] == "kaggle_plus_ddxplus"].iloc[0].to_dict()

    return {
        "scope_label": "DDXPlus shared-disease external evaluation",
        "shared_disease_count": shared_disease_count,
        "ddxplus_test_rows": ddxplus_test_rows,
        "kaggle_only": kaggle_only,
        "kaggle_plus_ddxplus": kaggle_plus,
    }


@st.cache_resource
def load_kaggle_ddxplus_xgboost_resources() -> dict[str, Any]:
    features, kaggle_encoder, feature_lookup = load_base_artifacts()
    row_alignment_df = load_ddxplus_alignment_rows()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if DDXPLUS_MODEL_CACHE.exists():
        cached_bundle = joblib.load(DDXPLUS_MODEL_CACHE)
        shared_diseases = list(cached_bundle["classes"])
        local_encoder = LabelEncoder()
        local_encoder.fit(shared_diseases)
        return {
            "model": cached_bundle["model"],
            "encoder": local_encoder,
            "features": features,
            "feature_lookup": feature_lookup,
            "shared_diseases": shared_diseases,
            "mode_note": f"{len(shared_diseases)} shared Kaggle-DDXPlus diseases",
        }

    train_df = row_alignment_df.loc[row_alignment_df["source_split"].isin(["train", "validate"])].copy().reset_index(drop=True)
    train_df = train_df.drop_duplicates(subset=["mapped_disease", "mapped_features"], keep="first").reset_index(drop=True)
    test_df = row_alignment_df.loc[row_alignment_df["source_split"] == "test"].copy().reset_index(drop=True)

    shared_diseases = sorted(set(train_df["mapped_disease"]).intersection(set(test_df["mapped_disease"])))
    train_df = train_df.loc[train_df["mapped_disease"].isin(shared_diseases)].reset_index(drop=True)

    local_encoder = LabelEncoder()
    local_encoder.fit(shared_diseases)

    X_kaggle = pd.read_csv(KAGGLE_INPUT_DIR / "X_train.csv")[features]
    y_kaggle = pd.read_csv(KAGGLE_INPUT_DIR / "y_train.csv")["target"].to_numpy()
    kaggle_disease_text = kaggle_encoder.inverse_transform(y_kaggle.astype(int))
    kaggle_df = X_kaggle.copy()
    kaggle_df["mapped_disease"] = kaggle_disease_text
    kaggle_df = kaggle_df.loc[kaggle_df["mapped_disease"].isin(shared_diseases)].reset_index(drop=True)

    X_ddx_train = build_feature_matrix_from_texts(features, train_df["mapped_features"])
    y_ddx_train = local_encoder.transform(train_df["mapped_disease"].astype(str))

    X_kaggle_shared = kaggle_df[features].reset_index(drop=True)
    y_kaggle_shared = local_encoder.transform(kaggle_df["mapped_disease"].astype(str))

    X_multi = pd.concat([X_kaggle_shared, X_ddx_train], ignore_index=True)
    y_multi = np.concatenate([y_kaggle_shared, y_ddx_train])

    model = build_xgboost_model(len(shared_diseases))
    model.fit(X_multi, y_multi)
    joblib.dump({"model": model, "classes": shared_diseases}, DDXPLUS_MODEL_CACHE)

    return {
        "model": model,
        "encoder": local_encoder,
        "features": features,
        "feature_lookup": feature_lookup,
        "shared_diseases": shared_diseases,
        "mode_note": f"{len(shared_diseases)} shared Kaggle-DDXPlus diseases",
    }


def get_model_resources(model_label: str) -> dict[str, Any]:
    mode = MODEL_OPTIONS[model_label]
    if mode == "kaggle_xgboost":
        return load_kaggle_xgboost_resources()
    if mode == "kaggle_ddxplus_xgboost":
        return load_kaggle_ddxplus_xgboost_resources()
    raise ValueError(f"Unknown model mode: {mode}")


@st.cache_data
def build_disease_library() -> tuple[list[str], dict[str, Any]]:
    features, kaggle_encoder, _ = load_base_artifacts()
    row_alignment_df = load_ddxplus_alignment_rows().copy()

    train_df = row_alignment_df.loc[row_alignment_df["source_split"].isin(["train", "validate"])].copy().reset_index(drop=True)
    train_df = train_df.drop_duplicates(subset=["mapped_disease", "mapped_features"], keep="first").reset_index(drop=True)
    test_df = row_alignment_df.loc[row_alignment_df["source_split"] == "test"].copy().reset_index(drop=True)
    shared_diseases = sorted(set(train_df["mapped_disease"]).intersection(set(test_df["mapped_disease"])))

    X_kaggle = pd.read_csv(KAGGLE_INPUT_DIR / "X_train.csv")[features]
    y_kaggle = pd.read_csv(KAGGLE_INPUT_DIR / "y_train.csv")["target"].to_numpy()
    kaggle_disease_text = kaggle_encoder.inverse_transform(y_kaggle.astype(int))
    kaggle_df = X_kaggle.copy()
    kaggle_df["mapped_disease"] = kaggle_disease_text
    kaggle_df = kaggle_df.loc[kaggle_df["mapped_disease"].isin(shared_diseases)].reset_index(drop=True)

    ddx_all_df = row_alignment_df.loc[row_alignment_df["mapped_disease"].isin(shared_diseases)].copy().reset_index(drop=True)

    library: dict[str, Any] = {}
    for disease in shared_diseases:
        kaggle_rows = kaggle_df.loc[kaggle_df["mapped_disease"] == disease, features]
        ddx_rows = ddx_all_df.loc[ddx_all_df["mapped_disease"] == disease].copy()

        kaggle_counts = kaggle_rows.sum(axis=0).astype(int).to_dict()
        ddx_counter: Counter[str] = Counter()
        for feature_text in ddx_rows["mapped_features"]:
            for feature in str(feature_text).split(" | "):
                if feature:
                    ddx_counter[feature] += 1

        symptom_rows = []
        symptom_names = sorted({feature for feature, count in kaggle_counts.items() if count > 0} | set(ddx_counter))
        for symptom in symptom_names:
            kaggle_count = int(kaggle_counts.get(symptom, 0))
            ddx_count = int(ddx_counter.get(symptom, 0))
            if kaggle_count > 0 and ddx_count > 0:
                source = "Both"
            elif kaggle_count > 0:
                source = "Kaggle"
            else:
                source = "DDXPlus"
            symptom_rows.append(
                {
                    "symptom": symptom,
                    "source": source,
                    "kaggle_rows": kaggle_count,
                    "ddxplus_patterns": ddx_count,
                }
            )

        symptom_df = pd.DataFrame(symptom_rows)
        if not symptom_df.empty:
            source_rank = {"Both": 0, "Kaggle": 1, "DDXPlus": 2}
            symptom_df["source_rank"] = symptom_df["source"].map(source_rank)
            symptom_df["combined_support"] = symptom_df["kaggle_rows"] + symptom_df["ddxplus_patterns"]
            symptom_df = symptom_df.sort_values(
                ["source_rank", "combined_support", "symptom"],
                ascending=[True, False, True],
            ).reset_index(drop=True)
            symptom_df = symptom_df.drop(columns=["source_rank", "combined_support"])

        library[disease] = {
            "kaggle_rows": int(len(kaggle_rows)),
            "ddxplus_patterns": int(len(ddx_rows)),
            "symptom_table": symptom_df,
        }

    return shared_diseases, library


def extract_symptoms_from_text(user_text: str, valid_features: list[str]) -> tuple[list[str], str | None]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return [], "Missing OPENAI_API_KEY environment variable."

    client = OpenAI(api_key=api_key)
    prompt = (
        "You extract symptoms from patient descriptions.\n"
        "Return only a JSON array of strings.\n"
        "Use only items from the valid symptom list below.\n"
        "Do not invent new symptoms.\n"
        "If nothing is clearly matchable, return [].\n\n"
        f"Valid symptom list:\n{json.dumps(valid_features)}\n\n"
        f"User description:\n{user_text}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Return only a JSON array of strings."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or "[]"
        extracted = json.loads(content)
        if not isinstance(extracted, list):
            return [], "The model returned an unexpected response format."

        cleaned = [item for item in extracted if item in valid_features]
        return sorted(set(cleaned), key=valid_features.index), None
    except Exception as exc:
        return [], f"LLM extraction failed: {exc}"


def build_input_vector(features: list[str], selected_symptoms: list[str]) -> np.ndarray:
    feature_index = {feature: idx for idx, feature in enumerate(features)}
    vector = np.zeros(len(features), dtype=np.uint8)
    for symptom in selected_symptoms:
        idx = feature_index.get(symptom)
        if idx is not None:
            vector[idx] = 1
    return vector


def predict_top_k(model, encoder, features: list[str], selected_symptoms: list[str]) -> list[dict[str, float | str]]:
    input_vector = build_input_vector(features, selected_symptoms)
    probabilities = model.predict_proba(input_vector.reshape(1, -1))[0]
    top_indices = probabilities.argsort()[-TOP_K:][::-1]
    return [
        {
            "disease": str(encoder.inverse_transform([idx])[0]),
            "confidence_percent": float(probabilities[idx]) * 100,
        }
        for idx in top_indices
    ]


def resolve_manual_text(user_text: str, feature_lookup: dict[str, str]) -> tuple[list[str], list[str]]:
    tokens = [normalize_text(token) for token in re.split(r"[,;\n]+", user_text) if token.strip()]
    matched: list[str] = []
    unknown: list[str] = []
    for token in tokens:
        canonical = feature_lookup.get(token)
        if canonical:
            if canonical not in matched:
                matched.append(canonical)
        else:
            unknown.append(token)
    return matched, unknown


def render_prediction_cards(predictions: list[dict[str, float | str]]) -> None:
    for rank, row in enumerate(predictions, start=1):
        disease = row["disease"]
        confidence = float(row["confidence_percent"])
        st.markdown(
            f"""
            <div style="
                border: 1px solid #d9d4c7;
                background: linear-gradient(180deg, #ffffff 0%, #f8f4ea 100%);
                padding: 16px 18px;
                margin-bottom: 12px;
                border-radius: 8px;
            ">
                <div style="font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: #6b6257;">
                    Rank {rank}
                </div>
                <div style="font-size: 24px; color: #1d1a17; margin-top: 6px;">
                    {disease}
                </div>
                <div style="font-size: 16px; color: #0f766e; margin-top: 8px;">
                    {confidence:.2f}% confidence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_model_snapshot(model_label: str) -> None:
    if MODEL_OPTIONS[model_label] == "kaggle_xgboost":
        metrics = load_kaggle_xgboost_metrics()
        st.caption(metrics["scope_label"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Top-1 Accuracy", format_pct(metrics["top1_accuracy"]))
        m2.metric("Top-3 Accuracy", format_pct(metrics["top3_accuracy"]))
        m3.metric("Macro-F1", f"{metrics['macro_f1']:.3f}")
        m4.metric("ECE", f"{metrics['ece']:.3f}")
        st.info(
            "This is the full-coverage deployment model. It was trained on Kaggle and evaluated on the Kaggle holdout set across 727 disease classes."
        )
        return

    metrics = load_ddxplus_comparison_metrics()
    kaggle_only = metrics["kaggle_only"]
    kaggle_plus = metrics["kaggle_plus_ddxplus"]
    st.caption(metrics["scope_label"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Top-1 Accuracy",
        format_pct(float(kaggle_plus["top1_accuracy"])),
        f"+{format_pct(float(kaggle_plus['top1_gain_vs_kaggle_only']))}",
    )
    m2.metric(
        "Top-3 Accuracy",
        format_pct(float(kaggle_plus["top3_accuracy"])),
        f"+{format_pct(float(kaggle_plus['top3_gain_vs_kaggle_only']))}",
    )
    m3.metric(
        "Macro-F1",
        f"{float(kaggle_plus['macro_f1']):.3f}",
        f"+{float(kaggle_plus['macro_f1_gain_vs_kaggle_only']):.3f}",
    )
    m4.metric("DDXPlus Test Rows", f"{int(metrics['ddxplus_test_rows']):,}")
    st.info(
        f"Research comparison mode. The Kaggle-only baseline reached {format_pct(float(kaggle_only['top1_accuracy']))} Top-1 on the same DDXPlus external test set, while Kaggle + DDXPlus reached {format_pct(float(kaggle_plus['top1_accuracy']))} across {metrics['shared_disease_count']} shared diseases."
    )


def render_predict_page() -> None:
    model_label = st.selectbox("Prediction model", options=list(MODEL_OPTIONS.keys()), index=0)
    resources = get_model_resources(model_label)
    model = resources["model"]
    encoder = resources["encoder"]
    features = resources["features"]
    feature_lookup = resources["feature_lookup"]

    if "selected_symptoms" not in st.session_state:
        st.session_state["selected_symptoms"] = []

    st.markdown("**Model Snapshot**")
    render_model_snapshot(model_label)

    col1, col2, col3 = st.columns(3)
    col1.metric("Symptom features", len(features))
    col2.metric("Disease classes", len(encoder.classes_))
    col3.metric("Prediction scope", resources["mode_note"])

    left_col, right_col = st.columns([1.15, 0.85])

    with left_col:
        st.subheader("Symptoms")
        tab_ai, tab_manual = st.tabs(["Natural Language Input", "Manual Checklist"])

        with tab_ai:
            user_text = st.text_area(
                "Describe your symptoms",
                placeholder="Example: I have fever, cough, body ache, and chest pain.",
                height=140,
                key=f"nl_input_{MODEL_OPTIONS[model_label]}",
            )

            if st.button("Extract Symptoms with AI", use_container_width=True):
                if not user_text.strip():
                    st.warning("Please type some symptoms first.")
                else:
                    extracted, error_message = extract_symptoms_from_text(user_text, features)
                    if error_message:
                        st.error(error_message)
                    else:
                        st.session_state["selected_symptoms"] = extracted
                        if extracted:
                            st.success(f"Extracted symptoms: {', '.join(extracted)}")
                        else:
                            st.warning("No clear symptoms were extracted from the description.")

            quick_text = st.text_area(
                "Optional direct symptom text",
                placeholder="Example: cough, fever, chest pain",
                height=110,
                key=f"quick_text_{MODEL_OPTIONS[model_label]}",
            )
            if st.button("Add Typed Symptoms", use_container_width=True):
                matched, unknown = resolve_manual_text(quick_text, feature_lookup)
                current = list(st.session_state["selected_symptoms"])
                for symptom in matched:
                    if symptom not in current:
                        current.append(symptom)
                st.session_state["selected_symptoms"] = current

                if matched:
                    st.success(f"Added symptoms: {', '.join(matched)}")
                if unknown:
                    st.warning(f"Unrecognized symptoms: {', '.join(unknown)}")

        with tab_manual:
            st.session_state["selected_symptoms"] = st.multiselect(
                "Select symptoms",
                options=features,
                default=st.session_state["selected_symptoms"],
                help="These are the 328 learned Kaggle symptom features used by the selected model.",
            )

        if st.session_state["selected_symptoms"]:
            st.markdown("**Current selected symptoms**")
            st.write(", ".join(st.session_state["selected_symptoms"]))
        else:
            st.info("No symptoms selected yet.")

        clear_col, _ = st.columns([0.35, 0.65])
        if clear_col.button("Clear Symptoms", use_container_width=True):
            st.session_state["selected_symptoms"] = []
            st.rerun()

    with right_col:
        st.subheader("Prediction Output")
        if MODEL_OPTIONS[model_label] == "kaggle_ddxplus_xgboost":
            st.caption("Top-3 predictions across the 31 shared Kaggle-DDXPlus diseases.")
        else:
            st.caption("Top-3 predictions across the Kaggle disease space.")

        if st.button("Run Diagnosis", type="primary", use_container_width=True):
            selected_symptoms = st.session_state["selected_symptoms"]
            if not selected_symptoms:
                st.error("Please select at least one symptom before running diagnosis.")
            else:
                predictions = predict_top_k(model, encoder, features, selected_symptoms)
                render_prediction_cards(predictions)
                st.markdown("**Symptoms used for inference**")
                st.write(", ".join(selected_symptoms))
        else:
            st.info("Choose symptoms on the left, then run diagnosis here.")


def render_disease_library_page() -> None:
    shared_diseases, library = build_disease_library()

    st.subheader("Disease Library")
    st.caption("Shared Kaggle-DDXPlus diseases with combined mapped symptoms and dataset source markers.")
    search_term = st.text_input("Search disease", placeholder="Type a disease name")
    filtered_diseases = [disease for disease in shared_diseases if search_term.lower() in disease.lower()]
    if not filtered_diseases:
        st.warning("No diseases matched the current search.")
        return

    selected_disease = st.selectbox("Disease", options=filtered_diseases)
    disease_info = library[selected_disease]
    symptom_table = disease_info["symptom_table"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Shared diseases", len(shared_diseases))
    col2.metric("Kaggle rows", disease_info["kaggle_rows"])
    col3.metric("DDXPlus patterns", disease_info["ddxplus_patterns"])

    if symptom_table.empty:
        st.warning("No mapped symptoms were available for this disease.")
        return

    both_count = int((symptom_table["source"] == "Both").sum())
    kaggle_only_count = int((symptom_table["source"] == "Kaggle").sum())
    ddxplus_only_count = int((symptom_table["source"] == "DDXPlus").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Both", both_count)
    c2.metric("Kaggle Only", kaggle_only_count)
    c3.metric("DDXPlus Only", ddxplus_only_count)

    source_filter = st.multiselect(
        "Symptom source filter",
        options=["Both", "Kaggle", "DDXPlus"],
        default=["Both", "Kaggle", "DDXPlus"],
    )
    filtered_table = symptom_table.loc[symptom_table["source"].isin(source_filter)].reset_index(drop=True)
    top_symptoms = filtered_table.head(8)["symptom"].tolist()
    if top_symptoms:
        st.markdown("**Top mapped symptoms**")
        st.write(", ".join(top_symptoms))

    st.markdown("**Combined symptom profile**")
    st.dataframe(
        filtered_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "symptom": st.column_config.TextColumn("Symptom"),
            "source": st.column_config.TextColumn("Source"),
            "kaggle_rows": st.column_config.NumberColumn("Kaggle Rows"),
            "ddxplus_patterns": st.column_config.NumberColumn("DDXPlus Patterns"),
        },
    )


def main() -> None:
    st.set_page_config(page_title="AI Symptom Checker", layout="wide")
    st.title("AI Symptom Checker")

    try:
        load_base_artifacts()
        load_ddxplus_alignment_rows()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    predict_tab, library_tab = st.tabs(["Predict", "Disease Library"])

    with predict_tab:
        render_predict_page()

    with library_tab:
        render_disease_library_page()

    elapsed = time.perf_counter() - RUNTIME_START
    st.caption(f"Prototype ready in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
