from __future__ import annotations

import json
import os
import re
import time
import html
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
    "Kaggle-only XGBoost": "kaggle_xgboost",
    "Kaggle + DDXPlus XGBoost": "kaggle_ddxplus_xgboost",
}
MODEL_USER_LABELS = {
    "Kaggle-only XGBoost": "Broad Checker (Kaggle-only)",
    "Kaggle + DDXPlus XGBoost": "Focused Checker (Kaggle + DDXPlus)",
}
PAGE_OPTIONS = ["Model Coverage", "LLM Chat", "Manual Check"]
DEFAULT_MODEL_LABEL = "Kaggle-only XGBoost"
NAV_ICONS = {
    "Model Coverage": "[+]",
    "LLM Chat": "[ ]",
    "Manual Check": "[x]",
}
NAV_LABELS = {
    "Model Coverage": "Coverage",
    "LLM Chat": "Symptom Chat",
    "Manual Check": "Manual Check",
}
BODY_REGION_KEYWORDS = {
    "General / Whole body": [
        "fever",
        "chill",
        "fatigue",
        "malaise",
        "weakness",
        "weight",
        "sweat",
        "appetite",
        "dehydration",
        "lethargy",
    ],
    "Head / Brain": ["head", "headache", "migraine", "confusion", "memory", "seizure", "fainting", "syncope"],
    "Eyes": ["eye", "vision", "visual", "blurred", "photophobia", "red eye"],
    "Ear / Nose / Throat": [
        "ear",
        "hearing",
        "nose",
        "nasal",
        "sinus",
        "sneeze",
        "sneezing",
        "throat",
        "hoarse",
        "mouth",
        "tongue",
        "dental",
        "gum",
        "swallow",
    ],
    "Chest / Breathing": [
        "chest",
        "cough",
        "breath",
        "breathing",
        "shortness of breath",
        "wheeze",
        "wheezing",
        "sputum",
        "palpitation",
    ],
    "Abdomen / Digestive": [
        "abdominal",
        "abdomen",
        "stomach",
        "nausea",
        "vomit",
        "diarrhea",
        "constipation",
        "bowel",
        "stool",
        "heartburn",
        "reflux",
        "bloating",
    ],
    "Back / Spine": ["back", "neck", "spine", "lumbar"],
    "Arms / Hands": ["arm", "hand", "shoulder", "elbow", "wrist", "finger"],
    "Legs / Feet": ["leg", "foot", "feet", "ankle", "knee", "calf", "thigh", "toe", "walking", "gait"],
    "Skin": ["skin", "rash", "itch", "lesion", "swelling", "bruise", "blister", "redness", "jaundice", "pallor"],
    "Urinary / Reproductive": [
        "urine",
        "urinary",
        "urination",
        "bladder",
        "pelvic",
        "vaginal",
        "menstrual",
        "pregnancy",
        "testicular",
        "penis",
    ],
}


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
        "mode_note": "Broad disease coverage",
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

    train_df_for_scope = row_alignment_df.loc[row_alignment_df["source_split"].isin(["train", "validate"])].copy().reset_index(drop=True)
    train_df_for_scope = train_df_for_scope.drop_duplicates(subset=["mapped_disease", "mapped_features"], keep="first").reset_index(drop=True)
    test_df_for_scope = row_alignment_df.loc[row_alignment_df["source_split"] == "test"].copy().reset_index(drop=True)
    current_shared_diseases = sorted(
        set(train_df_for_scope["mapped_disease"]).intersection(set(test_df_for_scope["mapped_disease"]))
    )

    if DDXPLUS_MODEL_CACHE.exists():
        cached_bundle = joblib.load(DDXPLUS_MODEL_CACHE)
        shared_diseases = list(cached_bundle["classes"])
        if shared_diseases == current_shared_diseases:
            local_encoder = LabelEncoder()
            local_encoder.fit(shared_diseases)
            return {
                "model": cached_bundle["model"],
                "encoder": local_encoder,
                "features": features,
                "feature_lookup": feature_lookup,
                "shared_diseases": shared_diseases,
                "mode_note": f"{len(shared_diseases)} focused diseases",
            }

    train_df = train_df_for_scope
    shared_diseases = current_shared_diseases
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
        "mode_note": f"{len(shared_diseases)} focused diseases",
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


@st.cache_data
def build_kaggle_coverage_library() -> tuple[list[str], dict[str, Any]]:
    features, kaggle_encoder, _ = load_base_artifacts()
    X_train = pd.read_csv(KAGGLE_INPUT_DIR / "X_train.csv")[features]
    y_train = pd.read_csv(KAGGLE_INPUT_DIR / "y_train.csv")["target"].to_numpy()
    disease_labels = kaggle_encoder.inverse_transform(y_train.astype(int))

    kaggle_df = X_train.copy()
    kaggle_df["mapped_disease"] = disease_labels
    diseases = sorted(kaggle_encoder.classes_.astype(str).tolist())
    row_counts = kaggle_df["mapped_disease"].value_counts().to_dict()
    feature_sums = kaggle_df.groupby("mapped_disease")[features].sum()

    library: dict[str, Any] = {}
    for disease in diseases:
        if disease in feature_sums.index:
            counts = feature_sums.loc[disease].astype(int)
        else:
            counts = pd.Series(0, index=features, dtype=int)
        symptom_table = (
            counts[counts > 0]
            .rename_axis("symptom")
            .reset_index(name="kaggle_rows")
            .sort_values(["kaggle_rows", "symptom"], ascending=[False, True])
            .reset_index(drop=True)
        )
        library[disease] = {
            "kaggle_rows": int(row_counts.get(disease, 0)),
            "symptom_table": symptom_table,
        }

    return diseases, library


def get_coverage_for_model(model_label: str) -> tuple[list[str], dict[str, Any], str]:
    if MODEL_OPTIONS[model_label] == "kaggle_xgboost":
        diseases, library = build_kaggle_coverage_library()
        return diseases, library, "Kaggle-only coverage"

    diseases, library = build_disease_library()
    return diseases, library, "Kaggle + DDXPlus aligned coverage"


def get_covered_symptoms_for_model(model_label: str, features: list[str]) -> list[str]:
    if MODEL_OPTIONS[model_label] == "kaggle_xgboost":
        return features

    _, library = build_disease_library()
    covered_symptoms: set[str] = set()
    for disease_info in library.values():
        symptom_table = disease_info["symptom_table"]
        if not symptom_table.empty:
            covered_symptoms.update(symptom_table["symptom"].astype(str).tolist())
    return [feature for feature in features if feature in covered_symptoms]


def infer_body_regions(symptom: str) -> list[str]:
    normalized = normalize_text(symptom)
    regions = [
        region
        for region, keywords in BODY_REGION_KEYWORDS.items()
        if any(keyword in normalized for keyword in keywords)
    ]
    return regions or ["General / Whole body"]


def build_body_region_map(symptoms: list[str]) -> dict[str, list[str]]:
    symptom_order = {symptom: idx for idx, symptom in enumerate(symptoms)}
    region_map: dict[str, list[str]] = {}
    for symptom in symptoms:
        for region in infer_body_regions(symptom):
            region_map.setdefault(region, []).append(symptom)

    return {
        region: sorted(region_symptoms, key=lambda symptom: symptom_order[symptom])
        for region, region_symptoms in region_map.items()
    }


def body_region_color(region: str, selected_regions: set[str]) -> str:
    return "#0f766e" if region in selected_regions else "#cfd7df"


def render_body_map_svg(selected_regions: list[str]) -> None:
    selected = set(selected_regions)
    general = "#e7eef3" if not selected else "#dce7ec"
    skin_outline = "#0f766e" if "Skin" in selected else "#8a96a3"
    svg = f"""
    <div style="display:flex; justify-content:center; padding: 8px 0 4px;">
      <svg viewBox="0 0 260 420" width="100%" style="max-width:260px;">
        <rect x="0" y="0" width="260" height="420" rx="18" fill="#111827" opacity="0.02"/>
        <circle cx="130" cy="54" r="34" fill="{body_region_color('Head / Brain', selected)}" stroke="{skin_outline}" stroke-width="3"/>
        <circle cx="116" cy="49" r="4" fill="{body_region_color('Eyes', selected)}"/>
        <circle cx="144" cy="49" r="4" fill="{body_region_color('Eyes', selected)}"/>
        <path d="M102 56 Q130 76 158 56" fill="none" stroke="{body_region_color('Ear / Nose / Throat', selected)}" stroke-width="6" stroke-linecap="round"/>
        <rect x="92" y="91" width="76" height="82" rx="24" fill="{body_region_color('Chest / Breathing', selected)}" stroke="{skin_outline}" stroke-width="3"/>
        <rect x="94" y="160" width="72" height="82" rx="22" fill="{body_region_color('Abdomen / Digestive', selected)}" stroke="{skin_outline}" stroke-width="3"/>
        <rect x="104" y="226" width="52" height="44" rx="16" fill="{body_region_color('Urinary / Reproductive', selected)}" stroke="{skin_outline}" stroke-width="3"/>
        <path d="M88 107 C58 126 43 163 37 213" fill="none" stroke="{body_region_color('Arms / Hands', selected)}" stroke-width="24" stroke-linecap="round"/>
        <path d="M172 107 C202 126 217 163 223 213" fill="none" stroke="{body_region_color('Arms / Hands', selected)}" stroke-width="24" stroke-linecap="round"/>
        <circle cx="35" cy="224" r="15" fill="{body_region_color('Arms / Hands', selected)}" stroke="{skin_outline}" stroke-width="3"/>
        <circle cx="225" cy="224" r="15" fill="{body_region_color('Arms / Hands', selected)}" stroke="{skin_outline}" stroke-width="3"/>
        <path d="M112 266 C104 312 99 352 91 392" fill="none" stroke="{body_region_color('Legs / Feet', selected)}" stroke-width="26" stroke-linecap="round"/>
        <path d="M148 266 C156 312 161 352 169 392" fill="none" stroke="{body_region_color('Legs / Feet', selected)}" stroke-width="26" stroke-linecap="round"/>
        <path d="M93 104 C83 149 83 197 91 243" fill="none" stroke="{body_region_color('Back / Spine', selected)}" stroke-width="7" stroke-linecap="round"/>
        <path d="M167 104 C177 149 177 197 169 243" fill="none" stroke="{body_region_color('Back / Spine', selected)}" stroke-width="7" stroke-linecap="round"/>
        <circle cx="130" cy="86" r="9" fill="{general}" stroke="{skin_outline}" stroke-width="2"/>
        <text x="130" y="407" text-anchor="middle" font-size="13" fill="#9ca3af">Choose an area</text>
      </svg>
    </div>
    """
    st.markdown(svg, unsafe_allow_html=True)


def render_body_map_selector(
    region_options: list[str],
    region_map: dict[str, list[str]],
    state_key: str,
) -> list[str]:
    current = st.session_state.get(state_key, [])
    if not isinstance(current, list):
        current = []
    current = [region for region in current if region in region_options]
    st.session_state[state_key] = current

    st.markdown("**Choose body areas**")
    region_set = set(region_options)

    def toggle_region(region: str) -> None:
        selected = list(st.session_state[state_key])
        if region in selected:
            selected.remove(region)
        else:
            selected.append(region)
        st.session_state[state_key] = selected
        st.rerun()

    def region_button(region: str) -> None:
        if region not in region_set:
            st.empty()
            return
        is_selected = region in st.session_state[state_key]
        label = f"{region}\n{len(region_map[region])} symptoms"
        if st.button(
            label,
            key=f"{state_key}_{normalize_text(region).replace(' ', '_').replace('/', '_')}",
            type="primary" if is_selected else "secondary",
            use_container_width=True,
        ):
            toggle_region(region)

    map_col, preview_col = st.columns([0.6, 0.4], vertical_alignment="center")
    with map_col:
        row = st.columns([1, 1, 1])
        with row[1]:
            region_button("General / Whole body")

        row = st.columns([1, 1, 1])
        with row[0]:
            region_button("Ear / Nose / Throat")
        with row[1]:
            region_button("Head / Brain")
        with row[2]:
            region_button("Eyes")

        row = st.columns([1, 1, 1])
        with row[0]:
            region_button("Arms / Hands")
        with row[1]:
            region_button("Chest / Breathing")
        with row[2]:
            region_button("Skin")

        row = st.columns([1, 1, 1])
        with row[0]:
            region_button("Back / Spine")
        with row[1]:
            region_button("Abdomen / Digestive")
        with row[2]:
            region_button("Urinary / Reproductive")

        row = st.columns([1, 1, 1])
        with row[1]:
            region_button("Legs / Feet")

        other_regions = [region for region in region_options if region not in BODY_REGION_KEYWORDS]
        if other_regions:
            st.markdown("**Other areas**")
            for region in other_regions:
                region_button(region)

    with preview_col:
        render_body_map_svg(st.session_state[state_key])
        if st.session_state[state_key]:
            st.caption("Chosen areas: " + ", ".join(st.session_state[state_key]))
        else:
            st.caption("No area chosen yet. Showing all symptoms.")
        quick_cols = st.columns(2)
        if quick_cols[0].button("Show all areas", key=f"{state_key}_all", use_container_width=True):
            st.session_state[state_key] = region_options
            st.rerun()
        if quick_cols[1].button("Clear areas", key=f"{state_key}_clear", use_container_width=True):
            st.session_state[state_key] = []
            st.rerun()

    return st.session_state[state_key]


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
            model="gpt-5.4",
            messages=[
                {"role": "system", "content": "Return only a JSON array of strings."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or "[]"
        extracted = json.loads(content)
        if not isinstance(extracted, list):
            return [], "The symptom reader returned an unexpected response."

        cleaned = [item for item in extracted if item in valid_features]
        return sorted(set(cleaned), key=valid_features.index), None
    except Exception as exc:
        return [], f"Symptom reading failed: {exc}"


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
                border: 1px solid rgba(255, 45, 85, 0.18);
                background: linear-gradient(180deg, #ffffff 0%, #fff4f7 100%);
                padding: 16px 18px;
                margin-bottom: 12px;
                border-radius: 8px;
            ">
                <div style="font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: #8a5f6c;">
                    Suggestion {rank}
                </div>
                <div style="font-size: 24px; color: #1d1a17; margin-top: 6px;">
                    {disease}
                </div>
                <div style="font-size: 16px; color: #ff2d55; margin-top: 8px;">
                    Match score: {confidence:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_model_snapshot(model_label: str) -> None:
    resources = get_model_resources(model_label)
    covered_symptoms = get_covered_symptoms_for_model(model_label, resources["features"])

    st.caption(resources["mode_note"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Diseases Covered", len(resources["encoder"].classes_))
    m2.metric("Symptoms Available", len(resources["features"]))
    m3.metric("Symptoms You Can Choose", len(covered_symptoms))

    if MODEL_OPTIONS[model_label] == "kaggle_xgboost":
        st.info("This option can suggest from a wider list of possible diseases.")
    else:
        st.info("This option uses a smaller disease list shared between Kaggle and DDXPlus.")


def render_prediction_result(model_label: str, selected_symptoms: list[str]) -> None:
    resources = get_model_resources(model_label)
    if not selected_symptoms:
        st.error("Please add at least one symptom first.")
        return

    predictions = predict_top_k(
        resources["model"],
        resources["encoder"],
        resources["features"],
        selected_symptoms,
    )
    render_prediction_cards(predictions)
    st.markdown("**Symptoms used**")
    st.write(", ".join(selected_symptoms))


def render_symptom_checkbox_grid(
    symptom_options: list[str],
    selected_key: str,
    all_symptoms: list[str],
) -> list[str]:
    if selected_key not in st.session_state:
        st.session_state[selected_key] = []

    selected_set = {
        symptom
        for symptom in st.session_state.get(selected_key, [])
        if symptom in all_symptoms
    }
    st.session_state[selected_key] = [symptom for symptom in all_symptoms if symptom in selected_set]

    st.markdown("**Choose symptoms**")
    search_text = st.text_input(
        "Search symptoms",
        placeholder="Search symptoms to tick...",
        key=f"{selected_key}_search",
    )
    visible_symptoms = [
        symptom for symptom in symptom_options
        if search_text.lower() in symptom.lower()
    ]

    if not visible_symptoms:
        st.info("No symptoms matched your search.")
        return st.session_state[selected_key]

    st.caption(
        f"Showing {len(visible_symptoms)} symptoms from the selected area. "
        f"{len(st.session_state[selected_key])} chosen."
    )

    symptom_index = {symptom: index for index, symptom in enumerate(all_symptoms)}
    checkbox_columns = st.columns(3)

    def sync_symptom(symptom: str, checkbox_key: str) -> None:
        current = set(st.session_state.get(selected_key, []))
        if st.session_state.get(checkbox_key, False):
            current.add(symptom)
        else:
            current.discard(symptom)
        st.session_state[selected_key] = [item for item in all_symptoms if item in current]

    for position, symptom in enumerate(visible_symptoms):
        original_index = symptom_index[symptom]
        checkbox_key = f"{selected_key}_check_{original_index}_{normalize_text(symptom).replace(' ', '_')}"
        if checkbox_key not in st.session_state:
            st.session_state[checkbox_key] = symptom in selected_set
        with checkbox_columns[position % len(checkbox_columns)]:
            st.checkbox(
                symptom,
                key=checkbox_key,
                on_change=sync_symptom,
                args=(symptom, checkbox_key),
            )

    selected_symptoms = st.session_state[selected_key]
    if selected_symptoms:
        st.markdown("**Chosen symptoms**")
        st.write(", ".join(selected_symptoms))
        if st.button("Clear chosen symptoms", key=f"{selected_key}_clear_selected"):
            st.session_state[selected_key] = []
            for key in list(st.session_state.keys()):
                if str(key).startswith(f"{selected_key}_check_"):
                    del st.session_state[key]
            st.rerun()
    else:
        st.info("No symptoms chosen yet.")

    return selected_symptoms


def render_light_table(df: pd.DataFrame, max_rows: int = 14) -> None:
    if df.empty:
        st.info("No rows to display.")
        return

    display_df = df.head(max_rows).copy()
    friendly_columns = {
        "symptom": "Symptom",
        "source": "Shown In",
        "kaggle_rows": "Kaggle Examples",
        "ddxplus_patterns": "DDXPlus Examples",
    }
    display_df.columns = [
        friendly_columns.get(str(column), str(column).replace("_", " ").title())
        for column in display_df.columns
    ]

    header_html = "".join(f"<th>{html.escape(str(column))}</th>" for column in display_df.columns)
    body_rows = []
    for _, row in display_df.iterrows():
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row.tolist())
        body_rows.append(f"<tr>{cells}</tr>")

    remainder = max(len(df) - max_rows, 0)
    footer = f"<div class=\"light-table-footer\">Showing {len(display_df)} of {len(df)} items</div>" if remainder else ""

    st.markdown(
        f"""
        <div class="light-table-wrap">
          <table class="light-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
          </table>
          {footer}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_coverage_summary() -> None:
    features, kaggle_encoder, _ = load_base_artifacts()
    ddx_metrics = load_ddxplus_comparison_metrics()
    ddx_active_symptoms = len(get_covered_symptoms_for_model("Kaggle + DDXPlus XGBoost", features))

    kaggle_col, ddx_col = st.columns(2)
    with kaggle_col:
        st.markdown(
            f"""
            <div class="coverage-card">
              <div class="coverage-card-title">Broad Checker</div>
              <div class="coverage-card-subtitle">Covers more possible diseases</div>
              <div class="coverage-card-grid">
                <div><span>Diseases Covered</span><strong>{len(kaggle_encoder.classes_)}</strong></div>
                <div><span>Symptoms Available</span><strong>{len(features)}</strong></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with ddx_col:
        st.markdown(
            f"""
            <div class="coverage-card">
              <div class="coverage-card-title">Focused Checker</div>
              <div class="coverage-card-subtitle">Uses a smaller shared disease list</div>
              <div class="coverage-card-grid">
                <div><span>Diseases Covered</span><strong>{ddx_metrics["shared_disease_count"]}</strong></div>
                <div><span>Symptoms Available</span><strong>{ddx_active_symptoms}</strong></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_coverage_page() -> None:
    st.header("What This App Covers")
    st.caption("See which diseases and symptoms each checker can work with.")
    render_coverage_summary()

    st.divider()
    st.markdown("**Choose a checker**")
    if "coverage_model" not in st.session_state:
        st.session_state["coverage_model"] = DEFAULT_MODEL_LABEL

    model_cols = st.columns(2)
    with model_cols[0]:
        if st.button(
            "Broad Checker\nMore possible diseases",
            key="coverage_model_kaggle",
            type="primary" if st.session_state["coverage_model"] == "Kaggle-only XGBoost" else "secondary",
            use_container_width=True,
        ):
            st.session_state["coverage_model"] = "Kaggle-only XGBoost"
            st.rerun()

    with model_cols[1]:
        if st.button(
            "Focused Checker\nSmaller shared disease list",
            key="coverage_model_ddxplus",
            type="primary" if st.session_state["coverage_model"] == "Kaggle + DDXPlus XGBoost" else "secondary",
            use_container_width=True,
        ):
            st.session_state["coverage_model"] = "Kaggle + DDXPlus XGBoost"
            st.rerun()

    model_label = st.session_state["coverage_model"]
    diseases, library, scope_label = get_coverage_for_model(model_label)

    st.markdown('<div class="coverage-section">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Diseases Covered")
    with col2:
        st.markdown("### Symptoms Covered")

    disease_col, symptom_panel_col = st.columns(2)
    with disease_col:
        search_term = st.text_input("Search disease", placeholder="Search diseases...", key="coverage_search")
    filtered_diseases = [disease for disease in diseases if search_term.lower() in disease.lower()]
    if not filtered_diseases:
        st.warning("No diseases matched your search.")
        return

    with disease_col:
        selected_disease = st.selectbox("Choose a disease", options=filtered_diseases, key="coverage_disease")
        st.caption(f"Showing {len(filtered_diseases)} of {len(diseases)} diseases")

    disease_info = library[selected_disease]
    symptom_table = disease_info["symptom_table"]

    with disease_col:
        m1, m2 = st.columns(2)
        m1.metric("Diseases Covered", len(diseases))
        m2.metric("Training Examples", f"{int(disease_info['kaggle_rows']):,}")

        if "ddxplus_patterns" in disease_info:
            st.metric("DDXPlus Examples", f"{int(disease_info['ddxplus_patterns']):,}")

    if symptom_table.empty:
        st.warning("No symptoms are listed for this disease.")
        return

    with symptom_panel_col:
        if "source" in symptom_table.columns:
            both_count = int((symptom_table["source"] == "Both").sum())
            kaggle_only_count = int((symptom_table["source"] == "Kaggle").sum())
            ddxplus_only_count = int((symptom_table["source"] == "DDXPlus").sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("In Both", both_count)
            c2.metric("Kaggle", kaggle_only_count)
            c3.metric("DDXPlus", ddxplus_only_count)

            source_filter = st.multiselect(
                "Show symptoms from",
                options=["Both", "Kaggle", "DDXPlus"],
                default=["Both", "Kaggle", "DDXPlus"],
                key="coverage_source_filter",
            )
            symptom_table = symptom_table.loc[symptom_table["source"].isin(source_filter)].reset_index(drop=True)

    if symptom_table.empty:
        st.warning("No symptoms match the selected source.")
        return

    with symptom_panel_col:
        symptom_search = st.text_input("Search symptom", placeholder="Search symptoms...", key="coverage_symptom_search")
    filtered_symptoms = symptom_table.loc[
        symptom_table["symptom"].str.contains(symptom_search, case=False, na=False)
    ].reset_index(drop=True)
    if filtered_symptoms.empty:
        st.warning("No symptoms matched your search.")
        return

    with symptom_panel_col:
        selected_symptom = st.selectbox("Choose a symptom", options=filtered_symptoms["symptom"].tolist(), key="coverage_symptom")
    selected_symptom_row = filtered_symptoms.loc[filtered_symptoms["symptom"] == selected_symptom].iloc[0]

    with symptom_panel_col:
        s1, s2 = st.columns(2)
        s1.metric("Symptoms for This Disease", len(symptom_table))
        s2.metric("Examples With This Symptom", f"{int(selected_symptom_row.get('kaggle_rows', 0)):,}")

        render_light_table(filtered_symptoms)
    st.markdown("</div>", unsafe_allow_html=True)


def render_llm_chat_page() -> None:
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = DEFAULT_MODEL_LABEL

    model_label = st.session_state["llm_model"]
    resources = get_model_resources(model_label)
    mode_note = resources["mode_note"]

    st.markdown(
        """
        <style>
        .llm-shell {
            min-height: 46vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            gap: 18px;
        }
        .llm-title {
            font-size: 30px;
            font-weight: 600;
            text-align: center;
            color: var(--main-text);
        }
        .llm-model-chip {
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 999px;
            padding: 6px 14px;
            color: var(--main-text);
            font-size: 13px;
            text-align: center;
            opacity: 0.82;
        }
        .llm-disclaimer {
            color: var(--main-text);
            font-size: 13px;
            text-align: center;
            max-width: 760px;
            opacity: 0.72;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="llm-shell">
          <div class="llm-title">Where do you feel unwell today?</div>
          <div class="llm-model-chip">{MODEL_USER_LABELS[model_label]} | {mode_note}</div>
          <div class="llm-disclaimer">
            Describe how you feel. The app will match your words to known symptoms and suggest possible diseases.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_cols = st.columns([0.14, 0.66, 0.2], vertical_alignment="bottom")
    with input_cols[0]:
        if st.button(":material/add: Option", key="llm_model_toggle", use_container_width=True):
            st.session_state["llm_model_menu_open"] = not st.session_state.get("llm_model_menu_open", False)
            st.rerun()

    with input_cols[1]:
        user_text = st.text_input(
            "Symptom description",
            placeholder="Describe your symptoms...",
            label_visibility="collapsed",
            key=f"llm_text_{MODEL_OPTIONS[model_label]}",
        )

    with input_cols[2]:
        submit = st.button("Send", type="primary", use_container_width=True)

    if st.session_state.get("llm_model_menu_open", False):
        menu_cols = st.columns([0.08, 0.55, 0.37])
        with menu_cols[1]:
            selected_model = st.radio(
                "Choose checker",
                options=list(MODEL_OPTIONS.keys()),
                index=list(MODEL_OPTIONS.keys()).index(model_label),
                format_func=lambda option: MODEL_USER_LABELS[option],
                key="llm_model_picker",
                horizontal=True,
            )
            if selected_model != st.session_state["llm_model"]:
                st.session_state["llm_model"] = selected_model
                st.session_state["llm_model_menu_open"] = False
                st.rerun()
            st.caption("Broad Checker covers more diseases. Focused Checker uses a smaller shared list.")
    if submit:
        if not user_text.strip():
            st.warning("Please describe your symptoms first.")
            return

        extracted, error_message = extract_symptoms_from_text(user_text, resources["features"])
        if error_message:
            st.error(error_message)
            return
        if not extracted:
            st.warning("I could not match your description to the symptom list. Try simpler symptom words.")
            return

        st.session_state["llm_last_result"] = {
            "model_label": model_label,
            "user_text": user_text,
            "extracted": extracted,
        }

    last_result = st.session_state.get("llm_last_result")
    if last_result:
        st.divider()
        st.chat_message("user").write(last_result["user_text"])
        with st.chat_message("assistant"):
            st.markdown("**Symptoms found**")
            st.write(", ".join(last_result["extracted"]))
            render_prediction_result(last_result["model_label"], last_result["extracted"])


def render_manual_check_page() -> None:
    st.header("Manual Symptom Check")
    st.caption("Pick where you feel discomfort, then choose symptoms to check possible diseases.")

    model_label = st.selectbox(
        "Choose checker",
        options=list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index(DEFAULT_MODEL_LABEL),
        format_func=lambda option: MODEL_USER_LABELS[option],
        key="manual_model",
    )
    resources = get_model_resources(model_label)
    render_model_snapshot(model_label)

    covered_symptoms = get_covered_symptoms_for_model(model_label, resources["features"])
    region_map = build_body_region_map(covered_symptoms)
    region_options = [region for region in BODY_REGION_KEYWORDS if region in region_map]
    extra_regions = sorted(region for region in region_map if region not in BODY_REGION_KEYWORDS)
    region_options.extend(extra_regions)

    st.markdown("**Where do you feel it?**")
    st.caption("Choose one or more body areas to narrow the symptom list.")
    selected_regions = render_body_map_selector(
        region_options,
        region_map,
        state_key=f"manual_body_regions_{MODEL_OPTIONS[model_label]}",
    )

    if selected_regions:
        symptom_options = sorted(
            {symptom for region in selected_regions for symptom in region_map[region]},
            key=covered_symptoms.index,
        )
    else:
        symptom_options = covered_symptoms

    c1, c2, c3 = st.columns(3)
    c1.metric("Body Areas Available", len(region_options))
    c2.metric("Symptoms Available", len(covered_symptoms))
    c3.metric("Symptoms Shown", len(symptom_options))

    symptom_key = f"manual_symptoms_{MODEL_OPTIONS[model_label]}"
    selected_symptoms = render_symptom_checkbox_grid(
        symptom_options=symptom_options,
        selected_key=symptom_key,
        all_symptoms=covered_symptoms,
    )

    if st.button("Check Possible Diseases", type="primary", use_container_width=True):
        render_prediction_result(model_label, selected_symptoms)


def apply_app_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #fff7fa;
            --sidebar-bg: #ffffff;
            --panel-bg: #ffffff;
            --panel-border: rgba(255, 45, 85, 0.16);
            --muted-text: #7f6b73;
            --main-text: #23151b;
            --accent-red: #ff2d55;
            --accent-pink: #ff6b9a;
            --accent-rose: #fb7185;
        }

        .stApp {
            background: var(--app-bg);
            color: var(--main-text);
        }

        .stMarkdown,
        .stMarkdown p,
        h1, h2, h3, h4,
        label,
        [data-testid="stCaptionContainer"] {
            color: var(--main-text);
        }

        [data-testid="stHeader"] {
            background: rgba(255, 247, 250, 0);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 82% 10%, rgba(255, 45, 85, 0.15), transparent 28%),
                radial-gradient(circle at 15% 74%, rgba(255, 107, 154, 0.12), transparent 30%),
                linear-gradient(180deg, #fff7fa 0%, #fff1f5 54%, #fff8fb 100%);
        }

        [data-testid="stSidebar"] {
            background: var(--sidebar-bg);
            border-right: 1px solid rgba(255, 45, 85, 0.12);
            box-shadow: 12px 0 32px rgba(255, 45, 85, 0.06);
        }

        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.75rem;
        }

        [data-testid="stSidebar"] .stButton > button {
            justify-content: flex-start;
            border-radius: 14px;
            border: 1px solid transparent;
            background: transparent;
            color: var(--muted-text);
            font-weight: 700;
            padding: 0.8rem 1rem;
            min-height: 48px;
        }

        .stButton > button[kind="secondary"],
        [data-testid="stBaseButton-secondary"],
        [data-testid="stPopover"] > button {
            background: #ffffff !important;
            border: 1px solid rgba(255, 45, 85, 0.18) !important;
            border-radius: 14px !important;
            color: var(--main-text) !important;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: rgba(251, 113, 133, 0.3);
            color: var(--main-text);
            background: rgba(255, 45, 85, 0.06);
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"],
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
            background: linear-gradient(100deg, #ff2d55 0%, #ff6b9a 58%, #c026d3 100%);
            color: #ffffff;
            border: 0;
            box-shadow: 0 14px 30px rgba(255, 45, 85, 0.3);
        }

        [data-testid="stSidebar"] .stButton > button[kind="secondary"],
        [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
            background: transparent !important;
            border-color: transparent !important;
            color: var(--muted-text) !important;
        }

        .stButton > button[kind="primary"],
        [data-testid="stBaseButton-primary"] {
            background: linear-gradient(100deg, #ff2d55 0%, #ff6b9a 100%);
            border: 0;
            color: white;
            font-weight: 700;
        }

        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            background-color: #ffffff;
            border-color: rgba(255, 45, 85, 0.22);
            color: var(--main-text);
        }

        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {
            color: #9f8b93;
            opacity: 1;
        }

        [data-testid="stPopover"] > button {
            font-size: 22px !important;
            min-height: 42px;
        }

        [data-testid="stPopover"] > button:hover {
            background: #fff0f4;
            border-color: rgba(255, 45, 85, 0.5);
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid var(--panel-border);
            border-radius: 14px;
            padding: 0.85rem 1rem;
        }

        [data-testid="stMetric"] label,
        [data-testid="stMetric"] [data-testid="stMetricLabel"],
        [data-testid="stMetric"] [data-testid="stMetricValue"],
        [data-testid="stMetric"] div {
            color: var(--main-text) !important;
        }

        [data-testid="stWidgetLabel"] p,
        [data-testid="stSelectbox"] label,
        [data-testid="stTextInput"] label,
        [data-testid="stMultiSelect"] label {
            color: #5f4b54 !important;
            opacity: 1 !important;
        }

        .stRadio label,
        .stRadio p {
            color: var(--main-text) !important;
        }

        .brand-shell {
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 22px 8px 28px 8px;
            border-bottom: 1px solid rgba(255, 45, 85, 0.12);
            margin-bottom: 22px;
        }

        .brand-logo {
            width: 48px;
            height: 48px;
            border-radius: 16px;
            display: grid;
            place-items: center;
            background: linear-gradient(135deg, #ff2d55, #ff6b9a);
            color: white;
            font-size: 28px;
            font-weight: 800;
            box-shadow: 0 14px 30px rgba(255, 45, 85, 0.26);
        }

        .brand-title {
            color: var(--main-text);
            font-size: 27px;
            font-weight: 800;
            line-height: 1.08;
            letter-spacing: 0;
        }

        .brand-subtitle {
            margin-top: 8px;
            color: var(--muted-text);
            font-size: 14px;
        }

        .sidebar-warning {
            margin-top: 34vh;
            border: 1px solid rgba(245, 158, 11, 0.34);
            background: #fff8db;
            border-radius: 16px;
            padding: 16px 18px;
            color: #7c4a03;
            font-size: 14px;
            line-height: 1.35;
        }

        .sidebar-warning strong {
            display: block;
            color: #b45309;
            margin-bottom: 4px;
        }

        .llm-input-wrap {
            position: sticky;
            bottom: 0;
            background: linear-gradient(180deg, rgba(255, 247, 250, 0), rgba(255, 247, 250, 0.96) 28%);
            padding-top: 32px;
            padding-bottom: 18px;
            z-index: 5;
        }

        .coverage-card {
            background: linear-gradient(145deg, #ffffff, #fff4f7);
            border: 1px solid rgba(251, 113, 133, 0.22);
            border-radius: 18px;
            padding: 24px 28px;
            min-height: 128px;
            box-shadow: 0 18px 44px rgba(255, 45, 85, 0.08);
        }

        .coverage-card-title {
            color: var(--main-text);
            font-size: 20px;
            font-weight: 800;
            text-align: center;
        }

        .coverage-card-subtitle {
            color: #d91f4f;
            font-size: 15px;
            font-weight: 600;
            text-align: center;
            margin-top: 6px;
        }

        .coverage-card-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 22px;
        }

        .coverage-card-grid div {
            border: 1px solid rgba(251, 113, 133, 0.14);
            background: #fff9fb;
            border-radius: 14px;
            padding: 14px;
        }

        .coverage-card-grid span {
            display: block;
            color: var(--muted-text);
            font-size: 13px;
        }

        .coverage-card-grid strong {
            display: block;
            color: var(--main-text);
            font-size: 30px;
            margin-top: 4px;
        }

        .coverage-section {
            margin-top: 22px;
        }

        .light-table-wrap {
            width: 100%;
            border: 1px solid rgba(255, 45, 85, 0.16);
            border-radius: 14px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 14px 28px rgba(255, 45, 85, 0.06);
        }

        .light-table {
            width: 100%;
            border-collapse: collapse;
            color: var(--main-text);
            font-size: 14px;
        }

        .light-table th {
            text-align: left;
            padding: 13px 14px;
            background: #fff1f5;
            color: #7f1d3a;
            font-weight: 800;
            border-bottom: 1px solid rgba(255, 45, 85, 0.14);
        }

        .light-table td {
            padding: 12px 14px;
            border-bottom: 1px solid rgba(255, 45, 85, 0.08);
            background: rgba(255, 255, 255, 0.72);
        }

        .light-table tr:nth-child(even) td {
            background: #fff8fb;
        }

        .light-table td:last-child,
        .light-table th:last-child {
            text-align: right;
        }

        .light-table-footer {
            padding: 10px 14px;
            color: #7f6b73;
            background: #fff8fb;
            font-size: 13px;
            border-top: 1px solid rgba(255, 45, 85, 0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "LLM Chat"

    st.sidebar.markdown(
        """
        <div class="brand-shell">
          <div class="brand-logo">+</div>
          <div>
            <div class="brand-title">AI Symptom<br/>Checker</div>
            <div class="brand-subtitle">Possible Disease Suggestions</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for page_option in PAGE_OPTIONS:
        is_active = st.session_state["active_page"] == page_option
        label = f"{NAV_ICONS[page_option]}  {NAV_LABELS[page_option]}"
        if st.sidebar.button(label, key=f"nav_{page_option}", type="primary" if is_active else "secondary", use_container_width=True):
            st.session_state["active_page"] = page_option
            st.rerun()

    st.sidebar.markdown(
        """
        <div class="sidebar-warning">
          <strong>Prototype Only</strong>
          This tool does not provide a formal medical diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.session_state["active_page"]


def main() -> None:
    st.set_page_config(page_title="AI Symptom Checker", layout="wide")
    apply_app_theme()

    try:
        load_base_artifacts()
        load_ddxplus_alignment_rows()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    page = render_sidebar()

    if page == "Model Coverage":
        render_coverage_page()
    elif page == "LLM Chat":
        render_llm_chat_page()
    elif page == "Manual Check":
        render_manual_check_page()


if __name__ == "__main__":
    main()
