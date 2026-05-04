# FYP PIPELINE REDESIGN

## Current structure
- `0. Data Preparation`
- `1. Kaggle_Training`
- `2. External_Validation`
- `3. Prototype`
- `Artifacts`
- `shared`

## Kaggle training
Put `Kaggle.csv` into `0. Data Preparation/input/`, then run:

```bash
python "0. Data Preparation/01_prepare_kaggle_data.py"
cd "1. Kaggle_Training"
python 02_train_xgboost.py
python 03_train_lightgbm.py
python 04_train_randomforest.py
```

`0. Data Preparation/01_prepare_kaggle_data.py` handles cleaning, label encoding, singleton handling, and train/test/CV split creation.

Prepared training inputs are written directly into `1. Kaggle_Training/input/`.

`02/03/04_train_*.py` each train one model only.

## External validation
Run:

```bash
python "2. External_Validation/step2_external_validation.py"
```

This step:
- downloads or reuses the Hugging Face dataset `QuyenAnhDE/Diseases_Symptoms`
- aligns external disease/symptom text to the Kaggle-trained feature and label space
- evaluates the selected XGBoost model externally
- writes one workbook to `2. External_Validation/output/external_validation_report.xlsx`
