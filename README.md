# FYP PIPELINE REDESIGN

This repository contains the implementation pipeline for the dissertation project on an AI symptom checker for disease prediction. The work is organised around four main stages: Kaggle data preparation, Kaggle model training, external validation through aligned datasets, and prototype deployment. A separate appendix-materials folder contains dissertation-supporting evidence files.

## Repository structure

- `0. Data Preparation/`
  - Kaggle preprocessing script
  - raw Kaggle input and preprocessing report output
- `1. Kaggle_Training/`
  - model training scripts for XGBoost, LightGBM, and Random Forest
  - Kaggle-only XGBoost tuning script
  - generated Kaggle model artifacts, prepared inputs, and training reports
- `2. External_Validation/`
  - SymbiPredict and DDXPlus alignment scripts
  - external comparison scripts
  - aligned-dataset reports and comparison workbooks
- `3. Prototype/`
  - Streamlit prototype application
  - local prototype cache assets
- `4. Appendix_Materials/`
  - dissertation appendix evidence files grouped by appendix section
- `shared/`
  - lightweight shared package placeholder

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

The prototype also expects an OpenAI API key in the environment:

```bash
set OPENAI_API_KEY=your_key_here
```

## Stage 0: Kaggle data preparation

Place `Kaggle.csv` in:

```text
0. Data Preparation/input/
```

Then run:

```bash
python "0. Data Preparation/01_prepare_kaggle_data.py"
```

This step:

- cleans and standardises the Kaggle symptom-disease dataset
- removes duplicates and constant features
- removes singleton disease classes
- exports train/test/CV splits into `1. Kaggle_Training/input/`
- writes reusable artifacts into `1. Kaggle_Training/Artifacts_kaggle/`

## Stage 1: Kaggle model training

Run the training scripts from the repository root:

```bash
python "1. Kaggle_Training/02_train_xgboost.py"
python "1. Kaggle_Training/03_train_lightgbm.py"
python "1. Kaggle_Training/04_train_randomforest.py"
python "1. Kaggle_Training/05_tune_xgboost_kaggle_only.py"
```

This stage produces:

- trained model artifacts in `1. Kaggle_Training/Artifacts_kaggle/`
- evaluation workbooks in `1. Kaggle_Training/output/`

## Stage 2: External validation

Run the external validation pipeline scripts from the repository root:

```bash
python "2. External_Validation/align_symbipredict_2022.py"
python "2. External_Validation/02_compare_single_vs_multidataset.py"
python "2. External_Validation/align_ddxplus.py"
python "2. External_Validation/03_compare_kaggle_vs_ddxplus.py"
```

This stage:

- aligns SymbiPredict into the Kaggle disease and feature space
- aligns DDXPlus evidence and disease labels into the Kaggle space
- compares Kaggle-only and multidataset training in aligned disease spaces
- writes alignment and external-validation reports into `2. External_Validation/output/`

## Stage 3: Prototype

Run the Streamlit prototype from the repository root:

```bash
streamlit run "3. Prototype/app.py"
```

The prototype includes:

- a model coverage page
- an LLM chat symptom-entry mode
- a manual symptom selection mode
- support for both the Kaggle-only broad checker and the Kaggle + DDXPlus focused checker

## Appendix materials

The folder `4. Appendix_Materials/` collects dissertation-supporting files used in the appendices:

- `Appendix_A_SymbiPredict/`
- `Appendix_B_DDXPlus/`
- `Appendix_C_LLM_Chat_Testing/`
- `Appendix_D_Manual_Testing/`

These materials are copies of key reports and testing logs grouped for dissertation packaging, so the main pipeline folders can stay focused on implementation work.
