# Credit Card Fraud Detection Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-0.11.0-red.svg)](https://imbalanced-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning project designed to detect fraudulent credit card transactions in heavily imbalanced financial data. This project features a modular 8-stage Jupyter pipeline alongside a standalone production training script (`main.py`) with feature scaling, SMOTE class balancing, multi-model benchmarking, financial cost evaluation, and model serialization.

---

## Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Dataset Details](#dataset-details)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Technologies & Libraries](#technologies--libraries)
- [Setup & Installation](#setup--installation)
- [Usage Instructions](#usage-instructions)
- [Model Evaluation & Financial Impact](#model-evaluation--financial-impact)
- [Deployment & Inference](#deployment--inference)
- [Author & Acknowledgments](#author--acknowledgments)

---

## Project Overview

Credit card fraud presents a significant challenge for financial institutions, characterized by extreme class imbalance where fraudulent transactions represent less than **0.2%** of total activity. Standard accuracy metrics fail in this domain because a naive model predicting all transactions as legitimate would achieve over 99.8% accuracy while missing 100% of frauds.

This project addresses these challenges by:
1. **Handling Severe Imbalance:** Applying Synthetic Minority Over-sampling Technique (**SMOTE**) on training partitions to prevent data leakage.
2. **Robust Preprocessing:** Using `RobustScaler` for transaction amounts (resilient to heavy outliers) and `StandardScaler` for timestamps.
3. **Multi-Model Comparison:** Benchmarking **Logistic Regression**, **Decision Trees**, and **K-Nearest Neighbors (KNN)** across Precision, Recall, F1-score, and ROC-AUC.
4. **Financial Impact Modeling:** Translating confusion matrix counts into concrete monetary risk metrics (false negative fraud losses vs false positive friction costs).
5. **Production Readiness:** Saving trained artifacts (`models/fraud_detection_model.pkl` and `scalers.pkl`) with a reusable inference function.

---

## System Architecture

```
Raw Data (creditcard.csv)
       │
       ▼
Data Cleaning & EDA (01_Data_Exploration.ipynb)
       │
       ▼
Feature Scaling: RobustScaler (Amount) + StandardScaler (Time)
       │
       ▼
Stratified Train/Test Split (70% Train / 30% Test)
       │
       ▼
SMOTE Resampling on Training Set (sampling_strategy=0.30)
       │
 ┌─────┴───────────────────┬────────────────────────┐
 ▼                         ▼                        ▼
Logistic Regression   Decision Tree Classifier    KNN Classifier
 └─────┬───────────────────┴────────────────────────┘
       │
       ▼
Model Evaluation & Benchmarking (Precision, Recall, F1, ROC-AUC)
       │
       ▼
Financial Risk Impact Analysis (Misclassification Cost Matrix)
       │
       ▼
Artifact Serialization -> models/fraud_detection_model.pkl + scalers.pkl
```

---

## Repository Structure

```text
CREDITCARD-FRAUD-DETECTION-PROJECT/
├── 01_Data_Exploration.ipynb       # Exploratory analysis, distributions & correlation heatmaps
├── 02_Feature_Engineering.ipynb     # Scaling, outlier management & SMOTE class balancing
├── 03_Logistic_Regression.ipynb     # Baseline linear classification & hyperparameter tuning
├── 04_Decision_Tree.ipynb           # Non-linear tree classification with tree pruning (depth=10)
├── 05_KNN_Classifier.ipynb          # Distance-based classification (k=5)
├── 06_Model_Comparison.ipynb        # Side-by-side performance metrics & ROC curves
├── 07_Model_Evaluation.ipynb        # Confusion matrix, Precision-Recall curve & financial impact
├── 08_Model_Deployment.ipynb        # Model loading, sample prediction & inference pipeline
├── main.py                         # Complete standalone end-to-end training pipeline script
├── utils.py                        # Reusable helper utilities (plotting, metrics, cost analysis)
├── REQUIREMENTS                    # Dependency specifications with exact versions
├── .gitignore                      # Git ignore patterns for datasets and cache files
└── README.md                       # Comprehensive project documentation
```

---

## Dataset Details

The project utilizes the benchmark [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset containing European cardholder transactions from September 2013:

- **Total Transactions:** 284,807 records
- **Legitimate Transactions:** 284,315 (99.828%)
- **Fraudulent Transactions:** 492 (0.172%)
- **Features:**
  - `Time`: Elapsed seconds between each transaction and the first transaction in the dataset.
  - `Amount`: Transaction transaction amount in EUR (heavily right-skewed).
  - `V1` to `V28`: 28 principal components obtained via Principal Component Analysis (PCA) for privacy preservation.
  - `Class`: Binary ground truth target variable (`1` for fraud, `0` for legitimate).

> **Note:** To run the project locally, download `creditcard.csv` from Kaggle and place it in the project root directory.

---

## Machine Learning Workflow

### 1. Preprocessing & Scaling
- `Amount` is transformed using `RobustScaler` (median and IQR) to reduce the influence of extreme outlier purchase amounts.
- `Time` is transformed using `StandardScaler`.
- Features `V1` through `V28` are retained in their existing normalized PCA representations.

### 2. Handling Imbalance
- The data is partitioned using a **Stratified 70/30 Train-Test Split** (`random_state=42`) to maintain equal fraud ratios in both subsets.
- **SMOTE** is fitted strictly on `X_train` to synthesize minority class samples up to a 30% ratio (`sampling_strategy=0.30`), ensuring **zero data leakage** into `X_test`.

### 3. Model Training & Comparison
Three distinct classifier paradigms are trained and evaluated on the held-out test split:
- **Logistic Regression:** Scaled linear decision boundary (`max_iter=1000`).
- **Decision Tree Classifier:** Non-linear rule-based classification constrained to `max_depth=10` to mitigate overfitting.
- **K-Nearest Neighbors (KNN):** Distance-weighted local neighborhood voting (`n_neighbors=5`).

---

## Technologies & Libraries

| Category | Tools & Libraries |
|---|---|
| **Language** | Python 3.9+ |
| **Data Manipulation** | Pandas, NumPy, SciPy |
| **Machine Learning** | Scikit-Learn (`sklearn`) |
| **Imbalanced Data** | Imbalanced-Learn (`imblearn` SMOTE) |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Serialization** | Pickle, Joblib |
| **Notebook Environment** | Jupyter Notebook, JupyterLab |
| **Deployment / API** | Flask |

---

## Setup & Installation

### Prerequisites
- Python 3.9 or higher installed
- Git installed

### 1. Clone the Repository
```bash
git clone https://github.com/jayakumarjk2007/CREDITCARD-FRAUD-DETECTION-PROJECT.git
cd CREDITCARD-FRAUD-DETECTION-PROJECT
```

### 2. Create and Activate a Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r REQUIREMENTS
```

### 4. Place Dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it directly into the project root directory:
```text
CREDITCARD-FRAUD-DETECTION-PROJECT/
├── creditcard.csv
├── main.py
└── ...
```

---

## Usage Instructions

### Option A: Run the Automated CLI Pipeline
To run the full preprocessing, training, evaluation, and artifact saving pipeline in a single command:
```bash
python main.py
```
This script will:
1. Load and validate `creditcard.csv`.
2. Scale `Amount` and `Time` features.
3. Apply stratified splitting and SMOTE oversampling.
4. Train Logistic Regression, Decision Tree, and KNN.
5. Display a comparative metric summary table.
6. Automatically serialize the top-performing model and scalers to the `models/` directory.

### Option B: Interactive Jupyter Notebook Workflow
Launch Jupyter to explore individual stages step-by-step:
```bash
jupyter notebook
```
Follow the sequential notebook numbered order:
1. `01_Data_Exploration.ipynb`
2. `02_Feature_Engineering.ipynb`
3. `03_Logistic_Regression.ipynb`
4. `04_Decision_Tree.ipynb`
5. `05_KNN_Classifier.ipynb`
6. `06_Model_Comparison.ipynb`
7. `07_Model_Evaluation.ipynb`
8. `08_Model_Deployment.ipynb`

---

## Model Evaluation & Financial Impact

### Metric Comparison
Models are compared using **Precision**, **Recall**, **F1-Score**, and **ROC-AUC** on the unseen test set:

| Model | Precision | Recall | F1-Score | ROC-AUC | Notes |
|---|:---:|:---:|:---:|:---:|---|
| **Logistic Regression** | ~0.87 | ~0.65 | ~0.74 | ~0.97 | High interpretability, rapid inference |
| **Decision Tree (depth=10)** | ~0.78 | ~0.76 | ~0.77 | ~0.89 | Balanced detection, low latency |
| **K-Nearest Neighbors (k=5)** | ~0.85 | ~0.77 | ~0.81 | ~0.92 | Strongest local boundary capture |

*Values reflect performance on held-out stratified test data after SMOTE training.*

### Financial Cost Evaluation
Using the custom cost calculation function in `utils.py`:
- **False Negative (Missed Fraud):** High cost penalty ($10\times$ average transaction value).
- **False Positive (False Alarm):** Low customer friction / SMS verification cost ($0.10\times$ average transaction value).
- **True Positive (Intervention):** Operational manual review cost ($0.10\times$ average transaction value).

This metric ensures the deployed threshold maximizes financial savings rather than solely optimizing statistical accuracy.

---

## Deployment & Inference

The deployment notebook (`08_Model_Deployment.ipynb`) and `main.py` output serialized artifacts for production inference:

```python
import pickle
import pandas as pd

# 1. Load trained model and scalers
with open("models/fraud_detection_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scalers.pkl", "rb") as f:
    scalers = pickle.load(f)

# 2. Predict on an incoming transaction
def predict_fraud(transaction_df):
    tx = transaction_df.copy()
    tx["Amount_Scaled"] = scalers["amount_scaler"].transform(tx[["Amount"]])
    tx["Time_Scaled"] = scalers["time_scaler"].transform(tx[["Time"]])
    features = tx.drop(columns=["Time", "Amount"])
    
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    return {"is_fraud": bool(pred), "fraud_probability": round(float(prob), 4)}
```

---

## Author & Acknowledgments

- **Author:** Jayakumar P
- **GitHub:** [@jayakumarjk2007](https://github.com/jayakumarjk2007)
- **Dataset:** [ULB Machine Learning Group / Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
