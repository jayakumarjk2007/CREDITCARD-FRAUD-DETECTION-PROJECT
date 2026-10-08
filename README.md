# Credit Card Fraud Detection Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-0.11%2B-red.svg)](https://imbalanced-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end, modular machine learning project designed to detect fraudulent credit card transactions in heavily imbalanced financial data. This project follows the exact curriculum workflow covering data exploration, feature engineering with log transformation, dedicated model training notebooks (**Logistic Regression**, **Decision Trees**, **K-Nearest Neighbors**, and **Random Forest with SMOTE**), multi-model benchmarking, and production deployment.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Files & Modules](#repository-files--modules)
- [Dataset Specifications](#dataset-specifications)
- [Modular Notebook Workflow](#modular-notebook-workflow)
- [Machine Learning Modeling & Imbalance Treatment](#machine-learning-modeling--imbalance-treatment)
- [Model Comparison & Evaluation Results](#model-comparison--evaluation-results)
- [Production Deployment & Real-Time Scoring](#production-deployment--real-time-scoring)
- [Financial Cost Impact & Business Insights](#financial-cost-impact--business-insights)
- [Setup & Installation](#setup--installation)
- [Usage Instructions](#usage-instructions)
- [Author & Acknowledgments](#author--acknowledgments)

---

## Project Overview

Credit card fraud presents extreme class imbalance, where fraudulent transactions account for less than **0.2%** of all transactions. Conventional accuracy metrics are ineffective in this domain, as a naive classifier predicting all transactions as legitimate would achieve 99.83% accuracy while missing 100% of actual fraud.

This project delivers:
1. **Modular Notebook Architecture:** Individual notebooks dedicated to each stage of the data science lifecycle.
2. **Feature Engineering:** Removing uninformative timestamps (`Time`) and applying $\log(x + 1)$ transformations to heavy-tailed transaction amounts (`Amount`).
3. **Multi-Model Benchmarking:** Evaluating Logistic Regression, Decision Tree, KNN, and Random Forest across Precision, Recall, F1-Score, and ROC-AUC.
4. **Class Imbalance Resolution:** Applying **SMOTE** (Synthetic Minority Over-sampling Technique) strictly to training partitions to prevent data leakage.
5. **Deployment Readiness:** Packaging the top-performing model and standardizer into an end-to-end Scikit-Learn `Pipeline` with a real-time transaction scoring function and serialized model artifact (`fraud_detection_model.pkl`).

---

## Repository Files & Modules

```text
CREDITCARD-FRAUD-DETECTION-PROJECT/
├── creditcard.csv.zip                                    # Full transactions dataset (284,807 records, 31 features)
├── Data_Exploration_and_Visualization.ipynb              # Module 1: Exploratory analysis, summary info, distributions
├── Feature_Engineering.ipynb                             # Module 2: Drop Time, LogAmount transform, feature scaling
├── Credit Card Fraud Detection - Logistic Regression.ipynb # Module 3: Balanced-weight linear probabilistic classification
├── Credit Card Fraud Detection - Decision Tree.ipynb     # Module 4: Non-linear tree classification (depth=6)
├── Credit Card Fraud Detection - K-Nearest Neighbor.ipynb# Module 5: Distance-based neighborhood classification (k=5)
├── credit-card-fraud-prediction-rf-smote.ipynb           # Module 6: Extra Model: Random Forest + SMOTE over-sampling
├── Model_Comparison.ipynb                                # Module 7: Side-by-side benchmark table & ROC curve comparison
├── Model_evaluation.ipynb                                # Module 8: Confusion Matrix heatmap & financial risk analysis
├── Model_deployment.ipynb                                # Module 9: Production Pipeline, serialization & predict_fraud()
├── fraud_detection_model.pkl                             # Serialized Scikit-Learn production Pipeline artifact
├── requirements.txt                                      # Project package dependencies
└── README.md                                             # Comprehensive documentation and project report
```

---

## Dataset Specifications

The project analyzes the benchmark [Kaggle / ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset containing transactions by European cardholders:

- **Total Records:** 284,807 transactions
- **Legitimate Transactions:** 284,315 (99.828%)
- **Fraudulent Transactions:** 492 (0.172%)
- **Features:** 30 numerical predictors:
  - `Time`: Elapsed seconds from the initial transaction in the dataset.
  - `V1` to `V28`: Anonymized principal components extracted through PCA.
  - `Amount`: Transaction expenditure amount (heavily right-skewed).
  - `Class`: Ground-truth binary target (`1` for fraud, `0` for legitimate).

---

## Modular Notebook Workflow

1. **`Data_Exploration_and_Visualization.ipynb`:**
   - Ingests `creditcard.csv` / `creditcard.csv.zip`.
   - Audits missing records, summary types (`.info()`), and head/tail entries.
   - Generates log-scale class imbalance count plots and Amount boxplots.

2. **`Feature_Engineering.ipynb`:**
   - Drops non-generalizable `Time` feature.
   - Applies natural logarithm transformation $\text{LogAmount} = \ln(\text{Amount} + 1)$ to normalize positive skewness.
   - Drops raw `Amount` and plots the normalized `LogAmount` distribution.

3. **`Credit Card Fraud Detection - Logistic Regression.ipynb`:**
   - Fits balanced-weighted linear classification (`class_weight='balanced'`).
   - Evaluates confusion matrix and classification report.

4. **`Credit Card Fraud Detection - Decision Tree.ipynb`:**
   - Fits non-linear tree rules constrained to `max_depth=6` to prevent branch overfitting.

5. **`Credit Card Fraud Detection - K-Nearest Neighbor.ipynb`:**
   - Standardizes features and evaluates distance-based local voting ($k = 5$).

6. **`credit-card-fraud-prediction-rf-smote.ipynb` (Extra Model):**
   - Applies **SMOTE** strictly on the training partition to balance the minority fraud class up to a 10% ratio.
   - Fits an ensemble Random Forest with 100 bagged trees.

7. **`Model_Comparison.ipynb`:**
   - Evaluates models on the same held-out test split.
   - Outputs side-by-side metric tables and combined ROC curves.

8. **`Model_evaluation.ipynb`:**
   - Generates Seaborn confusion matrix heatmaps.
   - Calculates financial impact: balancing verification costs against unrecovered fraud losses.

9. **`Model_deployment.ipynb`:**
   - Builds an end-to-end `Pipeline([('scaler', StandardScaler()), ('log_reg', LogisticRegression(...))])`.
   - Serializes the pipeline to `fraud_detection_model.pkl` via `joblib`.
   - Implements a real-time `predict_fraud(transaction_frame)` inference function.

---

## Model Benchmark & Evaluation Results

Evaluated on the held-out stratified test partition:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Primary Strength |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Logistic Regression (Balanced)** | 97.46% | 5.86% | **89.80%** | 0.1100 | **0.9702** | **Highest Fraud Capture (Recall)** |
| **Random Forest + SMOTE** | **99.94%** | **83.13%** | 70.41% | **0.7624** | 0.9406 | **Best Overall Precision & F1-Score** |
| **Decision Tree (depth=6)** | 98.39% | 8.87% | 85.71% | 0.1608 | 0.9329 | Non-linear Interpretability |
| **K-Nearest Neighbors (k=5)** | 99.88% | 80.00% | 61.54% | 0.6957 | 0.9168 | Instance-based Baseline |

### Metric Trade-Off Discussion:
- **Why Recall is Paramount in Fraud:** In financial risk management, the cost of a **False Negative** (allowing an unauthorized transaction through) is vastly greater than a **False Positive** (prompting an SMS two-factor verification for an approved purchase).
- **Logistic Regression (Balanced)** catches **~90% of all fraudulent attempts** with an exceptional ROC-AUC of **0.9702**.
- **Random Forest + SMOTE** delivers minimal false alarms with an **83.1% Precision** and **99.94% Accuracy**.

---

## Production Deployment & Real-Time Scoring

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load serialized pipeline
model_pipeline = joblib.load('fraud_detection_model.pkl')

# Real-time scoring function
def predict_fraud(transaction_frame, threshold=0.50):
    probs = model_pipeline.predict_proba(transaction_frame)[:, 1]
    predictions = (probs >= threshold).astype(int)
    risk_level = ['Fraud Alert (Block)' if p == 1 else 'Approved' for p in predictions]
    return pd.DataFrame({
        'Fraud_Probability': np.round(probs, 4),
        'Prediction': predictions,
        'Decision': risk_level
    })
```

---

## Financial Cost Impact & Business Insights

1. **Multi-Tiered Decision Policy:**
   - **Auto-Block:** Transactions with `Fraud_Probability > 0.80` are declined automatically.
   - **Step-Up Verification:** Transactions with `0.30 <= Fraud_Probability <= 0.80` trigger immediate 2FA SMS or mobile app confirmation.
   - **Auto-Approve:** Transactions with `Fraud_Probability < 0.30` pass with zero friction.
2. **Asymmetric Risk Management:** Calibrating decision thresholds toward higher Recall eliminates over 90% of total unauthorized chargeback losses.

---

## Setup & Installation

### Prerequisites
- Python 3.8 or higher installed
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
pip install -r requirements.txt
```

---

## Usage Instructions

Launch Jupyter to open and run any of the modular notebooks:
```bash
jupyter notebook
```
All notebooks automatically detect and load `creditcard.csv` or `creditcard.csv.zip` without manual configuration.

---

## Author & Acknowledgments

- **Author:** Jayakumar P
- **GitHub:** [@jayakumarjk2007](https://github.com/jayakumarjk2007)
- **Dataset:** [ULB Machine Learning Group (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
