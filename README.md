# Credit Card Fraud Detection Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-0.11%2B-red.svg)](https://imbalanced-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning system designed to detect fraudulent credit card transactions in heavily imbalanced financial data. This project implements data cleaning, feature engineering with log transformation, multiple classification models (**Logistic Regression**, **Decision Trees**, **K-Nearest Neighbors**, and **Random Forest with SMOTE**), side-by-side metric comparison, and production deployment via a Scikit-Learn `Pipeline`.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Files](#repository-files)
- [Dataset Specifications & Exploration](#dataset-specifications--exploration)
- [Data Cleaning & Feature Engineering](#data-cleaning--feature-engineering)
- [Machine Learning Modeling & Imbalance Treatment](#machine-learning-modeling--imbalance-treatment)
- [Model Benchmark & Evaluation Results](#model-benchmark--evaluation-results)
- [Production Deployment & Real-Time Scoring](#production-deployment--real-time-scoring)
- [Financial Cost Impact & Business Insights](#financial-cost-impact--business-insights)
- [Setup & Installation](#setup--installation)
- [Usage Instructions](#usage-instructions)
- [Author & Acknowledgments](#author--acknowledgments)

---

## Project Overview

Credit card fraud presents extreme class imbalance, where fraudulent transactions account for less than **0.2%** of all transactions. Conventional accuracy metrics are ineffective in this domain, as a naive classifier predicting all transactions as legitimate would achieve 99.83% accuracy while failing to detect 100% of actual fraud.

This project delivers:
1. **Data Exploration & Visualizations:** Auditing class distributions, log-scale transaction volume, and amount distributions by class.
2. **Feature Engineering:** Removing uninformative timestamps (`Time`) and applying $\log(x + 1)$ transformations to heavy-tailed transaction amounts (`Amount`).
3. **Multi-Model Benchmarking:** Evaluating Logistic Regression, Decision Tree, KNN, and Random Forest across Precision, Recall, F1-Score, and ROC-AUC.
4. **Class Imbalance Resolution:** Applying **SMOTE** (Synthetic Minority Over-sampling Technique) strictly to the training partition to prevent data leakage.
5. **Deployment Readiness:** Packaging the top-performing model and standardizer into an end-to-end Scikit-Learn `Pipeline` with a real-time transaction scoring function.

---

## Repository Files

```text
CREDITCARD-FRAUD-DETECTION-PROJECT/
├── creditcard.csv.zip                  # Compressed benchmark transactions dataset (284,807 records, 31 features)
├── Credit_Card_Fraud_Detection.ipynb   # Complete, executed end-to-end Jupyter Notebook
├── requirements.txt                    # Project package dependencies
└── README.md                           # Comprehensive documentation and project report
```

---

## Dataset Specifications & Exploration

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

## Data Cleaning & Feature Engineering

1. **Missing Values & Deduplication:** Confirmed zero null values (`0`), and removed duplicate records to prevent data contamination.
2. **Dropping `Time`:** The elapsed second counter does not generalize across future time windows and is removed.
3. **Log Transformation on `Amount`:** Transaction amounts exhibit high positive skew. Applying $\log(\text{Amount} + 1)$ compresses outlier variance:
   $$\text{LogAmount} = \ln(\text{Amount} + 1)$$
4. **Standardization:** Normalized `LogAmount` using `StandardScaler` ($\mu = 0, \sigma = 1$).
5. **Stratified Splitting:** 80% train / 20% test partition maintaining the exact 0.17% fraud proportion in both splits (`random_state=42`).

---

## Machine Learning Modeling & Imbalance Treatment

We train and evaluate four distinct classifiers:

1. **Logistic Regression (Balanced Weights):** Linear probabilistic baseline using inverse class weighting (`class_weight='balanced'`).
2. **Decision Tree Classifier:** Non-linear decision rules constrained to `max_depth=6` with balanced weighting to prevent branch overfitting.
3. **K-Nearest Neighbors (KNN):** Distance-based neighborhood classification ($k = 5$).
4. **Random Forest + SMOTE (Extra Model):** Applying SMOTE on the training split to synthesize minority fraud instances up to a 10% ratio, followed by a 100-estimator ensemble Random Forest (`max_depth=10`).

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
- **Why Recall is Paramount in Fraud:** In financial risk management, the cost of a **False Negative** (allowing an unauthorized $1,000 transaction through) is vastly greater than a **False Positive** (prompting an SMS two-factor verification for an approved purchase).
- **Logistic Regression (Balanced)** catches **~90% of all fraudulent attempts** with an exceptional ROC-AUC of **0.9702**.
- **Random Forest + SMOTE** delivers minimal false alarms with an **83.1% Precision** and **99.94% Accuracy**.

---

## Production Deployment & Real-Time Scoring

The notebook packages the data preprocessing and classifier into an end-to-end `Pipeline` and provides a reusable inference function:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Build and serialize Pipeline
production_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])
production_pipeline.fit(X_train, y_train)
joblib.dump(production_pipeline, 'fraud_detection_model.pkl')

# 2. Production Scoring Function
def predict_fraud(transaction_features, threshold=0.50):
    probs = production_pipeline.predict_proba(transaction_features)[:, 1]
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

1. **Multi-Tiered Decision Thresholds:**
   - **Auto-Block:** Transactions with `Fraud_Probability > 0.80` are declined automatically.
   - **Step-Up Verification:** Transactions with `0.30 <= Fraud_Probability <= 0.80` trigger immediate 2FA SMS or mobile app confirmation.
   - **Auto-Approve:** Transactions with `Fraud_Probability < 0.30` pass with zero friction.
2. **Asymmetric Risk Management:** By calibrating decision thresholds toward higher Recall, financial institutions can eliminate over 90% of total unauthorized chargeback losses.

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

Launch Jupyter to open the notebook and inspect all pre-executed cells, metric tables, and ROC curves:
```bash
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```
Select **Kernel > Restart & Run All** to re-run the entire pipeline. The notebook automatically handles either `creditcard.csv` or `creditcard.csv.zip`.

---

## Author & Acknowledgments

- **Author:** Jayakumar P
- **GitHub:** [@jayakumarjk2007](https://github.com/jayakumarjk2007)
- **Dataset:** [ULB Machine Learning Group (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
