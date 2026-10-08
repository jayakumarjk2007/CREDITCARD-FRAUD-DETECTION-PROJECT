# Credit Card Fraud Detection System

A machine learning system for real-time financial fraud detection on imbalanced transaction streams.

---

## Executive Summary

Credit card fraud poses a multi-billion dollar threat to financial institutions worldwide. This repository contains a production-grade Credit Card Fraud Detection System developed using real-world transaction data.

The project addresses the fundamental challenge in financial fraud detection: extreme class imbalance (only 492 frauds out of 284,807 transactions, or 0.172%). Through exploratory data analysis, domain-driven feature engineering, cost-sensitive learning, and synthetic minority over-sampling (SMOTE), the system achieves high fraud recall while minimizing false positive alerts.

---

## Project Structure and Modular Notebooks

The pipeline is organized in sequential order (01 through 09):

```text
CREDITCARD-FRAUD-DETECTION-PROJECT/
|-- Code/                                                        # Sequential module implementations
|   |-- 01_Data_Exploration_and_Visualization.ipynb              # Module 1: EDA and Class Imbalance Analysis
|   |-- 02_Feature_Engineering.ipynb                             # Module 2: Skewness Correction and Scaling
|   |-- 03_Credit_Card_Fraud_Detection_Logistic_Regression.ipynb # Module 3: Balanced Logistic Regression
|   |-- 04_Credit_Card_Fraud_Detection_Decision_Tree.ipynb       # Module 4: Constrained Decision Tree
|   |-- 05_Credit_Card_Fraud_Detection_K_Nearest_Neighbor.ipynb  # Module 5: Scaled Distance-Based Classification
|   |-- 06_Credit_Card_Fraud_Prediction_RF_SMOTE.ipynb           # Module 6: Random Forest Ensemble with SMOTE
|   |-- 07_Model_Comparison.ipynb                                # Module 7: Unified Benchmark Evaluation
|   |-- 08_Model_Evaluation.ipynb                                # Module 8: Business Cost Matrix and Calibration
|   |-- 09_Model_Deployment.ipynb                                # Module 9: Pipeline Inference and Latency
|-- creditcard.csv                                               # Complete dataset (284,807 rows, 31 features)
|-- creditcard.csv.zip                                           # Compressed dataset archive
|-- README.md                                                    # Project documentation
`-- requirements.txt                                             # Dependencies
```

---

## Dataset Overview

- Source: European cardholder credit card transactions
- Total Transactions: 284,807
- Legitimate Transactions: 284,315 (99.828%)
- Fraudulent Transactions: 492 (0.172%)
- Total Features: 31
  - Time: Elapsed seconds since the first transaction
  - V1 through V28: Principal components obtained via PCA
  - Amount: Transaction monetary amount
  - Class: Target variable (0 = Legitimate, 1 = Fraudulent)

---

## Methodology and Workflow

### 1. Exploratory Data Analysis (01_Data_Exploration_and_Visualization.ipynb)
- Quantified the 578:1 class imbalance ratio.
- Analyzed transaction Amount distributions showing heavy right-skewness.
- Analyzed Time feature showing cyclical patterns over 48 hours.
- Evaluated correlation matrix showing strong negative signals in V14, V12, V10, V17 and positive signals in V4, V11.

### 2. Feature Engineering (02_Feature_Engineering.ipynb)
- Dropped Time column to prevent sequence memorization.
- Applied log transformation (np.log1p) on transaction Amount.
- Standardized features with StandardScaler.
- Examined outlier boundaries using the Interquartile Range (IQR) method.
- Established an 80/20 stratified split preserving class ratios.

### 3. Model Benchmark and Comparison (07_Model_Comparison.ipynb)

All models were evaluated on the same stratified test set:

| Architecture | Strategy / Hyperparameters | Fraud Recall | Precision | F1-Score | ROC-AUC | PR-AUC |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Random Forest + SMOTE | 100 Trees, Depth 10, SMOTE 50/50 | 85.7% | 86.6% | 0.861 | 0.978 | 0.865 |
| Logistic Regression (Balanced) | class_weight='balanced', lbfgs | 91.8% | 6.8% | 0.127 | 0.974 | 0.748 |
| Decision Tree (Pruned) | Depth 5, min_samples_split=50 | 87.8% | 34.5% | 0.496 | 0.941 | 0.612 |
| K-Nearest Neighbors | k=5, Minkowski distance | 84.7% | 46.2% | 0.597 | 0.943 | 0.639 |

### 4. Key Performance Insights
1. Due to extreme imbalance, Precision-Recall AUC (PR-AUC) and F1-Score are the primary evaluation metrics.
2. Random Forest combined with SMOTE achieved the highest balanced performance (F1 = 0.861).
3. Balanced Logistic Regression achieved the highest fraud recall (91.8%), making it effective for low-latency screening.

---

## Business Cost Optimization (08_Model_Evaluation.ipynb)

Financial institutions incur asymmetric costs on classification errors:
- False Negative (Missed Fraud): Direct financial loss (~$122.00 average fraud amount).
- False Positive (False Alarm): Operational and customer friction cost (~$5.00).

Threshold optimization reveals an optimal operating threshold between 0.28 and 0.35, minimizing total financial loss while detecting over 94% of fraud attempts.

---

## Deployment and Inference (09_Model_Deployment.ipynb)

A Scikit-learn Pipeline processes raw transaction inputs and outputs fraud probabilities in real time:

- Inference Latency: Under 0.35 ms per transaction
- Throughput: Exceeds 3,000 transactions per second on CPU
- Decision Categories:
  - Probability < 0.20: Low Risk (Approve)
  - Probability 0.20 to 0.60: Medium Risk (2FA Challenge)
  - Probability >= 0.60: Critical Risk (Block Transaction)

---

## Installation and Setup

```bash
# 1. Clone repository
git clone https://github.com/jayakumarjk2007/CREDITCARD-FRAUD-DETECTION-PROJECT.git
cd CREDITCARD-FRAUD-DETECTION-PROJECT

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook
```

---

## Dependencies (requirements.txt)

- pandas>=2.0.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- scikit-learn>=1.3.0
- imbalanced-learn>=0.11.0
- jupyter>=1.0.0

---

## Author
- Jayakumar P
- Credit Card Fraud Detection Project
