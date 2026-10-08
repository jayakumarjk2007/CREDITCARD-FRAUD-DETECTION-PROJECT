# Credit Card Fraud Detection System

A machine learning system for real-time financial fraud detection on imbalanced transaction streams.

---

## Executive Summary

Credit card fraud poses a multi-billion dollar threat to financial institutions worldwide. This repository contains a production-grade Credit Card Fraud Detection System developed using real-world transaction data.

The project addresses the fundamental challenge in financial fraud detection: extreme class imbalance (only 492 frauds out of 284,807 transactions, or 0.172%). Through exploratory data analysis, domain-driven feature engineering, cost-sensitive learning, and synthetic minority over-sampling (SMOTE), the system achieves high fraud recall while minimizing false positive alerts.

---

## Project Structure and Notebooks

The repository provides both a sequential modular breakdown (Modules 01 through 09) and a complete all-in-one master pipeline notebook covering the entire process from start to finish.

```text
CREDITCARD-FRAUD-DETECTION-PROJECT/
|-- Credit_Card_Fraud_Detection_Complete_Pipeline.ipynb      # Complete Master Notebook (Covers All 9 Modules End-to-End)
|-- Credit_Card_Fraud_Detection.ipynb                       # Unified Pipeline Reference
|-- Code/                                                   # Modular sequential notebooks
|   |-- 01_Data_Exploration_and_Visualization.ipynb         # Module 1: EDA and Class Imbalance Analysis
|   |-- 02_Feature_Engineering.ipynb                        # Module 2: Skewness Correction and Scaling
|   |-- 03_Credit_Card_Fraud_Detection_Logistic_Regression.ipynb # Module 3: Balanced Logistic Regression
|   |-- 04_Credit_Card_Fraud_Detection_Decision_Tree.ipynb  # Module 4: Constrained Decision Tree
|   |-- 05_Credit_Card_Fraud_Detection_K_Nearest_Neighbor.ipynb # Module 5: Scaled Distance-Based Classification
|   |-- 06_Credit_Card_Fraud_Prediction_RF_SMOTE.ipynb      # Module 6: Random Forest Ensemble with SMOTE
|   |-- 07_Model_Comparison.ipynb                           # Module 7: Unified Benchmark Evaluation
|   |-- 08_Model_Evaluation.ipynb                           # Module 8: Business Cost Matrix and Calibration
|   `-- 09_Model_Deployment.ipynb                           # Module 9: Pipeline Inference and Latency
|-- 01_Data_Exploration_and_Visualization.ipynb
|-- 02_Feature_Engineering.ipynb
|-- 03_Credit_Card_Fraud_Detection_Logistic_Regression.ipynb
|-- 04_Credit_Card_Fraud_Detection_Decision_Tree.ipynb
|-- 05_Credit_Card_Fraud_Detection_K_Nearest_Neighbor.ipynb
|-- 06_Credit_Card_Fraud_Prediction_RF_SMOTE.ipynb
|-- 07_Model_Comparison.ipynb
|-- 08_Model_Evaluation.ipynb
|-- 09_Model_Deployment.ipynb
|-- creditcard.csv                                          # Complete dataset (284,807 rows, 31 features)
|-- creditcard.csv.zip                                      # Compressed dataset archive
|-- README.md                                               # Technical documentation
`-- requirements.txt                                        # Dependencies
```

---

## Master Pipeline Notebook (Credit_Card_Fraud_Detection_Complete_Pipeline.ipynb)

For users who want to run the entire pipeline in a single session without switching files, `Credit_Card_Fraud_Detection_Complete_Pipeline.ipynb` includes:

1. Data loading and validation
2. Class distribution and transaction amount analysis
3. Correlation heatmaps and top anomaly indicators
4. Skewness correction and standardized preprocessing
5. Logistic Regression with class weighting
6. Pruned Decision Tree with feature importances
7. Scaled K-Nearest Neighbor classification
8. Random Forest with SMOTE oversampling
9. Unified side-by-side benchmark table, ROC curves, and Precision-Recall curves
10. Banking cost-utility curve and threshold optimization
11. In-memory production inference pipeline and real-time transaction scoring

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

## Model Benchmark and Comparison

All models were evaluated on the same stratified test set:

| Architecture | Strategy / Hyperparameters | Fraud Recall | Precision | F1-Score | ROC-AUC | PR-AUC |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Random Forest + SMOTE | 100 Trees, Depth 10, SMOTE 50/50 | 85.7% | 86.6% | 0.861 | 0.978 | 0.865 |
| Logistic Regression (Balanced) | class_weight='balanced', lbfgs | 91.8% | 6.8% | 0.127 | 0.974 | 0.748 |
| Decision Tree (Pruned) | Depth 5, min_samples_split=50 | 87.8% | 34.5% | 0.496 | 0.941 | 0.612 |
| K-Nearest Neighbors | k=5, Minkowski distance | 84.7% | 46.2% | 0.597 | 0.943 | 0.639 |

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
