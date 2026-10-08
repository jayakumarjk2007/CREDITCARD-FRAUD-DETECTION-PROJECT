# Credit Card Fraud Detection System

> **High-Performance Machine Learning System for Real-Time Financial Fraud Detection on Imbalanced Transaction Streams.**

---

## 📌 Executive Summary

Credit card fraud poses a multi-billion dollar threat to financial institutions worldwide. This repository contains an end-to-end, production-grade **Credit Card Fraud Detection System** developed using real-world transaction data.

The project addresses the fundamental challenge in financial fraud detection: **extreme class imbalance** (only **492 frauds** out of **284,807 transactions**, or **0.172%**). Through rigorous exploratory data analysis, domain-driven feature engineering, cost-sensitive learning, and synthetic minority over-sampling (SMOTE), the system achieves exceptional fraud recall while minimizing customer friction from false positives.

---

## 📂 Project Structure & Modular Notebooks

The project is structured into 9 modular, self-contained notebooks mirroring the production machine learning lifecycle:

```text
CREDITCARD-FRAUD-DETECTION-PROJECT/
├── Code/                                                   # Core module implementations
│   ├── Data_Exploration_and_Visualization.ipynb            # Module 1: Comprehensive EDA & Class Imbalance Analysis
│   ├── Feature_Engineering.ipynb                           # Module 2: Skewness Correction, Log Transform & Outlier Treatment
│   ├── Credit Card Fraud Detection - Logistic Regression.ipynb # Module 3: Cost-Sensitive Logistic Regression & Threshold Tuning
│   ├── Credit Card Fraud Detection - Decision Tree.ipynb   # Module 4: Constrained Decision Tree & Rule Extraction
│   ├── Credit Card Fraud Detection - K-Nearest Neighbor.ipynb # Module 5: Scaled Distance-Based Instance Classification
│   ├── credit-card-fraud-prediction-rf-smote.ipynb        # Module 6: Random Forest Ensemble with SMOTE Oversampling
│   ├── Model_Comparison.ipynb                              # Module 7: Unified Benchmark (PR-AUC, ROC-AUC, F1, Latency)
│   ├── Model_evaluation.ipynb                              # Module 8: Financial Cost Matrix, Calibration & K-Fold CV
│   ├── Model_deployment.ipynb                              # Module 9: Scikit-learn Pipeline Serialization & FastAPI Spec
│   ├── fraud_detection_model.pkl                          # Trained & serialized production pipeline
│   └── scaler.pkl                                          # Serialized feature standardizer
├── creditcard.csv                                          # Complete transaction dataset (284,807 rows, 31 features)
├── creditcard.csv.zip                                      # Compressed dataset for portable distribution
├── README.md                                               # Technical documentation & project report
└── requirements.txt                                        # Python dependencies
```

---

## 📊 Dataset Overview

- **Source:** European cardholder credit card transactions (September 2013).
- **Total Transactions:** 284,807
- **Total Legitimate:** 284,315 (99.828%)
- **Total Fraudulent:** 492 (0.172%)
- **Features:** 31 total
  - `Time`: Elapsed seconds since the first transaction in the dataset.
  - `V1` – `V28`: Principal components obtained via PCA (anonymized for user confidentiality).
  - `Amount`: Transaction amount in Euros/Dollars.
  - `Class`: Target variable (`0 = Legitimate`, `1 = Fraudulent`).

---

## 🔬 Methodology & Workflow

### 1. Exploratory Data Analysis & Visualization (`Data_Exploration_and_Visualization.ipynb`)
- Quantified the 578:1 class imbalance ratio.
- Identified that transaction `Amount` exhibits extreme positive skewness, with fraud transactions having higher variance and distinct amount clusterings.
- Analyzed transaction timing across the 48-hour recording window, highlighting time-of-day fraud activity spikes.
- Correlation analysis identified key fraud discriminators: `V14`, `V12`, `V10`, and `V17` exhibit significant negative shifts during fraud, whereas `V4` and `V11` show strong positive shifts.

### 2. Feature Engineering & Preprocessing (`Feature_Engineering.ipynb`)
- **Time Feature Removal:** Dropped `Time` to prevent temporal overfitting and spurious sequence memorization.
- **Log Transformation:** Applied y = log(1 + x) (`np.log1p`) to compress the extreme right skew of transaction `Amount`.
- **Feature Scaling:** Applied `StandardScaler` to ensure zero-mean, unit-variance inputs across all features.
- **Outlier Analysis:** Used the Interquartile Range (IQR) method to examine extreme feature anomalies in the training split.
- **Stratified Partitioning:** Implemented 80/20 stratified splitting preserving the exact 0.172% fraud prevalence across both sets.

### 3. Model Training & Comparison (`Model_Comparison.ipynb`)

Four diverse classification architectures were implemented and systematically evaluated on the identical stratified test set:

| Architecture | Strategy / Hyperparameters | Fraud Recall | Precision | F1-Score | ROC-AUC | PR-AUC |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest + SMOTE** | 100 Trees, Depth 10, SMOTE 50/50 | **85.7%** | **86.6%** | **0.861** | **0.978** | **0.865** |
| **Logistic Regression (Balanced)** | `class_weight='balanced'`, lbfgs | **91.8%** | 6.8% | 0.127 | 0.974 | 0.748 |
| **Decision Tree (Pruned)** | Depth 5, `min_samples_split=50` | 87.8% | 34.5% | 0.496 | 0.941 | 0.612 |
| **K-Nearest Neighbors** | k=5, Minkowski metric | 84.7% | 46.2% | 0.597 | 0.943 | 0.639 |

### 4. Key Performance Insights
1. **Precision-Recall AUC is Paramount:** Due to severe class imbalance, ROC-AUC is overly optimistic (all models achieve > 0.94). PR-AUC and F1-score represent the true operational capability.
2. **Random Forest + SMOTE** delivers the highest balanced performance (F1 = 0.861) with high precision (minimal false alarms) and strong recall.
3. **Balanced Logistic Regression** achieves the highest raw recall (**91.8%**), making it the optimal first-line screening filter in high-throughput transaction gateways.

---

## 💼 Business Impact & Cost Optimization

In production banking systems, prediction errors carry asymmetric financial costs:
- **False Negative (Missed Fraud):** Direct loss of stolen funds (Average fraud amount: **~$122.00**).
- **False Positive (False Alarm):** Customer friction, automated SMS/2FA, or support verification call cost (**~$5.00**).

Through **threshold optimization** in `Model_evaluation.ipynb`, shifting the decision boundary from default 0.50 down to optimal **0.28 - 0.35**:
- Reduces total financial loss by over **42%** compared to uncalibrated baselines.
- Catches >94% of fraudulent transactions while maintaining a false alarm rate under 0.5%.

---

## 🚀 Deployment & Real-Time Inference (`Model_deployment.ipynb`)

The model is serialized into standard production artifacts:
- `fraud_detection_model.pkl`: Complete end-to-end Scikit-learn Pipeline (StandardScaler + Logistic Regression / Classifier).
- `scaler.pkl`: Standalone pre-fitted standardizer.

### Python Quickstart: Real-Time Transaction Scoring

```python
import joblib
import pandas as pd

# Load serialized pipeline
pipeline = joblib.load("fraud_detection_model.pkl")

# Incoming transaction dictionary
transaction = {
    "Time": 406.0,
    "V1": -2.31, "V2": 1.95, "V3": -1.61, "V4": 3.99,
    "V5": -0.52, "V6": -1.42, "V7": -2.53, "V8": 1.39,
    # ... V9 to V28
    "Amount": 149.62
}

df_tx = pd.DataFrame([transaction])
fraud_probability = pipeline.predict_proba(df_tx)[0, 1]

if fraud_probability >= 0.5:
    print(f"🚨 FRAUD ALERT! Probability: {fraud_probability:.2%}")
else:
    print(f"✅ Approved. Probability: {fraud_probability:.2%}")
```

### SLA Latency Benchmark:
- **Inference Time:** `< 0.35 ms` per transaction.
- **Throughput:** `> 3,000` transactions/second on standard CPU.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- Jupyter Notebook or VS Code Jupyter Extension

### Setup Instructions
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

## 📜 Requirements (`requirements.txt`)

```text
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
joblib>=1.3.0
jupyter>=1.0.0
```

---

## 👨‍💻 Author & Acknowledgments
- **Author:** Jayakumar P
- **Course:** GUVI Data Science & Machine Learning Program
- **Dataset:** European Cardholder Fraud Dataset (Credit Card Fraud Detection)
