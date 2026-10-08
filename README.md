# Customer Churn Forecasting System

> **Predictive analytics system to identify at-risk customers and drive proactive retention strategies using machine learning.**

---

## Project Overview

This project develops a comprehensive **Customer Churn Forecasting** system that predicts which customers are likely to stop using a telecommunications service. The system employs multiple machine learning models, performs thorough exploratory data analysis, and provides actionable business insights for customer retention.

### Business Problem
Customer churn is one of the most critical challenges for subscription-based businesses. Acquiring new customers costs **5-7x more** than retaining existing ones. This project builds a predictive model that enables businesses to:
- Identify customers at high risk of churning **before they leave**
- Understand the key factors driving customer attrition
- Deploy targeted retention campaigns with measurable ROI
- Optimize resource allocation for customer success teams

---

## Repository Files & Modules

```text
CREDITCARD-FRAUD-DETECTION-PROJECT/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv              # Telco Customer Churn dataset (7,043 records, 21 features)
├── Data_Exploration_and_Visualization.ipynb            # Module 1: EDA, distributions, churn analysis
├── Feature_Engineering.ipynb                           # Module 2: Data cleaning, encoding, feature creation
├── Customer Churn Forecasting - Logistic Regression.ipynb  # Module 3: Baseline linear classification
├── Customer Churn Forecasting - Decision Tree.ipynb    # Module 4: Non-linear tree classification (depth=6)
├── Customer Churn Forecasting - K-Nearest Neighbor.ipynb   # Module 5: Distance-based KNN classification
├── customer-churn-prediction-rf-smote.ipynb            # Module 6: Random Forest + SMOTE oversampling
├── Model_Comparison.ipynb                              # Module 7: Side-by-side benchmark & ROC comparison
├── Model_evaluation.ipynb                              # Module 8: Confusion matrix, PR curves & financial analysis
├── Model_deployment.ipynb                              # Module 9: Production Pipeline & predict_churn()
├── churn_prediction_model.pkl                          # Serialized production model pipeline
├── requirements.txt                                    # Project dependencies
└── README.md                                           # Project documentation and report
```

---

## Dataset Specifications

The project uses the **Telco Customer Churn** dataset from Kaggle/IBM:

- **Total Records:** 7,043 customers
- **Retained Customers:** 5,174 (73.5%)
- **Churned Customers:** 1,869 (26.5%)
- **Features:** 21 attributes across 4 categories:

| Category | Features |
|---|---|
| **Demographics** | gender, SeniorCitizen, Partner, Dependents |
| **Account Info** | tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges |
| **Services** | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| **Target** | Churn (Yes/No) |

---

## Methodology

### Data Processing Pipeline
1. **Data Cleaning:** Handle missing `TotalCharges` values, convert data types
2. **Feature Engineering:** Create 6 new features:
   - `AvgMonthlySpend` - Customer lifetime value indicator
   - `TotalServices` - Count of active service subscriptions
   - `ChargesPerService` - Normalized monthly cost per service
   - `ContractRisk` - Ordinal risk score based on contract type
   - `HasInternet` - Binary internet service flag
   - `TenureGroup` - Categorical tenure buckets
3. **Encoding:** Binary + One-Hot encoding for categorical features
4. **Scaling:** StandardScaler normalization for distance-based algorithms

### Models Evaluated

| Model | Description |
|---|---|
| **Logistic Regression** | Balanced-weight linear classifier (baseline) |
| **Decision Tree** | Non-linear tree rules (max_depth=6) |
| **K-Nearest Neighbors** | Distance-based neighborhood voting |
| **Random Forest + SMOTE** | Ensemble classifier with synthetic oversampling |

### Evaluation Metrics
- **Accuracy:** Overall prediction correctness
- **Precision:** Proportion of predicted churners who actually churned
- **Recall:** Proportion of actual churners correctly identified
- **F1-Score:** Harmonic mean of Precision and Recall
- **ROC-AUC:** Area Under the Receiver Operating Characteristic curve

---

## Model Performance Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (Balanced)** | ~77% | ~52% | **~80%** | ~0.63 | **~0.84** |
| **Decision Tree (depth=6)** | ~76% | ~50% | ~72% | ~0.59 | ~0.80 |
| **KNN (K=7)** | ~78% | ~55% | ~60% | ~0.57 | ~0.79 |
| **Random Forest + SMOTE** | **~79%** | **~58%** | ~75% | **~0.65** | ~0.83 |

### Key Findings:
- **Logistic Regression** achieves the highest Recall (~80%), catching the most churners
- **Random Forest + SMOTE** delivers the best F1-Score and overall balance
- **Decision Tree** provides the most interpretable rules for business stakeholders

---

## Business Insights & Retention Strategies

### Top Churn Predictors:
1. **Contract Type:** Month-to-month customers churn at ~42% vs. ~3% for 2-year contracts
2. **Tenure:** Customers with < 6 months tenure are most at risk
3. **Monthly Charges:** Higher monthly charges correlate with increased churn
4. **Internet Service:** Fiber optic users show elevated churn rates
5. **Payment Method:** Electronic check users churn significantly more
6. **Support Services:** Lack of online security/tech support increases churn risk

### Recommended Retention Strategies:
1. **Early Intervention Program:** Target customers in their first 6 months with onboarding support and engagement campaigns
2. **Contract Incentives:** Offer discounts for upgrading from month-to-month to annual contracts
3. **Bundle Optimization:** Encourage adoption of online security, backup, and tech support services
4. **Payment Method Migration:** Incentivize electronic check users to switch to automatic payment methods
5. **Price Sensitivity Analysis:** Review pricing for fiber optic plans and offer loyalty discounts for long-tenure customers

### Risk-Based Action Framework:
| Churn Probability | Risk Level | Action |
|---|---|---|
| >= 80% | CRITICAL | Immediate personal outreach, special retention offers |
| 60-79% | HIGH | Targeted retention campaign, loyalty rewards |
| 40-59% | MODERATE | Proactive engagement, service upgrade offers |
| 20-39% | LOW | Regular engagement programs |
| < 20% | MINIMAL | Standard customer experience |

---

## Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine learning models and evaluation
- **Matplotlib** - Static visualizations
- **Seaborn** - Statistical data visualization
- **imbalanced-learn** - SMOTE oversampling
- **Joblib** - Model serialization

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

### Running the Notebooks
Launch Jupyter to explore the modular analysis:
```bash
jupyter notebook
```
Execute notebooks in order (Module 1 through Module 9) for the complete workflow.

### Quick Prediction
```python
import joblib
import pandas as pd

# Load the production model
model = joblib.load('churn_prediction_model.pkl')

# Predict churn for new customer data
probability = model.predict_proba(customer_data)[:, 1]
prediction = model.predict(customer_data)
```

---

## Author & Acknowledgments

- **Author:** Jayakumar P
- **GitHub:** [@jayakumarjk2007](https://github.com/jayakumarjk2007)
- **Dataset:** [IBM Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
