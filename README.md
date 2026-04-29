# Credit Card Fraud Detection System

A complete machine learning project for detecting fraudulent credit card transactions. It includes exploratory analysis, feature engineering, model training, model comparison, detailed evaluation, and a simple deployment workflow.

## Dataset

Download the public Kaggle dataset from:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place `creditcard.csv` in the project root before running the notebooks or `main.py`.

## Quick Start

```bash
cd CREDITCARD-FRAUD-DETECTION-MAIN
pip install -r requirements.txt
python main.py
```

## Notebook Order

1. `Code/01_Data_Exploration.ipynb`
2. `Code/02_Feature_Engineering.ipynb`
3. `Code/03_Logistic_Regression.ipynb`
4. `Code/04_Decision_Tree.ipynb`
5. `Code/05_KNN_Classifier.ipynb`
6. `Code/06_Model_Comparison.ipynb`
7. `Code/07_Model_Evaluation.ipynb`
8. `Code/08_Model_Deployment.ipynb`

## Project Structure

```text
CREDITCARD-FRAUD-DETECTION-MAIN/
├── Code/
├── data/
├── models/
├── main.py
├── requirements.txt
└── README.md
```

## Models

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors

The workflow uses stratified train-test splitting and applies SMOTE only to the training set to avoid data leakage.
