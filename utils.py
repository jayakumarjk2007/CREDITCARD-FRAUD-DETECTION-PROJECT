"""Utility helpers for credit card fraud detection notebooks."""

import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve


def load_data(filepath="../creditcard.csv"):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError as exc:
        raise FileNotFoundError("Dataset not found. Download creditcard.csv from Kaggle and place it in the project root.") from exc
    print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def check_data_quality(df):
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        scores = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_precision_recall(y_true, scores):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_model(model, filepath):
    with open(filepath, "wb") as fh:
        pickle.dump(model, fh)
    print(f"Saved model to {filepath}")


def load_model(filepath):
    with open(filepath, "rb") as fh:
        model = pickle.load(fh)
    print(f"Loaded model from {filepath}")
    return model


def calculate_financial_impact(y_true, y_pred, avg_transaction=100, fraud_cost_multiplier=10):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    false_positive_cost = avg_transaction * 0.10
    missed_fraud_cost = avg_transaction * fraud_cost_multiplier
    review_cost = avg_transaction * 0.10
    total_cost = fp * false_positive_cost + fn * missed_fraud_cost + tp * review_cost
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_cost": float(total_cost),
        "avg_cost_per_transaction": float(total_cost / len(y_true)),
    }
