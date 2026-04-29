"""Credit Card Fraud Detection - Main Training Pipeline."""

from pathlib import Path
import pickle
import time
import warnings

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


def load_and_preprocess_data(dataset_path="creditcard.csv"):
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError("creditcard.csv not found. Download it from Kaggle and place it in the project root.")

    print("=" * 72)
    print("CREDIT CARD FRAUD DETECTION - MODEL TRAINING PIPELINE")
    print("=" * 72)
    print("\n[1/6] Loading dataset...")
    df = pd.read_csv(path)
    fraud = int((df["Class"] == 1).sum())
    legit = int((df["Class"] == 0).sum())
    print(f"Loaded {len(df):,} transactions")
    print(f"Legitimate: {legit:,} | Fraud: {fraud:,} | Fraud rate: {fraud / len(df) * 100:.4f}%")

    print("\n[2/6] Scaling Amount and Time...")
    processed = df.copy()
    amount_scaler = RobustScaler()
    time_scaler = StandardScaler()
    processed["Amount_Scaled"] = amount_scaler.fit_transform(processed[["Amount"]])
    processed["Time_Scaled"] = time_scaler.fit_transform(processed[["Time"]])

    X = processed.drop(columns=["Time", "Amount", "Class"])
    y = processed["Class"]

    print("\n[3/6] Creating stratified train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    print("\n[4/6] Applying SMOTE to training data...")
    smote = SMOTE(sampling_strategy=0.30, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"Resampled train rows: {len(X_train_smote):,}")
    return X_train_smote, X_test, y_train_smote, y_test, amount_scaler, time_scaler


def train_models(X_train, X_test, y_train, y_test):
    print("\n[5/6] Training models...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    }
    results = {}
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "model": model,
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_score),
            "seconds": time.time() - start,
            "report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }
        print(f"{name}: precision={results[name]['precision']:.4f}, recall={results[name]['recall']:.4f}, f1={results[name]['f1']:.4f}, auc={results[name]['roc_auc']:.4f}")
    return results


def save_artifacts(results, amount_scaler, time_scaler):
    print("\n[6/6] Saving artifacts...")
    Path("models").mkdir(exist_ok=True)
    best_name = max(results, key=lambda key: results[key]["f1"])
    with open("models/fraud_detection_model.pkl", "wb") as fh:
        pickle.dump(results[best_name]["model"], fh)
    with open("models/scalers.pkl", "wb") as fh:
        pickle.dump({"amount_scaler": amount_scaler, "time_scaler": time_scaler}, fh)
    print(f"Best model by F1-score: {best_name}")
    return best_name


def print_summary(results, best_name):
    print("\n" + "=" * 72)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 72)
    print(f"{'Model':<24} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 72)
    for name, metrics in results.items():
        marker = " <- BEST" if name == best_name else ""
        print(f"{name:<24} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['roc_auc']:>10.4f}{marker}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, amount_scaler, time_scaler = load_and_preprocess_data()
    results = train_models(X_train, X_test, y_train, y_test)
    best = save_artifacts(results, amount_scaler, time_scaler)
    print_summary(results, best)
