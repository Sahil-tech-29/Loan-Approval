import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score, train_test_split

from preprocess import clean_and_encode

# ── Config 
DATA_PATH  = "data/loan_data.csv"
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "loan_model.pkl")
META_PATH  = os.path.join(MODEL_DIR, "model_meta.json")
PLOT_PATH  = os.path.join(MODEL_DIR, "roc_curve.png")
os.makedirs(MODEL_DIR, exist_ok=True)


def train():
    print("=" * 55)
    print("   LOAN APPROVAL — MODEL TRAINING")
    print("=" * 55)

    # ── Load & preprocess 
    df = pd.read_csv(DATA_PATH)
    print(f"\n  Loaded {len(df)} rows from {DATA_PATH}")

    X, y, medians = clean_and_encode(df, is_train=True)
    print(f"  Features: {list(X.columns)}")
    print(f"  Class balance — Approved: {y.mean():.1%}  Rejected: {1-y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── Train 
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("\n  Random Forest trained.")

    # ── Evaluate 
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)
    cv_acc  = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()

    print(f"\n📊  Test Accuracy   : {acc:.4f}")
    print(f"📊  ROC-AUC Score   : {auc:.4f}")
    print(f"📊  5-Fold CV Acc   : {cv_acc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Rejected','Approved'])}")

    # ── Confusion matrix (text)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    # ── ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color="#4361ee", lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Loan Approval Model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    plt.close()
    print(f"\n📈  ROC curve saved → {PLOT_PATH}")

    # ── Feature importance 
    feat_imp = dict(zip(X.columns, model.feature_importances_.tolist()))
    feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
    print("\n🔑  Feature Importances:")
    for f, v in feat_imp.items():
        bar = "█" * int(v * 60)
        print(f"   {f:<22} {v:.4f}  {bar}")

    # ── Save model + metadata 
    artifact = {
        "model":   model,
        "medians": medians,
        "features": list(X.columns),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\n Model saved → {MODEL_PATH}")

    meta = {
        "accuracy":  round(acc,  4),
        "roc_auc":   round(auc,  4),
        "cv_accuracy": round(cv_acc, 4),
        "n_estimators": model.n_estimators,
        "n_train":   len(X_train),
        "n_test":    len(X_test),
        "feature_importances": feat_imp,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved → {META_PATH}")
    print("\n  Training complete!\n")


if __name__ == "__main__":
    train()