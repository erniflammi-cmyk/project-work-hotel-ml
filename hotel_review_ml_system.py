#!/usr/bin/env python3
"""hotel_review_ml_system.py
Pipeline ML minimale e riproducibile:

1) Training + valutazione:
- Preprocessing: lowercasing, rimozione punteggiatura, tokenizzazione semplice (split implicito in TF-IDF)
- TF-IDF (1-2gram) + Logistic Regression
- Valutazione: accuracy + F1 macro + confusion matrix (salvata in results/)
- Salvataggio modelli in /models

2) Predizione batch:
- Legge un CSV con colonne title, body (può essere anche il dataset completo)
- Esporta un CSV con predizioni + probabilità e timestamp

Uso:
  python src/hotel_review_ml_system.py train --csv dataset/synthetic_reviews.csv
  python src/hotel_review_ml_system.py predict --in dataset/synthetic_reviews.csv --out results/predictions.csv
"""
import argparse
import os
import re
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_DEPT = os.path.join(MODELS_DIR, "model_department.joblib")
MODEL_SENT = os.path.join(MODELS_DIR, "model_sentiment.joblib")

def clean_text(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = s.lower()
    s = re.sub(r"[^a-zàèéìòù0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_text(df: pd.DataFrame) -> pd.Series:
    title = df["title"].fillna("").astype(str) if "title" in df.columns else pd.Series([""]*len(df))
    body  = df["body"].fillna("").astype(str) if "body" in df.columns else pd.Series([""]*len(df))
    return (title + " " + body).map(clean_text)

def make_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=4000)),
        ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", solver="lbfgs")),
    ])

def plot_cm(cm: np.ndarray, labels: list, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predetto")
    ax.set_ylabel("Reale")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def train(csv_path: str, test_size: float = 0.2, seed: int = 42):
    df = pd.read_csv(csv_path)
    if not {"department","sentiment"}.issubset(df.columns):
        raise ValueError("Il CSV deve contenere: department, sentiment (e idealmente title, body).")

    X = build_text(df)
    y_dept = df["department"].astype(str)
    y_sent = df["sentiment"].astype(str)

    Xtr_d, Xts_d, ytr_d, yts_d = train_test_split(X, y_dept, test_size=test_size, random_state=seed, stratify=y_dept)
    Xtr_s, Xts_s, ytr_s, yts_s = train_test_split(X, y_sent, test_size=test_size, random_state=seed, stratify=y_sent)

    pipe_dept = make_pipeline()
    pipe_sent = make_pipeline()

    pipe_dept.fit(Xtr_d, ytr_d)
    pipe_sent.fit(Xtr_s, ytr_s)

    # Eval reparto
    pred_d = pipe_dept.predict(Xts_d)
    acc_d = accuracy_score(yts_d, pred_d)
    f1_d  = f1_score(yts_d, pred_d, average="macro", zero_division=0)
    labels_d = sorted(pd.Series(list(yts_d) + list(pred_d)).unique())
    cm_d = confusion_matrix(yts_d, pred_d, labels=labels_d)
    report_d = classification_report(yts_d, pred_d, zero_division=0)

    # Eval sentiment
    pred_s = pipe_sent.predict(Xts_s)
    acc_s = accuracy_score(yts_s, pred_s)
    f1_s  = f1_score(yts_s, pred_s, average="macro", zero_division=0)
    labels_s = sorted(pd.Series(list(yts_s) + list(pred_s)).unique())
    cm_s = confusion_matrix(yts_s, pred_s, labels=labels_s)
    report_s = classification_report(yts_s, pred_s, zero_division=0)

    joblib.dump(pipe_dept, MODEL_DEPT)
    joblib.dump(pipe_sent, MODEL_SENT)

    plot_cm(cm_d, labels_d, "Confusion Matrix - Department", os.path.join(RESULTS_DIR, "cm_department.png"))
    plot_cm(cm_s, labels_s, "Confusion Matrix - Sentiment", os.path.join(RESULTS_DIR, "cm_sentiment.png"))

    print("✅ Training completato.")
    print(f"Department -> accuracy={acc_d:.3f}  F1_macro={f1_d:.3f}")
    print(f"Sentiment  -> accuracy={acc_s:.3f}  F1_macro={f1_s:.3f}\n")
    print("=== Report Department ===\n" + report_d)
    print("=== Report Sentiment ===\n" + report_s)

def predict_batch(in_csv: str, out_csv: str = None):
    if out_csv is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join(RESULTS_DIR, f"predictions_{ts}.csv")

    pipe_dept = joblib.load(MODEL_DEPT)
    pipe_sent = joblib.load(MODEL_SENT)

    df = pd.read_csv(in_csv)
    X = build_text(df)

    out = df.copy()
    out["pred_department"] = pipe_dept.predict(X)
    out["pred_sentiment"]  = pipe_sent.predict(X)

    if hasattr(pipe_dept.named_steps["clf"], "predict_proba"):
        out["prob_department"] = pipe_dept.predict_proba(X).max(axis=1)
    if hasattr(pipe_sent.named_steps["clf"], "predict_proba"):
        out["prob_sentiment"] = pipe_sent.predict_proba(X).max(axis=1)

    out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✅ Predizioni esportate in: {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_train = sub.add_parser("train")
    ap_train.add_argument("--csv", required=True)
    ap_train.add_argument("--test_size", type=float, default=0.2)
    ap_train.add_argument("--seed", type=int, default=42)

    ap_pred = sub.add_parser("predict")
    ap_pred.add_argument("--in", dest="in_csv", required=True)
    ap_pred.add_argument("--out", dest="out_csv", default=None)

    args = ap.parse_args()
    if args.cmd == "train":
        train(args.csv, test_size=args.test_size, seed=args.seed)
    else:
        predict_batch(args.in_csv, args.out_csv)

if __name__ == "__main__":
    main()
