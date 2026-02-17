#!/usr/bin/env python3
"""dashboard_gradio.py
Dashboard (Gradio) per:
- predizione singola (titolo+testo) con probabilità
- batch: upload CSV (title,body) -> export CSV con timestamp

Avvio:
  pip install -r requirements.txt
  python src/dashboard_gradio.py
"""
import os
from datetime import datetime
import pandas as pd
import joblib
import gradio as gr
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_DEPT = os.path.join(MODELS_DIR, "model_department.joblib")
MODEL_SENT = os.path.join(MODELS_DIR, "model_sentiment.joblib")

def clean_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.lower()
    s = re.sub(r"[^a-zàèéìòù0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_models():
    if not (os.path.exists(MODEL_DEPT) and os.path.exists(MODEL_SENT)):
        raise RuntimeError("Modelli non trovati. Esegui prima il training.")
    return joblib.load(MODEL_DEPT), joblib.load(MODEL_SENT)

pipe_dept, pipe_sent = load_models()

def predict_single(title, body):
    text = clean_text((title or "") + " " + (body or ""))
    dept = pipe_dept.predict([text])[0]
    sent = pipe_sent.predict([text])[0]
    dept_p = pipe_dept.predict_proba([text]).max() if hasattr(pipe_dept.named_steps["clf"], "predict_proba") else 1.0
    sent_p = pipe_sent.predict_proba([text]).max() if hasattr(pipe_sent.named_steps["clf"], "predict_proba") else 1.0
    return f"{dept} ({dept_p:.2%})", f"{sent} ({sent_p:.2%})"

def predict_csv(file):
    df = pd.read_csv(file.name)
    if not {"title","body"}.issubset(df.columns):
        raise ValueError("Il CSV deve avere colonne: title, body")
    X = (df["title"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)).map(clean_text)

    out = df.copy()
    out["pred_department"] = pipe_dept.predict(X)
    out["pred_sentiment"]  = pipe_sent.predict(X)
    out["prob_department"] = pipe_dept.predict_proba(X).max(axis=1)
    out["prob_sentiment"]  = pipe_sent.predict_proba(X).max(axis=1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"batch_predictions_{ts}.csv")
    out.to_csv(out_path, index=False, encoding="utf-8")
    return out_path

with gr.Blocks(title="Hotel Review Analyzer") as demo:
    gr.Markdown("""# 🏨 Hotel Review Analyzer (minimale)
Esegui prima il training (vedi README).""")

    with gr.Tab("Predizione singola"):
        t = gr.Textbox(label="Titolo")
        b = gr.Textbox(label="Testo recensione", lines=4)
        btn = gr.Button("Analizza")
        o1 = gr.Textbox(label="Reparto")
        o2 = gr.Textbox(label="Sentiment")
        btn.click(predict_single, inputs=[t,b], outputs=[o1,o2])

    with gr.Tab("Batch da CSV"):
        f = gr.File(label="CSV con colonne: title, body")
        btn2 = gr.Button("Predici e salva CSV")
        out_file = gr.File(label="Risultato")
        btn2.click(predict_csv, inputs=[f], outputs=[out_file])

if __name__ == "__main__":
    demo.launch(share=True)
