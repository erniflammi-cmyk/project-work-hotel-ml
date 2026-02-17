# Project Work – Smistamento recensioni hotel e analisi sentiment (ML)

Repository minimale richiesto dalla commissione: **codice + dati + istruzioni**.

## Struttura
- `src/` → script Python
- `dataset/` → CSV sintetico
- `models/` → modelli salvati dopo il training
- `results/` → grafici e output predizioni

## 1) Generare il dataset (sintetico)
```bash
python src/generate_dataset.py --n 600 --out dataset/synthetic_reviews.csv --seed 42
```

Il CSV prodotto contiene:
`id, title, body, department, sentiment`

## 2) Training + valutazione
```bash
python src/hotel_review_ml_system.py train --csv dataset/synthetic_reviews.csv
```

Output:
- stampa **accuracy** e **F1 macro**
- salva le confusion matrix:
  - `results/cm_department.png`
  - `results/cm_sentiment.png`
- salva i modelli:
  - `models/model_department.joblib`
  - `models/model_sentiment.joblib`

## 3) Predizione batch (CSV → CSV)
```bash
python src/hotel_review_ml_system.py predict --in dataset/synthetic_reviews.csv --out results/predictions.csv
```

Aggiunge:
- `pred_department`, `pred_sentiment`
- `prob_department`, `prob_sentiment`

## 4) Dashboard (opzionale, locale)
```bash
pip install -r requirements.txt
python src/dashboard_gradio.py
```

### Note
- Dataset sintetico: nessun dato personale.
- Modello base (TF‑IDF + Logistic Regression) per chiarezza e riproducibilità.
