#!/usr/bin/env python3
"""generate_dataset.py
Genera un dataset sintetico di recensioni alberghiere (titolo+testo) con etichette:
- department: Housekeeping / Reception / Food & Beverage
- sentiment: positivo / negativo

Esporta un CSV con colonne: id, title, body, department, sentiment

Uso:
  python src/generate_dataset.py --n 600 --out dataset/synthetic_reviews.csv --seed 42
"""
import argparse, random, uuid, csv, os

TEMPLATES = {
    "Housekeeping": {
        "positivo": [
            ("Camera perfetta", "Camera pulita e profumata, bagno in ordine, asciugamani freschi."),
            ("Pulizie impeccabili", "Pulizia accurata ogni giorno, lenzuola pulite e stanza ordinata."),
            ("Bagno curato", "Bagno davvero pulito, doccia brillante e prodotti ben disposti."),
        ],
        "negativo": [
            ("Camera sporca", "Polvere sui mobili e lenzuola macchiate, pulizia insufficiente."),
            ("Bagno trascurato", "Doccia incrostata e cattivo odore, bagno non pulito come dovrebbe."),
            ("Igiene scarsa", "Ho trovato capelli nel bagno e cestino non svuotato."),
        ],
    },
    "Reception": {
        "positivo": [
            ("Check-in veloce", "Arrivo rapido, check-in in pochi minuti e staff cordiale."),
            ("Accoglienza ottima", "Personale disponibile, spiegazioni chiare e assistenza immediata."),
            ("Supporto perfetto", "Alla reception hanno risolto subito una richiesta senza problemi."),
        ],
        "negativo": [
            ("Check-in lento", "Attesa lunga per le chiavi e poca organizzazione al banco."),
            ("Problemi prenotazione", "Errore nella prenotazione e tempi lunghi per sistemare tutto."),
            ("Poca disponibilità", "Personale poco cortese e difficile ottenere informazioni."),
        ],
    },
    "Food & Beverage": {
        "positivo": [
            ("Colazione abbondante", "Buffet ricco e vario, prodotti freschi e buon caffè."),
            ("Ristorante ottimo", "Piatti ben preparati e servizio attento in sala."),
            ("Prodotti di qualità", "Colazione con dolci buoni e scelta ampia, tutto molto curato."),
        ],
        "negativo": [
            ("Colazione scarsa", "Poca scelta e prodotti non freschi, esperienza deludente."),
            ("Cibo freddo", "Piatti arrivati tiepidi e buffet poco rifornito."),
            ("Servizio lento", "Attesa troppo lunga al ristorante e personale distratto."),
        ],
    },
}

def generate(n: int, ambiguity_ratio: float, seed: int):
    random.seed(seed)
    depts = list(TEMPLATES.keys())
    sents = ["positivo", "negativo"]
    rows = []
    for _ in range(n):
        dept = random.choice(depts)
        sent = random.choice(sents)
        title, body = random.choice(TEMPLATES[dept][sent])

        if random.random() < ambiguity_ratio:
            other_dept = random.choice([d for d in depts if d != dept])
            other_sent = random.choice(sents)
            _, extra = random.choice(TEMPLATES[other_dept][other_sent])
            body = body + " Inoltre, " + extra.lower()

        rows.append({
            "id": str(uuid.uuid4()),
            "title": title,
            "body": body,
            "department": dept,
            "sentiment": sent
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--ambiguity", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="dataset/synthetic_reviews.csv")
    args = ap.parse_args()

    rows = generate(args.n, args.ambiguity, args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","title","body","department","sentiment"])
        w.writeheader()
        w.writerows(rows)

    print(f"✅ Dataset esportato: {args.out} (righe: {len(rows)})")

if __name__ == "__main__":
    main()
