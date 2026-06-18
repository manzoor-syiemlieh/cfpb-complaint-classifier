# Consumer Complaint Classifier — Product Triage

A classical-NLP model that reads a consumer-finance complaint and routes it to the right product team. Trained on real complaints from the US Consumer Financial Protection Bureau (CFPB), it predicts which of seven financial products a complaint is about, directly from the free-text narrative.

Built by a fraud & risk professional with 7+ years at PayPal adjudicating disputes and complaints — so the framing is operational: this is the triage tool a complaints queue actually needs.

## Problem

A large institution receives thousands of free-text complaints a day. Routing them to the correct team by hand is slow and inconsistent. The task here: given only the complaint narrative, predict the product category so the complaint can be auto-routed.

## Data

[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) — public. The full file holds 3.8M complaints that include a published narrative, across many overlapping product names. These were consolidated into **7 product groups**: Credit reporting, Debt collection, Card, Bank account, Mortgage, Money transfer, Student loan.

The raw CSV is large and is not committed (see `.gitignore`); download it from the link above and place it at `data/complaints.csv` to re-run.

## Approach

- **Balance the classes by sampling.** "Credit reporting" alone is 2.4M rows, so each class is sampled to at most 10,000. A balanced training set means the model isn't just predicting the dominant class, and macro-F1 reflects every product fairly.
- **Classical NLP pipeline.** Lowercase, remove CFPB `xxxx` redactions and numbers, strip punctuation, tokenize, remove stopwords, lemmatize.
- **TF-IDF features** (unigrams + bigrams, 20,000-term vocabulary), fit on the training split only.
- **Logistic Regression** with balanced class weights — a strong, interpretable baseline for sparse text. Linear models outperform tree ensembles on high-dimensional sparse text features.

The train/test split is done **before** vectorizing, and TF-IDF is fit on training text only, so no test-set vocabulary leaks into training.

## Results

| Metric | Value |
|---|---|
| Macro-F1 (7 classes) | **0.86** |
| Test set | 14,000 (2,000 per class) |

Per-class F1 ranges from 0.80 (Bank account) to 0.94 (Mortgage, Student loan).

**Why macro-F1, not accuracy:** on a multi-class triage problem, macro-F1 weights every product equally, so a strong score means the model handles small categories as well as large ones — not just the common ones.

### The model learned meaningful signal

Top tokens per class (from the model's coefficients) are domain-sensible, which is the evidence the model isn't guessing:

- **Mortgage** → escrow, foreclosure, refinance, modification
- **Debt collection** → debt, collection, owe, collector
- **Money transfer** → paypal, zelle, venmo, cashapp, coinbase
- **Student loan** → navient, mohela, nelnet, aidvantage
- **Card** → card, amex, citi, discover, synchrony

## Repo structure

| Path | What it is |
|---|---|
| `Customer_Complaint_Classifier.ipynb` | Full pipeline: load, clean, sample, TF-IDF, model, evaluation, interpretability |
| `model/tfidf.joblib` | Fitted TF-IDF vectorizer |
| `model/classifier.joblib` | Trained Logistic Regression model |
| `requirements.txt` | Dependencies |

## Run it

```bash
git clone https://github.com/manzoor-syiemlieh/cfpb-complaint-classifier.git
cd cfpb-complaint-classifier
pip install -r requirements.txt
# download complaints.csv from CFPB into data/, then run the notebook top to bottom
```

## Limitations & next steps

- Only complaints **with a published narrative** are modelled — a subset of all complaints filed.
- "Credit reporting" was downsampled for class balance; a production system would weigh real-world class frequencies.
- TF-IDF ignores word order and meaning; the next step is word embeddings or a transformer model for semantic understanding.
- This is a **prototype** triage classifier, not a production system (no live serving, monitoring, or retraining).

## Author

**Manzoor Syiemlieh** — Fraud & Risk Analytics → Data Science · BFSI & Fintech · 7+ yrs @ PayPal
[LinkedIn](https://www.linkedin.com/in/manzoor-syiemlieh-4193683a5/) · [GitHub](https://github.com/manzoor-syiemlieh)

MIT Licensed.
