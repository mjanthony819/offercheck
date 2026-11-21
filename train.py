"""Train + calibrate + evaluate the OfferCheck listing-risk model on EMSCAD.

Honesty rules baked in:
- Stratified train/calibration/test split; ALL reported metrics are on the
  untouched test split. Calibration (isotonic) is fitted on the calibration
  split only, never on test.
- We report PR-AUC (the meaningful metric under 4.8% prevalence), ROC-AUC,
  Brier score, and an empirical calibration table. Raw probabilities are
  never exposed to users; the app shows calibrated bands + abstention.
"""
import json
import os
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/hoangsonww/Spot-the-Scam-AI-Job-Fraud/master/data/fake_job_postings.csv"
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fake_job_postings.csv")

if not os.path.exists(DATA_PATH):
    # EMSCAD via public CC0 mirror (see README for source + license)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    print("Downloading EMSCAD (CC0) from public mirror...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve,
)
import joblib

TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
STRUCT_COLS = ["telecommuting", "has_company_logo", "has_questions"]
CAT_COLS = ["employment_type", "required_experience", "required_education"]

df = pd.read_csv(DATA_PATH)
df["fraudulent"] = df["fraudulent"].astype(int)

for c in TEXT_COLS:
    df[c] = df[c].fillna("")
df["text"] = (
    df["title"] + "\n" + df["company_profile"] + "\n" + df["description"]
    + "\n" + df["requirements"] + "\n" + df["benefits"]
)
for c in CAT_COLS:
    df[c] = df[c].fillna("unknown")

X_text = df["text"].values
X_struct = df[STRUCT_COLS].fillna(0).astype(float).values
X_cat = df[CAT_COLS].astype(str).agg(" | ".join, axis=1).values
y = df["fraudulent"].values

# 60/20/20 stratified split
idx = np.arange(len(df))
i_train, i_tmp, y_train, y_tmp = train_test_split(
    idx, y, test_size=0.4, random_state=42, stratify=y
)
i_cal, i_test, y_cal, y_test = train_test_split(
    i_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
)

tfidf = TfidfVectorizer(
    max_features=60000, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
    strip_accents="unicode", lowercase=True,
)
catvec = TfidfVectorizer(max_features=500, lowercase=True)

Xtr = sparse.hstack([
    tfidf.fit_transform(X_text[i_train]),
    catvec.fit_transform(X_cat[i_train]),
    sparse.csr_matrix(X_struct[i_train]),
]).tocsr()
Xca = sparse.hstack([
    tfidf.transform(X_text[i_cal]),
    catvec.transform(X_cat[i_cal]),
    sparse.csr_matrix(X_struct[i_cal]),
]).tocsr()
Xte = sparse.hstack([
    tfidf.transform(X_text[i_test]),
    catvec.transform(X_cat[i_test]),
    sparse.csr_matrix(X_struct[i_test]),
]).tocsr()

clf = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced", solver="lbfgs")
clf.fit(Xtr, y_train)

# Raw scores on calibration split -> fit isotonic calibrator
raw_cal = clf.predict_proba(Xca)[:, 1]
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(raw_cal, y_cal)

raw_test = clf.predict_proba(Xte)[:, 1]
cal_test = iso.predict(raw_test)

metrics = {
    "n_total": int(len(df)),
    "n_fraud": int(y.sum()),
    "prevalence": round(float(y.mean()), 4),
    "split": {"train": len(i_train), "calibration": len(i_cal), "test": len(i_test)},
    "test": {
        "pr_auc": round(float(average_precision_score(y_test, cal_test)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, cal_test)), 4),
        "brier": round(float(brier_score_loss(y_test, cal_test)), 4),
        "brier_baseline": round(float(brier_score_loss(y_test, np.full_like(y_test, y_test.mean()))), 4),
    },
}

# Empirical calibration table on test (deciles of calibrated prob)
bins = np.quantile(cal_test, np.linspace(0, 1, 11))
bins[0], bins[-1] = 0.0, 1.0
table = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (cal_test >= lo) & (cal_test <= hi)
    if m.sum() > 0:
        table.append({
            "pred_mean": round(float(cal_test[m].mean()), 4),
            "empirical_rate": round(float(y_test[m].mean()), 4),
            "n": int(m.sum()),
        })
metrics["calibration_table_test"] = table

# Precision/recall at candidate operating thresholds (calibrated prob)
prec, rec, thr = precision_recall_curve(y_test, cal_test)
ops = []
for t in [0.15, 0.3, 0.5, 0.7]:
    m = cal_test >= t
    if m.sum() > 0:
        ops.append({
            "threshold": t,
            "precision": round(float(y_test[m].mean()), 4),
            "recall": round(float((m & (y_test == 1)).sum() / y_test.sum()), 4),
            "flag_rate": round(float(m.mean()), 4),
        })
metrics["operating_points_test"] = ops

joblib.dump(
    {"tfidf": tfidf, "catvec": catvec, "clf": clf, "iso": iso,
     "text_cols": TEXT_COLS, "struct_cols": STRUCT_COLS, "cat_cols": CAT_COLS},
    "/downloads/offercheck/model/model.joblib",
)
with open("/downloads/offercheck/model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps({k: metrics[k] for k in ["n_total", "n_fraud", "prevalence", "split", "test"]}, indent=2))
print("calibration (pred_mean -> empirical_rate):",
      [(r["pred_mean"], r["empirical_rate"], r["n"]) for r in table])
print("operating points:", ops)
