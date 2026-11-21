# OfferCheck

**Paste a job or internship listing. Get an honest risk screen, not a verdict.**

OfferCheck screens job-listing text against patterns seen in confirmed job scams.
It combines a calibrated machine-learning model with a transparent set of
deterministic red-flag cues, and it always pairs its result with an independent
verification checklist. It never tells you a listing is "safe" or "legitimate" -
a low-risk result only means no strong known-scam patterns were found.

## Why

Students mass-apply to jobs and internships and are prime targets for
employment scams: registration-fee traps, fake recruiter webmail accounts,
WhatsApp-only "interviews", too-good-to-be-true pay. The usual advice is
"be careful". OfferCheck makes "careful" concrete: it highlights the exact
phrases that match scam patterns and tells you what to verify before you act.

## How it works

1. **Calibrated ML screen.** TF-IDF (1-2 grams) + logistic regression trained on
   the EMSCAD dataset (17,880 real job ads, 866 labeled fraudulent, postings
   from 2012-2014, CC0 public domain via the Kaggle mirror). Probabilities are
   isotonic-calibrated on a held-out calibration split. The app only shows
   risk bands derived from the calibrated probability - never raw scores.
2. **Deterministic cue extractor.** Nine literal, inspectable pattern families
   (payment requests, financial-data requests, free-webmail recruiters,
   chat-app contact, urgency pressure, too-good pay, vague company identity,
   no-interview offers, mass-mail greetings). Every cue shows the exact matched
   text. These fire independently of the model.
3. **Abstention.** Too little text (or no evidence either way) produces
   "Not enough text to screen", not a forced answer.
4. **Verification checklist.** Every result ends with concrete verification
   steps the user must do themselves.

## Honest limitations (read before trusting it)

- Training data is from **2012-2014**. Scam wording evolves; novel scams can be
  missed entirely. The model was **not** validated on 2026 personal emails,
  WhatsApp forwards, or internship offers, and we make no accuracy claims there.
- Held-out test metrics (same-distribution): **PR-AUC 0.90, ROC-AUC 0.99,
  Brier 0.011** (prevalence 4.8%). Precision ~0.90 at the high-risk band with
  ~0.80 recall on the test split. These describe performance on held-out EMSCAD
  data only.
- "No strong scam signals found" is **not** a safety guarantee.
- Pasted text is used only to compute the screening. It is not stored, logged,
  or sent anywhere else.

## Run it

```bash
pip install -r requirements.txt
python train.py          # trains, calibrates, evaluates, writes model/model.joblib
cd app && uvicorn main:app --port 8000
```

Open http://localhost:8000

## Repository layout

- `train.py` - data prep, stratified 60/20/20 split, training, isotonic
  calibration, held-out evaluation (writes `model/metrics.json`)
- `model/` - trained model artifact + evaluation metrics
- `app/main.py` - FastAPI backend (`/api/screen`)
- `app/cues.py` - deterministic red-flag cue extractor
- `app/static/index.html` - single-page UI (charcoal/lime)
- `data/fake_job_postings.csv` - EMSCAD (source + license below)

## Data source and license

EMSCAD (Employment Scam Aegean Dataset), published by the University of the
Aegean for research use; acquired via the Kaggle mirror
"kaggle.com/datasets/amruthjithrajvr/recruitment-scam", designated **CC0:
Public Domain**. Paper: Vidros et al., "Automatic Detection of Online
Recruitment Frauds: Characteristics, Methods, and a Public Dataset",
Future Internet 2017, 9(4), 59 (MDPI).

## Development disclosure

Built during the Global AI Builders Challenge window (Nov 21-28, 2026).
Model, calibration, evaluation, backend, cue extractor and UI were created in
this window. AI coding assistance (Instinct) was used for scaffolding and
iteration; all modeling, evaluation design, and copy decisions were reviewed
for the honesty constraints above. No pre-existing project code was reused.

## License

MIT (code). Dataset as above (CC0).
