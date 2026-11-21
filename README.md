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
   from 2012-2014; provenance below). Probabilities are
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
- `data/fake_job_postings.csv` - EMSCAD (provenance below)

## Data source and provenance

The model is trained on EMSCAD (Employment Scam Aegean Dataset): 17,880 real
job ads posted on Workable between 2012 and 2014, 866 labeled fraudulent.
EMSCAD was released by University of the Aegean researchers alongside Vidros
et al., "Automatic Detection of Online Recruitment Frauds: Characteristics,
Methods, and a Public Dataset", Future Internet 2017, 9(1), 6 (MDPI, open
access), which describes it as a publicly available dataset.

Provenance of the copy used here: a public GitHub mirror
(hoangsonww/Spot-the-Scam-AI-Job-Fraud) of the widely used Kaggle mirror
"Recruitment Scam" (kaggle.com/datasets/amruthjithrajvr/recruitment-scam),
whose page designates the dataset "CC0: Public Domain" (checked Nov 2026).
The authors' original download page (emscad.samos.aegean.gr) is no longer
reachable, so we cannot independently confirm the original distribution
terms; we rely on the mirror's designation and the paper's description of
the dataset as public.

The raw ads are not republished in this repository: the CSV is git-ignored
and `train.py` downloads it from the mirror above at training time. The demo
and screenshots use only synthetic listings. The trained artifact
(`model/model.joblib`) embeds a TF-IDF vocabulary derived from the ads -
individual tokens and bigrams, not the ads themselves.

## Development disclosure

Built during the Global AI Builders Challenge window (Nov 21-28, 2026).
Model, calibration, evaluation, backend, cue extractor and UI were created in
this window. AI coding assistance (Instinct) was used for scaffolding and
iteration; all modeling, evaluation design, and copy decisions were reviewed
for the honesty constraints above. No pre-existing project code was reused.

## License

MIT (code). Dataset: see provenance above.
