"""OfferCheck backend: calibrated listing-risk screening + deterministic cues.

Serves the static UI and POST /api/screen. Never returns a safe/legitimate
verdict; lowest band is explicitly "no strong signals found".
"""
import os
import re
import joblib
import numpy as np
from scipy import sparse
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cues import extract_cues

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = joblib.load(os.path.join(BASE, "..", "model", "model.joblib"))

MIN_CHARS = 120  # below this we abstain: not enough text to screen

app = FastAPI(title="OfferCheck", docs_url=None, redoc_url=None)


class ScreenRequest(BaseModel):
    text: str


def band(p: float) -> dict:
    if p >= 0.30:
        return {
            "band": "high",
            "label": "Multiple strong risk signals",
            "detail": "This listing closely matches confirmed scam patterns in the model's training data. Treat it as untrustworthy unless you can independently verify the employer.",
        }
    if p >= 0.08:
        return {
            "band": "uncertain",
            "label": "Some risk signals present",
            "detail": "The model found partial matches to known scam patterns but is not confident. Verify the employer independently before sharing anything.",
        }
    return {
        "band": "low",
        "label": "No strong scam signals found",
        "detail": "Nothing in this text strongly matches the model's known scam patterns. This is NOT proof the listing is legitimate - new scams may not match 2012-2014 patterns. Verify the company independently before applying.",
    }


def model_prob(text: str) -> float:
    X = sparse.hstack([
        MODEL["tfidf"].transform([text]),
        MODEL["catvec"].transform(["unknown | unknown | unknown"]),
        sparse.csr_matrix(np.zeros((1, 3))),
    ]).tocsr()
    raw = MODEL["clf"].predict_proba(X)[0, 1]
    return float(MODEL["iso"].predict([raw])[0])


CHECKLIST = [
    "Search the company name + \"scam\" or \"reviews\" - check results beyond the first link.",
    "Find the company's official website and confirm the role is listed there.",
    "Check the recruiter's email domain matches the company's real domain.",
    "Never pay any fee (registration, training, equipment, background check) to get a job.",
    "Never share bank login, full card details, OTPs, or ID documents at application stage.",
    "Be suspicious of interviews conducted only over WhatsApp/Telegram chat.",
    "If pay looks far above market for no experience, treat it as a lure until proven otherwise.",
]


@app.post("/api/screen")
def screen(req: ScreenRequest):
    text = (req.text or "").strip()
    plain = re.sub(r"\s+", " ", text)
    if len(plain) < MIN_CHARS:
        return {
            "verdict": "abstain",
            "label": "Not enough text to screen",
            "detail": "Paste the full listing (title, company profile, description, requirements). Short fragments do not give the model enough evidence either way.",
            "cues": [], "checklist": CHECKLIST, "signals": None,
        }
    p = model_prob(plain)
    cues = extract_cues(plain)
    b = band(p)
    return {
        "verdict": "screened",
        "band": b["band"],
        "label": b["label"],
        "detail": b["detail"],
        "cues": cues,
        "checklist": CHECKLIST,
        "signals": {
            "model_band": b["band"],
            "cue_count": len(cues),
            "high_severity_cues": sum(1 for c in cues if c["severity"] == "high"),
        },
        "method": "Calibrated text classifier trained on 17,880 labeled job ads (EMSCAD, 2012-2014, CC0). Held-out test PR-AUC 0.90. Bands reflect calibrated probabilities, not certainty.",
    }


@app.get("/api/health")
def health():
    return {"ok": True}


static_dir = os.path.join(BASE, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(static_dir, "index.html"))
