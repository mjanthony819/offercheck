"""Deterministic red-flag cue extractor.

Every cue is a literal, inspectable pattern matched against the listing text -
not model output - so the user can see exactly why it fired. The ML score is
reported separately as a calibrated band. Neither is a verdict.
"""
import re

CUES = [
    {
        "id": "payment_request",
        "label": "Asks for money",
        "severity": "high",
        "advice": "Legitimate employers do not charge application, training, registration or equipment fees.",
        "patterns": [
            r"registration fee", r"processing fee", r"training fee", r"application fee",
            r"starter kit", r"background check fee", r"refundable deposit", r"security deposit",
            r"pay\s*(?:a|the)?\s*(?:small\s*)?fee", r"payment is required", r"advance payment",
            r"western union", r"moneygram", r"money gram", r"wire transfer", r"gift card",
            r"itunes card", r"cryptocurrency", r"bitcoin", r"usdt",
        ],
    },
    {
        "id": "financial_data",
        "label": "Requests sensitive financial or identity data",
        "severity": "high",
        "advice": "Never share bank logins, full card numbers, OTPs or ID documents at application stage.",
        "patterns": [
            r"bank account (?:number|details)", r"bank details", r"credit card (?:number|details)",
            r"card verification", r"cvv", r"one[- ]time password", r"otp",
            r"social security number", r"\bssn\b", r"passport (?:number|copy)",
            r"copy of your (?:passport|id|driver)", r"aadhaar", r"pan card",
        ],
    },
    {
        "id": "free_email",
        "label": "Recruiter uses a free webmail address",
        "severity": "medium",
        "advice": "Established companies almost always recruit from their own domain. Verify the company independently.",
        "patterns": [
            r"@[a-z0-9.-]*(?:gmail|yahoo|hotmail|outlook|aol|protonmail|rediffmail|yandex)\.[a-z.]{2,}",
        ],
    },
    {
        "id": "chat_app_contact",
        "label": "Moves contact to WhatsApp/Telegram",
        "severity": "medium",
        "advice": "Interviews conducted only over chat apps are a common scam pattern.",
        "patterns": [
            r"whatsapp", r"telegram", r"signal app", r"contact (?:us|me) on (?:the )?(?:app|chat)",
        ],
    },
    {
        "id": "urgency_pressure",
        "label": "Urgency or pressure language",
        "severity": "medium",
        "advice": "Pressure to act immediately is used to stop you from verifying.",
        "patterns": [
            r"urgent(?:ly)? (?:hiring|required|opening)", r"apply (?:immediately|now|today|asap)",
            r"limited (?:positions|slots|seats)", r"act now", r"don'?t miss",
            r"offer expires", r"immediate joining", r"within 24 hours",
        ],
    },
    {
        "id": "too_good_pay",
        "label": "Pay looks too good for the ask",
        "severity": "medium",
        "advice": "High guaranteed pay for no experience is a classic lure.",
        "patterns": [
            r"earn \$?\s?\d{3,4}\s*(?:per|/|a)\s*(?:day|week)", r"guaranteed income",
            r"\$\d{2,3}\s*(?:per|an|/)\s*hour", r"unlimited earning", r"earn thousands",
            r"no experience (?:needed|required).{0,60}(?:\$|salary|pay)", r"weekly payout",
        ],
    },
    {
        "id": "vague_identity",
        "label": "Vague or hidden company identity",
        "severity": "medium",
        "advice": "A real employer names itself. Search the company name plus 'scam' before engaging.",
        "patterns": [
            r"confidential (?:company|employer|client)", r"leading company.{0,20}cannot disclose",
            r"reputed (?:firm|company) (?:name )?(?:will be )?revealed", r"company name withheld",
        ],
    },
    {
        "id": "no_interview",
        "label": "Job offer without an interview",
        "severity": "high",
        "advice": "Real employers interview before offering a job.",
        "patterns": [
            r"no interview", r"without (?:an )?interview", r"direct joining",
            r"you are (?:hired|selected) without", r"immediate offer letter",
        ],
    },
    {
        "id": "work_from_home_lure",
        "label": "Generic work-from-home bait phrasing",
        "severity": "low",
        "advice": "Common filler in scam reposts; weak signal on its own.",
        "patterns": [
            r"work from (?:the comfort of )?(?:your )?home", r"earn money from home",
            r"part[- ]time job.{0,40}full[- ]time (?:income|pay)",
        ],
    },
    {
        "id": "generic_greeting",
        "label": "Mass-mail greeting",
        "severity": "low",
        "advice": "Real recruiters address you by name about a specific role.",
        "patterns": [r"dear (?:candidate|applicant|job seeker|sir/madam)", r"dear friend"],
    },
]

NEGATION_RE = re.compile(
    r"(never|no|not|don't|do not|doesn't|does not|without|free of|won't|will not|"
    r"none|zero|refuse|decline|avoid)\W{0,20}$",
    re.IGNORECASE,
)


def _negated(tl, start):
    """True when the ~50 chars before the match negate it (e.g. 'we never ask for')."""
    return bool(NEGATION_RE.search(tl[max(0, start - 50):start].strip()))


def extract_cues(text):
    tl = text.lower()
    out = []
    for cue in CUES:
        hits = []
        for pat in cue["patterns"]:
            for m in re.finditer(pat, tl, flags=re.IGNORECASE | re.DOTALL):
                if _negated(tl, m.start()):
                    continue  # anti-scam reassurance, not a scam signal
                hits.append({"start": m.start(), "end": m.end(), "text": text[m.start():m.end()]})
        if hits:
            out.append({k: cue[k] for k in ("id", "label", "severity", "advice")} | {"matches": hits[:5]})
    sev_rank = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda c: sev_rank[c["severity"]])
    return out
