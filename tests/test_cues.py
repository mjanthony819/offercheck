"""Contract tests for the deterministic cue extractor (app/cues.py).

Zero-dependency: runs as a plain script (python tests/test_cues.py) or under
pytest. The cue extractor is the transparent half of OfferCheck: every cue
must show the exact matched text, fire independently of the ML model, and
suppress anti-scam reassurance ("we never ask for a registration fee").
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))

from cues import extract_cues


def ids(cues):
    return [c["id"] for c in cues]


def test_every_family_fires_on_canonical_phrase():
    cases = {
        "payment_request": "A registration fee of $50 is required before you start.",
        "financial_data": "Please send your bank account number and CVV to proceed.",
        "free_email": "Contact the hiring manager at hr.recruiter2026@gmail.com today.",
        "chat_app_contact": "Message us on WhatsApp to schedule your interview.",
        "urgency_pressure": "Urgent hiring, apply immediately, limited slots available.",
        "too_good_pay": "Earn $500 per day from your phone, guaranteed income.",
        "vague_identity": "We are a confidential company hiring for a confidential client.",
        "no_interview": "No interview needed, this is a direct joining offer.",
        "work_from_home_lure": "Work from home and earn money from home easily.",
        "generic_greeting": "Dear candidate, you have been shortlisted.",
    }
    for cue_id, text in cases.items():
        found = ids(extract_cues(text))
        assert cue_id in found, f"{cue_id} did not fire on: {text!r} (got {found})"


def test_match_shows_exact_original_text():
    text = "URGENT: send a REGISTRATION FEE via Western Union."
    cues = extract_cues(text)
    payment = next(c for c in cues if c["id"] == "payment_request")
    matched = [m["text"] for m in payment["matches"]]
    assert "REGISTRATION FEE" in matched, f"original casing not preserved: {matched}"
    assert any("Western Union" in m for m in matched), matched
    for c in cues:
        for m in c["matches"]:
            assert text[m["start"]:m["end"]] == m["text"], "span does not match text"


def test_cue_fields_complete_and_hits_capped():
    text = " ".join(["registration fee"] * 12)
    cues = extract_cues(text)
    payment = next(c for c in cues if c["id"] == "payment_request")
    assert len(payment["matches"]) <= 5, "matches must be capped at 5"
    for c in cues:
        assert set(c) == {"id", "label", "severity", "advice", "matches"}, set(c)
        assert c["severity"] in ("high", "medium", "low")
        assert c["label"] and c["advice"]


def test_sorted_by_severity():
    text = ("Dear friend, work from home! Contact hr@gmail.com on WhatsApp. "
            "Registration fee required, no interview needed.")
    cues = extract_cues(text)
    rank = {"high": 0, "medium": 1, "low": 2}
    sevs = [rank[c["severity"]] for c in cues]
    assert sevs == sorted(sevs), f"not sorted by severity: {sevs}"
    assert cues[0]["severity"] == "high"


def test_negated_reassurance_is_suppressed():
    suppressed = [
        "We never ask for a registration fee",
        "No registration fee is required",
        "We do not charge any training fee",
        "There is no registration fee and no processing fee",
        "You won't pay any application fee here",
    ]
    for text in suppressed:
        assert "payment_request" not in ids(extract_cues(text)), f"should be suppressed: {text!r}"


def test_negation_does_not_cross_sentence_boundary():
    text = "We never ask for money. Registration fee required before onboarding."
    assert "payment_request" in ids(extract_cues(text)), "negation leaked across a full stop"


def test_real_charge_still_fires_after_boundary():
    text = "Free of charge. Wire transfer needed for onboarding."
    assert "payment_request" in ids(extract_cues(text))


def test_case_insensitive():
    assert "chat_app_contact" in ids(extract_cues("WHATSAPP only interviews"))
    assert "free_email" in ids(extract_cues("write to HR@GMAIL.COM"))


def test_clean_listing_produces_no_cues():
    text = (
        "Software Engineer, Platform Team. We are looking for an engineer with "
        "3+ years of experience in distributed systems and Python. You will design, "
        "build and operate services that process millions of requests per day. "
        "Our interview process: one recruiter call, two technical rounds, one team "
        "fit conversation. We offer competitive compensation, health insurance and "
        "a learning budget. Apply through our careers page."
    )
    assert extract_cues(text) == [], extract_cues(text)


def main():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok   {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
    assert failed == 0, f"{failed} failures"
    print(f"cue extractor OK: {len(tests)} contract tests")


if __name__ == "__main__":
    main()
