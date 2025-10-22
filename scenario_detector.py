from __future__ import annotations

"""
TF-IDF based scenario detector.

- initialize(scenarios, enabled=True): optionally pre-fits the vectorizer.
- detect(text): returns only the formatted classification string.

This module keeps vectorizer state in-process so callers can pre-warm it at
startup or let it initialize lazily when first used.
"""

from typing import Any, List
import re

# Globals for cached vectorizer and scenario corpus
_vectorizer = None  # type: ignore[var-annotated]
_X = None  # type: ignore[var-annotated]
_examples: List[str] = []
_ready: bool = False


def _import_tfidf():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        import numpy as np  # type: ignore

        return TfidfVectorizer, np
    except Exception as e:  # pragma: no cover
        print(f"[CLASSIFIER] TF-IDF deps not available: {e}")
        return None, None


def initialize(scenarios: List[dict[str, Any]] | None, *, enabled: bool = True) -> bool:
    """
    Optionally pre-fit the vectorizer and cache scenario corpus.

    Returns True if ready, False otherwise.
    """
    global _vectorizer, _X, _examples, _ready

    if not enabled:
        return False

    if _ready:
        return True

    if not scenarios:
        return False

    TfidfVectorizer, np = _import_tfidf()
    if TfidfVectorizer is None or np is None:
        return False

    scenario_texts: List[str] = []
    examples: List[str] = []
    for sc in scenarios:
        ex = str(sc.get("question_example", ""))
        desc = str(sc.get("question_description", ""))
        text = (ex + " \n " + desc).strip()
        scenario_texts.append(text)
        examples.append(ex)

    try:
        vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), norm="l2")
        X = vect.fit_transform(scenario_texts)
    except Exception as e:
        print(f"[CLASSIFIER] TF-IDF init error: {e}")
        return False

    _vectorizer = vect
    _X = X
    _examples = examples
    _ready = True
    print("[CLASSIFIER] TF-IDF vectorizer initialized")
    return True


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\s\-_,.;:!?]+", " ", s)
    return s


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _scale_certainty(score: float, margin: float) -> int:
    # Normalize score to [0, 1] from [0.12, 0.70]
    lo, hi = 0.12, 0.70
    if score <= lo:
        base = 0.0
    elif score >= hi:
        base = 1.0
    else:
        base = (score - lo) / (hi - lo)
    # Emphasize high-confidence end
    curved = base ** 1.35
    # Margin bonus for clear separation
    bonus = 0.0
    if margin >= 0.30:
        bonus = 0.20
    elif margin >= 0.20:
        bonus = 0.12
    elif margin >= 0.10:
        bonus = 0.06
    curved = min(1.0, curved + bonus)
    # Map to 1..10, reserving 9-10 for very high
    raw = 1 + round(9 * curved)
    # Ensure only strongest get 9-10, but allow strong margin to lift
    if raw >= 9 and not (score >= 0.50 or margin >= 0.25):
        raw = 8
    return int(max(1, min(10, raw)))


def detect(text: str) -> str:
    """
    Detect scenario for the input text using the cached TF-IDF model.

    Returns only the formatted string:
    f"[CLASSIFIER] TF-IDF top score={top_score:.3f}, margin={margin:.3f}, match='{chosen_label}', certainty={certainty}/10"
    """
    global _vectorizer, _X, _examples, _ready

    TfidfVectorizer, np = _import_tfidf()
    if TfidfVectorizer is None or np is None or not _ready:
        # Fallback: not initialized or missing deps
        top_score = 0.0
        margin = 0.0
        chosen_label = "OTHER"
        certainty = 2
        return (
            f"[CLASSIFIER] TF-IDF top score={top_score:.3f}, margin={margin:.3f}, match='{chosen_label}', certainty={certainty}/10"
        )

    try:
        Xq = _vectorizer.transform([text or ""])  # type: ignore[attr-defined]
        sim = (Xq @ _X.T).toarray().ravel()  # type: ignore[operator]
        if sim.size == 0:
            top_score = 0.0
            margin = 0.0
            chosen_label = "OTHER"
            certainty = 2
            return (
                f"[CLASSIFIER] TF-IDF top score={top_score:.3f}, margin={margin:.3f}, match='{chosen_label}', certainty={certainty}/10"
            )

        order = np.argsort(-sim)
        top_idx = int(order[0])
        top_score = float(sim[top_idx])
        second_score = float(sim[order[1]]) if sim.size > 1 else 0.0
        margin = top_score - second_score

        chosen_label = _examples[top_idx] if top_score > 0 else "OTHER"
        if top_score < 0.12:
            chosen_label = "OTHER"

        q_norm = _normalize(text or "")
        ex_norm = _normalize(_examples[top_idx]) if top_score > 0 else ""
        tok_q = _tokens(q_norm)
        tok_ex = _tokens(ex_norm)
        jacc = (len(tok_q & tok_ex) / len(tok_q | tok_ex)) if (tok_q or tok_ex) else 0.0
        substr = (q_norm == ex_norm) or (q_norm in ex_norm) or (ex_norm in q_norm)

        certainty = _scale_certainty(top_score, margin)
        if q_norm and ex_norm:
            if q_norm == ex_norm:
                certainty = 10
            elif jacc >= 0.9 or (substr and jacc >= 0.8):
                certainty = max(certainty, 10 if margin >= 0.2 else 9)
            elif (top_score >= 0.45 and margin >= 0.25) or (jacc >= 0.75 and substr):
                certainty = max(certainty, 9)

        return (
            f"[CLASSIFIER] TF-IDF top score={top_score:.3f}, margin={margin:.3f}, match='{chosen_label}', certainty={certainty}/10"
        )

    except Exception as e:  # pragma: no cover
        print(f"[CLASSIFIER] TF-IDF detection error: {e}")
        top_score = 0.0
        margin = 0.0
        chosen_label = "OTHER"
        certainty = 2
        return (
            f"[CLASSIFIER] TF-IDF top score={top_score:.3f}, margin={margin:.3f}, match='{chosen_label}', certainty={certainty}/10"
        )


def is_ready() -> bool:
    return _ready

