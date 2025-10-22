from __future__ import annotations

from typing import TYPE_CHECKING, List
import re

from langchain_core.messages import HumanMessage

# We'll prefer TF-IDF search over LLM classification.
# These imports are local to avoid import errors at import-time if packages are missing.
def _import_tfidf():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        import numpy as np  # type: ignore

        return TfidfVectorizer, np
    except Exception as e:  # pragma: no cover
        print(f"[CLASSIFIER] TF-IDF deps not available: {e}")
        return None, None

if TYPE_CHECKING:
    from flow_backend import ConversationState, FlowBackend


def classifier_node(backend: "FlowBackend", state: "ConversationState") -> dict:
    """
    Classifies the latest user message using fast TF-IDF similarity over
    scenario examples, returning a scenario question_type and a 1-10 certainty.
    """
    messages = state["messages"]
    recent_messages = messages[-6:] if len(messages) > 6 else messages
    messages_string = backend._messages_to_string(recent_messages)
    last_user_message = next(
        (msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)),
        "",
    )

    # Build corpus of scenario example texts
    scenarios = backend._scenarios
    scenario_texts: List[str] = []
    scenario_examples: List[str] = []
    for sc in scenarios:
        # Combine example and description to enrich matching
        ex = str(sc.get("question_example", ""))
        desc = str(sc.get("question_description", ""))
        text = (ex + " \n " + desc).strip()
        scenario_texts.append(text)
        scenario_examples.append(ex)

    TfidfVectorizer, np = _import_tfidf()
    if TfidfVectorizer is None or np is None:
        # Dependencies missing: degrade gracefully
        print("[CLASSIFIER] Falling back due to missing TF-IDF dependencies")
        return {
            "query_type": "OTHER",
            "certainty": 2,
            "awaiting_confirmation": False,
        }

    try:
        # Fit TF-IDF on scenario corpus
        vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), norm="l2")
        X = vect.fit_transform(scenario_texts)

        # Transform the user query
        Xq = vect.transform([last_user_message or ""])
        sim = (Xq @ X.T).toarray().ravel()

        if sim.size == 0:
            return {"query_type": "OTHER", "certainty": 2, "awaiting_confirmation": False}

        order = np.argsort(-sim)
        top_idx = int(order[0])
        top_score = float(sim[top_idx])
        second_score = float(sim[order[1]]) if sim.size > 1 else 0.0
        margin = top_score - second_score

        chosen_label = scenario_examples[top_idx] if top_score > 0 else "OTHER"

        # Threshold for "OTHER" to avoid spurious matches
        if top_score < 0.12:
            chosen_label = "OTHER"

        # Heuristics: exact/near-exact match handling for higher certainty
        def _normalize(s: str) -> str:
            s = s.lower().strip()
            s = re.sub(r"[\s\-_,.;:!?]+", " ", s)
            return s

        def _tokens(s: str) -> set[str]:
            return set(re.findall(r"[a-z0-9]+", s.lower()))

        q_norm = _normalize(last_user_message or "")
        ex_norm = _normalize(scenario_examples[top_idx]) if top_score > 0 else ""
        tok_q = _tokens(q_norm)
        tok_ex = _tokens(ex_norm)

        jacc = (len(tok_q & tok_ex) / len(tok_q | tok_ex)) if (tok_q or tok_ex) else 0.0
        substr = (q_norm == ex_norm) or (q_norm in ex_norm) or (ex_norm in q_norm)

        # Certainty scaling (1-10): keep 9-10 for very confident matches
        # Base mapping from similarity with a soft floor at 0.15 and cap near 0.75
        def scale_certainty(score: float, margin: float) -> int:
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

        certainty = scale_certainty(top_score, margin)
        # Upgrades for exact/near-exact
        if q_norm and ex_norm:
            if q_norm == ex_norm:
                certainty = 10
            elif jacc >= 0.9 or (substr and jacc >= 0.8):
                certainty = max(certainty, 10 if margin >= 0.2 else 9)
            elif (top_score >= 0.45 and margin >= 0.25) or (jacc >= 0.75 and substr):
                certainty = max(certainty, 9)

        print(f"[CLASSIFIER] Query preview: '{messages_string[:50]}...'")
        print(
            f"[CLASSIFIER] TF-IDF top score={top_score:.3f}, margin={margin:.3f}, match='{chosen_label}', certainty={certainty}/10"
        )

        return {
            "query_type": chosen_label,
            "certainty": certainty,
            "awaiting_confirmation": False,
        }

    except Exception as e:
        print(f"[CLASSIFIER] TF-IDF classification error: {e}")
        return {
            "query_type": "OTHER",
            "certainty": 2,
            "awaiting_confirmation": False,
        }
