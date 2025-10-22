from __future__ import annotations

from typing import TYPE_CHECKING
import re

from langchain_core.messages import HumanMessage
from scenario_detector import detect, initialize as scenario_init, is_ready as scenario_ready

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

    # Ensure detector is initialized (lazy fallback if not pre-initialized)
    if not scenario_ready():
        try:
            scenario_init(backend._scenarios, enabled=True)
        except Exception:
            pass

    # Run detection and parse its compact output
    try:
        print(f"[CLASSIFIER] Query preview: '{messages_string[:50]}...'")
        result = detect(last_user_message or "")
        print(result)

        # Expected format:
        # [CLASSIFIER] TF-IDF top score=0.123, margin=0.045, match='LABEL', certainty=8/10
        m = re.search(r"match='([^']+)'\,\s+certainty=(\d+)/10", result)
        if m:
            chosen_label = m.group(1) or "OTHER"
            certainty = int(m.group(2)) if m.group(2) else 2
        else:
            chosen_label = "OTHER"
            certainty = 2

        return {
            "query_type": chosen_label,
            "certainty": certainty,
            "awaiting_confirmation": False,
        }

    except Exception as e:
        print(f"[CLASSIFIER] Detection error: {e}")
        return {
            "query_type": "OTHER",
            "certainty": 2,
            "awaiting_confirmation": False,
        }
