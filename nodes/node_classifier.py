from __future__ import annotations

from typing import TYPE_CHECKING
import re

from langchain_core.messages import HumanMessage
from scenario_detector import (
    detect,
    initialize as scenario_init,
    is_ready as scenario_ready,
)

if TYPE_CHECKING:
    from flow_backend import ConversationState, FlowBackend


def classifier_node(backend: "FlowBackend", state: "ConversationState") -> dict:
    """
    Classifies the latest user message using fast TF-IDF similarity over
    scenario examples. On non-first turns with low/mid certainty, augments with
    an LLM-based intent classifier over recent context to detect one of:
    QUESTION | FOLLOWUP | SAVING_SCENARIO.
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

        # If first user message in the conversation, keep the fast path
        human_count = sum(1 for msg in messages if isinstance(msg, HumanMessage))
        if human_count <= 1:
            return {
                "query_type": chosen_label,
                "certainty": certainty,
                "awaiting_confirmation": False,
            }

        # If high certainty on subsequent turns, keep fast path
        if certainty >= 8:
            return {
                "query_type": chosen_label,
                "certainty": certainty,
                "awaiting_confirmation": False,
            }

        # Low/mid certainty on non-first turn → ask LLM to classify context intent
        window = messages[-3:] if len(messages) > 3 else messages
        window_text = backend._messages_to_string(window)

        prompt = (
            "Classify the user's latest turn in the following conversation as EXACTLY one of: "
            "QUESTION, FOLLOWUP, SAVING_SCENARIO.\n\n"
            "Conversation (most recent last):\n"
            f"{window_text}\n\n"
            "Definitions:\n"
            "- QUESTION: The user asks a new question or restates a new request.\n"
            "- FOLLOWUP: The user refines/changes/asks a follow-up about the current topic; keep previous query type.\n"
            "- SAVING_SCENARIO: The user is answering the assistant's prompt to save the question+answer template (e.g., 'yes, save it').\n\n"
            "Reply with a single word only: QUESTION or FOLLOWUP or SAVING_SCENARIO."
        )

        llm_model_input = state.get("llm_model_input") or getattr(
            backend, "_default_llm_model", "gpt-4o"
        )
        resp = backend._llm_agent.get_text_response_from_llm(
            llm_model_input=llm_model_input,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise intent classifier. Output one token only.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        label_text = (resp.get("text_response") or "").strip().upper()

        # Normalize label variants
        normalized = ""
        if re.search(r"\bSAVING[_\s-]?SCENARIO\b", label_text):
            normalized = "SAVING_SCENARIO"
        elif re.search(r"\bFOLLOW[\s-]?UP\b", label_text):
            normalized = "FOLLOWUP"
        elif re.search(r"\bQUESTION\b", label_text):
            normalized = "QUESTION"

        print(f"[CLASSIFIER] LLM intent label: '{label_text}' → '{normalized}'")

        if normalized == "FOLLOWUP":
            prior_query_type = state.get("query_type") or chosen_label or "OTHER"
            # Route directly to parameter extraction by setting high certainty
            return {
                "query_type": prior_query_type,
                "certainty": 9,
                "awaiting_confirmation": False,
            }

        if normalized == "SAVING_SCENARIO":
            # Route to adding_scenario_node via router
            return {
                "query_type": "SAVING_SCENARIO",
                "certainty": 9,
                "awaiting_confirmation": False,
            }

        # Default: treat as a new QUESTION → keep detector label, ensure mid certainty
        mid_certainty = certainty
        if mid_certainty < 2:
            mid_certainty = 2
        if mid_certainty > 7:
            mid_certainty = 7
        return {
            "query_type": chosen_label,
            "certainty": mid_certainty,
            "awaiting_confirmation": False,
        }

    except Exception as e:
        print(f"[CLASSIFIER] Detection/intent error: {e}")
        return {
            "query_type": "OTHER",
            "certainty": 2,
            "awaiting_confirmation": False,
        }
