from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage
from pydantic import Field, create_model
from llms.structured_output import get_structured_response

if TYPE_CHECKING:
    from flow_backend import ConversationState, FlowBackend


def classifier_node(backend: "FlowBackend", state: "ConversationState") -> dict:
    """
    Classifies the user query with certainty level using LLM-based classification.
    """
    messages = state["messages"]
    recent_messages = messages[-6:] if len(messages) > 6 else messages
    messages_string = backend._messages_to_string(recent_messages)
    last_user_message = next(
        (msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)),
        "",
    )

    prompt = f"""You are classifying the latest user message in a conversation.

Conversation context (for reference only):
{messages_string}

Latest user message:
"{last_user_message}"

Decide whether the latest user message is about any of the following questions:
{backend._question_descriptions}

Respond ONLY in the following JSON format:
{{
    "question_type": "...",
    "certainty": 1-10
}}

Set question_type to the most relevant scenario, or "OTHER" if none of them are relevant to the latest user message.

If a user asks you about any company that is not Mondelez, set question_type to "OTHER".

"""

    print(prompt)

    query_types: tuple[str, ...] = tuple(
        scenario["question_type"] for scenario in backend._scenarios
    ) + ("OTHER",)

    QuestionClassification = create_model(
        "QuestionClassification",
        question_type=(str, Field(...)),
        certainty=(int, Field(..., ge=1, le=10)),
    )

    allowed_query_types = set(query_types)

    # Get LLM client from state
    client = state.get("llm_client")
    model_location = state.get("llm_model_location")
    model_name = state.get("llm_model_name")

    if not client or not model_location or not model_name:
        print("[CLASSIFIER] LLM client not initialized in state")
        return {
            "query_type": "unknown",
            "certainty": 1,
            "awaiting_confirmation": False,
        }

    # Use structured output module
    init_output = get_structured_response(
        client=client,
        model_location=model_location,
        model_name=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_model=QuestionClassification,
        max_retries=5,
    )

    if init_output:
        query_type = init_output.question_type
        if query_type not in allowed_query_types:
            query_type = "OTHER"
        certainty = init_output.certainty

        print(f"[CLASSIFIER] Query: '{messages_string[:50]}...'")
        print(f"[CLASSIFIER] Type: {query_type}, Certainty: {certainty}/10")

        return {
            "query_type": query_type,
            "certainty": certainty,
            "awaiting_confirmation": False,
        }

    # Fallback if structured output failed
    return {
        "query_type": "unknown",
        "certainty": 1,
        "awaiting_confirmation": False,
    }
