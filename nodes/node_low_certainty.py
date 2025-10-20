from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from flow_backend import ConversationState, FlowBackend


def low_certainty_node(backend: "FlowBackend", state: "ConversationState") -> dict:
    """Handles low certainty by asking user to start over with more details."""
    message = (
        "I'm not quite sure what you're asking about.\n"
        "Could you please rephrase your question with more details? \n"
        "For example, you can ask about: \n"
        "- Executive summary of performance for HQ \n"
        "- Executive summary of YTD performance by region for HQ \n"
        "- Show top and bottom Gross Profit growth country and category combination \n"
    )

    print("[LOW CERTAINTY] Asking user to clarify")

    return {
        "messages": [AIMessage(content=message)],
        "awaiting_confirmation": False,
    }
