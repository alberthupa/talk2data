from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from flow_backend import ConversationState, FlowBackend


def confirmation_node(backend: "FlowBackend", state: "ConversationState") -> dict:
    """Asks the user to confirm the classified query when certainty is medium."""
    query_type = state.get("query_type", "unknown")

    confirmation_msg = (
        f"I think you're asking about: **{query_type.replace('_', ' ')}**. "
        f"Is that correct? (yes/no)"
    )

    print(f"[CONFIRMATION] Asking for confirmation on: {query_type}")

    return {
        "messages": [AIMessage(content=confirmation_msg)],
        "awaiting_confirmation": True,
    }
