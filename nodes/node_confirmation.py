from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from flow_backend import ConversationState, FlowBackend


def confirmation_node(backend: "FlowBackend", state: "ConversationState") -> dict:
    """Asks the user to confirm the classified query when certainty is medium."""
    query_type = state.get("query_type", "unknown")

    confirmation_msg = (
        f"I think you're asking about: **{query_type.replace('_', ' ')}**.\n\n"
        f"Please choose an option:\n"
        f"1. **Confirm** - Yes, that's correct\n"
        f"2. **Generic SQL** - Let me craft a custom SQL query for you\n"
        f"3. **No** - That's wrong, try again\n\n"
        f"Reply with '1', '2', or 'no'."
    )

    print(f"[CONFIRMATION] Asking for confirmation on: {query_type}")

    return {
        "messages": [AIMessage(content=confirmation_msg)],
        "awaiting_confirmation": True,
        "confirmation_mode": "scenario",
        "awaiting_generic_choice": True,
    }
