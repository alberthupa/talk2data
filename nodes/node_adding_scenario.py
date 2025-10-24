from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from flow_backend import ConversationState, FlowBackend


def adding_scenario_node(backend: "FlowBackend", state: "ConversationState") -> dict:
    """Acknowledge saving scenario request.

    For now, just returns a confirmation message to the user. In the future,
    this node can persist the scenario template to the scenario library.
    """
    print("[ADDING SCENARIO] Acknowledging scenario save request")
    return {
        "messages": [AIMessage(content="ok, i added it to scenario library")],
        "awaiting_confirmation": False,
    }

