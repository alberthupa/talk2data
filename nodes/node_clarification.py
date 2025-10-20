from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from flow_backend import ConversationState, FlowBackend


def clarification_node(backend: "FlowBackend", state: "ConversationState") -> dict:
    """Asks user to provide missing parameters needed for the query."""
    missing_params = state.get("missing_params", [])
    query_type = state.get("query_type", "unknown")

    if not missing_params:
        return {
            "messages": [AIMessage(content="Please provide more details.")],
            "awaiting_clarification": True,
        }

    param_questions = []
    for param in missing_params:
        param_name = param["name"]
        param_example = param["example"]
        param_questions.append(
            f"- **{param_name.replace('_', ' ').title()}** (e.g., {param_example})"
        )

    param_list = "\n".join(param_questions)

    clarification_msg = (
        f"I need some additional information to answer your question about "
        f"**{query_type.replace('_', ' ')}**:\n\n{param_list}\n\nCould you please provide these details?"
    )

    print(
        f"[CLARIFICATION] Asking for missing parameters: "
        f"{[param['name'] for param in missing_params]}"
    )

    return {
        "messages": [AIMessage(content=clarification_msg)],
        "awaiting_clarification": True,
    }
