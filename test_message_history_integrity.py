#!/usr/bin/env python3
"""
Quick integrity check to ensure message history is preserved across
manual branches (e.g., generic SQL selection) and not overwritten.

This script does not assert; it prints diagnostics so it can run
without full test harness or API keys.
"""

from langchain_core.messages import HumanMessage, AIMessage

from flow_backend import FlowBackend


def test_generic_sql_preserves_history():
    print("=" * 80)
    print("TEST: Message History Integrity (Generic SQL path)")
    print("=" * 80)

    backend = FlowBackend()
    state = backend.create_initial_state()

    # Emulate a prior turn leading to confirmation with generic choice available
    user_q = "What are the top 5 categories by revenue?"
    state["messages"].append(HumanMessage(content=user_q))
    state["messages"].append(
        AIMessage(
            content=(
                "I think I matched a scenario. Reply '1' to confirm, "
                "'2' for a generic SQL query, or 'no' to reclassify."
            )
        )
    )
    state["awaiting_confirmation"] = True
    state["awaiting_generic_choice"] = True

    before_len = len(state["messages"])
    before_str = backend._messages_to_string(state["messages"])

    # User selects generic SQL (option 2)
    result = backend.run_conversation("2", state)
    state = result.state

    after_len = len(state["messages"])
    after_str = backend._messages_to_string(state["messages"])

    print(f"Messages before: {before_len}")
    print(f"Messages after:  {after_len}")
    grew = after_len >= before_len
    preserved = user_q in after_str

    print(f"✓ History grew or stayed: {grew}")
    print(f"✓ First user message preserved: {preserved}")

    if result.new_messages:
        print("\nNew assistant messages:")
        for m in result.new_messages:
            print(f"- {getattr(m, 'content', str(m))[:200]}")

    print("=" * 80)
    print("DONE")


if __name__ == "__main__":
    test_generic_sql_preserves_history()

