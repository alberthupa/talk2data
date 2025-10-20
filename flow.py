"""
CLI entry point for the Market Query Assistant.

This thin wrapper keeps the original terminal experience while delegating
all conversation logic to the FlowBackend module.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from flow_backend import FlowBackend


def interactive_chat() -> None:
    """
    Interactive terminal chat interface.
    """
    backend = FlowBackend()
    state = backend.create_initial_state()

    print("=" * 60)
    print("MARKET QUERY ASSISTANT - Interactive Chat")
    print("=" * 60)
    print("\nWelcome! Ask me about:")
    print("  • Provide executive summary of performance for HQ")
    print("  • Show HQ performance for the Buscuits in July 2025")
    print("  • Give me the HQ biscuit results please")
    print("  • What countries are in Brazil")
    print("  • Results vs forecast for top countries")
    print(
        "  • Top 10 best and worst regions in category in gross profit vs previous year"
    )
    print("\nType 'quit' or 'exit' to end the conversation\n")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n💬 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if user_input.lower() in ["quit", "exit", "bye", "goodbye"]:
            print("\n👋 Thanks for chatting! Goodbye!")
            break

        if not user_input:
            continue

        print()
        result = backend.run_conversation(user_input, state)
        state = result.state

        for message in result.new_messages:
            if isinstance(message, AIMessage):
                print(f"🤖 Assistant: {message.content}")


if __name__ == "__main__":
    interactive_chat()
