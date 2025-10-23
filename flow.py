"""
CLI entry point for the Market Query Assistant.

This thin wrapper keeps the original terminal experience while delegating
all conversation logic to the FlowBackend module.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from langchain_core.messages import AIMessage

from flow_backend import FlowBackend


def interactive_chat(llm_model: str = "gpt-4o") -> None:
    """
    Interactive terminal chat interface.

    Args:
        llm_model: LLM model to use (e.g., "gpt-4o", "azure_openai:gpt-4o", "groq:llama-3.1-8b-instant")
    """
    backend = FlowBackend(llm_model_input=llm_model)
    state = backend.create_initial_state()

    print("=" * 60)
    print("MARKET QUERY ASSISTANT - Interactive Chat")
    print(f"Using LLM: {llm_model}")
    print("=" * 60)
    print("\nWelcome! Ask me about:")

    # Load example questions from scenarios.json if it exists
    scenarios_path = Path("scenarios.json")
    if scenarios_path.exists():
        try:
            with open(scenarios_path, "r") as f:
                scenarios = json.load(f)
            # Display first 5 question examples
            for scenario in scenarios[:5]:
                if "question_example" in scenario:
                    print(f"  • {scenario['question_example']}")
            print("  • seasonality of promo effectiveness")
            print("  • how many poc do i have in the data")
        except (json.JSONDecodeError, KeyError, IOError):
            # Fallback to hardcoded examples if there's an error reading the file
            print("  • Provide executive summary of performance for HQ")
            print("  • Show HQ performance for the Buscuits in July 2025")
            print("  • Give me the HQ biscuit results please")
            print("  • What countries are in Brazil")
            print("  • Results vs forecast for top countries")
            print("  • seasonality of promo effectiveness")
            print("  • how many poc do i have in the data")
    else:
        # Fallback to hardcoded examples if file doesn't exist
        print("  • Provide executive summary of performance for HQ")
        print("  • Show HQ performance for the Buscuits in July 2025")
        print("  • Give me the HQ biscuit results please")
        print("  • What countries are in Brazil")
        print("  • Results vs forecast for top countries")

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
    # Check for command-line argument for LLM model
    llm_model = "gpt-4o"  # default
    if len(sys.argv) > 1:
        llm_model = sys.argv[1]
        print(f"[INFO] Using LLM model from command line: {llm_model}\n")
    else:
        print(f"[INFO] Using default LLM model: {llm_model}")
        print("[INFO] To use a different model, run: python flow.py <model_name>")
        print("[INFO] Examples: uv run flow.py groq:llama-3.1-8b-instant")
        print("[INFO] Examples: uv run flow.py google_ai_studio:gemini-2.5-pro")
        print("[INFO] Examples: uv run flow.py qwen/qwen3-coder:exacto")
        print("[INFO] Examples: uv run flow.py x-ai/grok-code-fast-1")

    interactive_chat(llm_model)
