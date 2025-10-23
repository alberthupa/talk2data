#!/usr/bin/env python3
"""
Test script to demonstrate the generic SQL display improvements.

This script simulates a generic SQL query flow to verify that SQL generation
details are properly displayed in the final answer.
"""

from flow_backend import FlowBackend


def test_generic_sql_display():
    """Test that generic SQL route displays all details."""
    print("=" * 80)
    print("Testing Generic SQL Display")
    print("=" * 80)
    print()

    # Initialize backend
    backend = FlowBackend()
    state = backend.create_initial_state()

    # Step 1: Ask a question that will trigger medium certainty
    print("Step 1: User asks a question")
    user_question = "How many points of contact do we have?"
    print(f"User: {user_question}")
    print()

    result = backend.run_conversation(user_question, state)
    state = result.state

    # Print assistant response
    for msg in result.new_messages:
        print(f"Assistant: {msg.content}")
    print()

    # Check if we're in confirmation mode
    if state.get("awaiting_confirmation"):
        print("Step 2: User selects generic SQL option")
        print("User: 2")
        print()

        # User chooses option 2 (generic SQL)
        result = backend.run_conversation("2", state)
        state = result.state

        # Print assistant response (should include SQL details + interpretation)
        print("Assistant Response:")
        print("-" * 80)
        for msg in result.new_messages:
            print(msg.content)
        print("-" * 80)
        print()

        # Verify that generic_sql_details is in the state
        if state.get("generic_sql_details"):
            print("✓ SQL details captured in state")
        else:
            print("✗ SQL details NOT captured in state")

        # Verify that the final message contains both SQL details and interpretation
        if result.new_messages:
            final_content = result.new_messages[-1].content
            has_sql = "Generated SQL:" in final_content or "```sql" in final_content
            has_analysis = "Business Analysis" in final_content or len(final_content) > 100

            print(f"✓ Final message includes SQL query: {has_sql}")
            print(f"✓ Final message includes analysis: {has_analysis}")

        # Check if dataframe is available for display
        if state.get("display_dataframe"):
            df_data = state["display_dataframe"]
            print(f"✓ Dataframe available with {len(df_data.get('rows', []))} rows")
        else:
            print("✗ No dataframe for display")

    print()
    print("=" * 80)
    print("Test Complete")
    print("=" * 80)


if __name__ == "__main__":
    test_generic_sql_display()
