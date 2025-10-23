"""
Test script to verify the LLM migration is working correctly.

This tests:
1. FlowBackend initialization with custom LLM models
2. LLM client initialization
3. State management
4. Node operations (at least classifier and parameter extraction)
"""

from flow_backend import FlowBackend


def test_backend_initialization():
    """Test that FlowBackend initializes correctly with different models."""
    print("=" * 60)
    print("TEST 1: Backend Initialization")
    print("=" * 60)

    models_to_test = [
        "gpt-4o",
        "azure_openai:gpt-4o",
        # Add more if you have API keys configured
    ]

    for model in models_to_test:
        try:
            backend = FlowBackend(llm_model_input=model)
            print(f"✓ Successfully created backend with model: {model}")
        except Exception as e:
            print(f"✗ Failed to create backend with model {model}: {e}")

    print()


def test_state_creation():
    """Test that state is created with LLM fields."""
    print("=" * 60)
    print("TEST 2: State Creation")
    print("=" * 60)

    backend = FlowBackend(llm_model_input="gpt-4o")
    state = backend.create_initial_state()

    required_llm_fields = [
        "llm_model_input",
        "llm_client",
        "llm_model_location",
        "llm_model_name",
    ]

    for field in required_llm_fields:
        if field in state:
            print(f"✓ State has field: {field}")
        else:
            print(f"✗ State missing field: {field}")

    print()


def test_client_initialization():
    """Test that LLM client initializes in state."""
    print("=" * 60)
    print("TEST 3: Client Initialization")
    print("=" * 60)

    backend = FlowBackend(llm_model_input="azure_openai:gpt-4o")
    state = backend.create_initial_state()

    # Initialize client
    client, location, model_name = backend._initialize_llm_client(state)

    if client is not None:
        print(f"✓ Client initialized successfully")
        print(f"  Location: {location}")
        print(f"  Model: {model_name}")
        print(f"  State updated: {state['llm_client'] is not None}")
    else:
        print("✗ Client initialization failed")
        print("  This might be due to missing API keys in .env")

    print()


def test_conversation_flow_simulation():
    """Test a simulated conversation to see if client initialization works."""
    print("=" * 60)
    print("TEST 4: Conversation Flow Simulation")
    print("=" * 60)

    backend = FlowBackend(llm_model_input="azure_openai:gpt-4o")
    state = backend.create_initial_state()

    # Simulate a user query
    test_query = "Show HQ performance for Biscuits in July 2025"

    print(f"Test query: '{test_query}'")
    print("Attempting to process conversation turn...")

    try:
        result = backend.run_conversation(test_query, state)

        print(f"✓ Conversation processed successfully")
        print(f"  Messages in result: {len(result.new_messages)}")
        print(f"  Query type classified as: {result.state.get('query_type', 'N/A')}")
        print(f"  Certainty: {result.state.get('certainty', 'N/A')}")
        print(f"  LLM model used: {result.state.get('llm_model_input', 'N/A')}")

        if result.new_messages:
            print(f"\n  First response preview:")
            first_msg = result.new_messages[0]
            content = first_msg.content if hasattr(first_msg, "content") else str(first_msg)
            print(f"  {content[:200]}...")

    except Exception as e:
        print(f"✗ Conversation processing failed: {e}")
        import traceback
        traceback.print_exc()

    print()


def test_model_switching():
    """Test switching models mid-conversation."""
    print("=" * 60)
    print("TEST 5: Model Switching")
    print("=" * 60)

    backend = FlowBackend(llm_model_input="azure_openai:gpt-4o")
    state = backend.create_initial_state()

    # Initialize with first model
    client1, loc1, model1 = backend._initialize_llm_client(state)
    print(f"Initial model: {loc1}:{model1}")

    # Change model in state
    state["llm_model_input"] = "azure_openai:gpt-4o-mini"
    state["llm_client"] = None  # Clear to force reinitialization

    # Reinitialize with second model
    client2, loc2, model2 = backend._initialize_llm_client(state)

    if model1 != model2:
        print(f"✓ Successfully switched from {model1} to {model2}")
    else:
        print(f"✗ Model switch did not work as expected")

    print()


def test_generic_sql_state():
    """Test that generic SQL state fields are properly initialized."""
    print("=" * 60)
    print("TEST 6: Generic SQL State Fields")
    print("=" * 60)

    backend = FlowBackend(llm_model_input="gpt-4o")
    state = backend.create_initial_state()

    generic_sql_fields = [
        "confirmation_mode",
        "awaiting_generic_choice",
        "generic_sql_attempted",
        "generic_sql_error",
        "generic_sql_question",
    ]

    print("Checking generic SQL state fields...")
    all_present = True
    for field in generic_sql_fields:
        if field in state:
            print(f"  ✓ {field}: {state[field]}")
        else:
            print(f"  ✗ Missing field: {field}")
            all_present = False

    if all_present:
        print("✓ All generic SQL state fields are present")
    else:
        print("✗ Some generic SQL state fields are missing")

    print()


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 12 + "LLM MIGRATION TEST SUITE" + " " * 22 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")

    test_backend_initialization()
    test_state_creation()
    test_client_initialization()
    test_conversation_flow_simulation()
    test_model_switching()
    test_generic_sql_state()

    print("=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)
    print("\nNote: Some tests may fail if API keys are not configured.")
    print("Check your .env file for required credentials.")


if __name__ == "__main__":
    main()
