"""
Streamlit frontend for the Market Query Assistant.

Relies on FlowBackend to handle all business logic while Streamlit manages UX.
"""

from __future__ import annotations

import json
import os
import yaml
import pandas as pd
import streamlit as st

from flow_backend import FlowBackend, get_backend


def _load_scenarios() -> list:
    """Load scenarios from scenarios.json file."""
    try:
        with open("scenarios.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("scenarios.json file not found.")
        return []
    except json.JSONDecodeError:
        st.error("Error parsing scenarios.json file.")
        return []


def _load_llm_config() -> dict:
    """Load available LLM models from llm_config.yaml."""
    try:
        with open("llm_config.yaml", "r") as f:
            config = yaml.safe_load(f)
            return config.get("llm_location", {})
    except FileNotFoundError:
        st.warning("llm_config.yaml not found. Using default model.")
        return {}
    except yaml.YAMLError:
        st.error("Error parsing llm_config.yaml.")
        return {}


def _ensure_session_state(backend: FlowBackend) -> None:
    if "flow_state" not in st.session_state:
        st.session_state["flow_state"] = backend.create_initial_state()
    if "latest_new_messages" not in st.session_state:
        st.session_state["latest_new_messages"] = []
    if "selected_llm_model" not in st.session_state:
        st.session_state["selected_llm_model"] = "gpt-4o"


def _render_sidebar(state: dict, backend: FlowBackend, llm_config: dict) -> None:
    with st.sidebar:
        st.header("⚙️ Settings")

        # Check if running on Azure (or any production environment)
        IS_PRODUCTION = os.getenv("ENVIRONMENT") == "production"

        # LLM Model Selection (only in production)
        if not IS_PRODUCTION:
            st.subheader("LLM Model")

            # Build list of available models
            model_options = []
            for location, models in llm_config.items():
                for model in models:
                    model_options.append(f"{location}:{model}")

            # Add default option if no config
            if not model_options:
                model_options = ["gpt-4o"]

            # Current model from session state
            current_model = st.session_state.get("selected_llm_model", "gpt-4o")

            # Find index of current model
            try:
                current_index = model_options.index(current_model)
            except ValueError:
                current_index = 0

            selected_model = st.selectbox(
                "Select LLM Model",
                options=model_options,
                index=current_index,
                help="Choose which LLM model to use for the conversation",
            )

            # Update state if model changed
            if selected_model != st.session_state.get("selected_llm_model"):
                st.session_state["selected_llm_model"] = selected_model
                # Update the state's LLM model input to trigger reinitialization
                if "flow_state" in st.session_state:
                    st.session_state["flow_state"]["llm_model_input"] = selected_model
                    # Clear client to force reinitialization
                    st.session_state["flow_state"]["llm_client"] = None
                st.info(f"Switched to: {selected_model}")

        st.markdown("---")

        st.header("Conversation Status")
        st.write(f"**Query type**: {state.get('query_type') or '—'}")
        certainty = state.get("certainty")
        st.write(f"**Certainty**: {certainty if certainty is not None else '—'} / 10")

        # Show current LLM in use
        current_llm = state.get("llm_model_input") or "Not initialized"
        st.write(f"**Active LLM**: {current_llm}")

        if state.get("awaiting_confirmation"):
            if state.get("awaiting_generic_choice"):
                st.info(
                    "Reply with '1' to confirm, '2' for generic SQL, or 'no' to reclassify."
                )
            else:
                st.info("Awaiting your confirmation (yes/no).")

        if state.get("awaiting_clarification"):
            missing_params = state.get("missing_params") or []
            if missing_params:
                st.warning("Missing parameters:")
                for param in missing_params:
                    st.write(f"- {param['name']} (e.g., {param['example']})")

        # Show generic SQL error if any
        generic_sql_error = state.get("generic_sql_error")
        if generic_sql_error:
            st.error(f"Generic SQL Error: {generic_sql_error}")

        if st.button("Reset conversation"):
            st.session_state["flow_state"] = backend.create_initial_state()
            st.session_state["latest_new_messages"] = []
            st.rerun()

        st.markdown("---")
        st.subheader("Example Questions")

        # Load scenarios and display top 10 examples
        scenarios = _load_scenarios()
        if scenarios:
            # Get up to 10 question examples
            examples = [
                scenario.get("question_example", "No example available")
                for scenario in scenarios[:10]
            ]

            for example in examples:
                st.markdown(f"- {example}")
        else:
            st.warning("No example questions available.")


def main() -> None:
    st.set_page_config(page_title="Lingaro SQL Copilot", layout="wide")

    # Password authentication

    # Check if running on Azure (or any production environment)
    IS_PRODUCTION = os.getenv("ENVIRONMENT") == "production"

    # Password authentication
    if IS_PRODUCTION:
        if "authenticated" not in st.session_state:
            st.session_state["authenticated"] = False

        if not st.session_state["authenticated"]:
            st.title("🔒 Authentication Required")
            password_input = st.text_input("Enter password:", type="password")

            if st.button("Login"):
                if password_input == "LingaroDelivers":
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Incorrect password. Please try again.")

            st.stop()

    st.title("📊 Lingaro SQL Copilot")
    st.caption(
        "Ask data questions about performance metrics. The assistant classifies your query, "
        "extracts required parameters, and returns insightful answers."
    )

    # Load LLM configuration
    llm_config = _load_llm_config()

    # Initialize or get backend with selected model
    selected_model = st.session_state.get("selected_llm_model", "gpt-4o")
    backend = FlowBackend(llm_model_input=selected_model)

    _ensure_session_state(backend)

    state = st.session_state["flow_state"]
    _render_sidebar(state, backend, llm_config)

    # Create four tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Intro", "Chat", "Scenarios", "Source Data"])

    # Tab 1: Intro
    with tab1:
        st.header("Intro")

        st.subheader("What this is (for business users)")
        st.markdown(
            """
            - Ask plain-English questions about market performance (revenue, volume, profit, trends).
            - If your question matches a known business scenario, the app runs a vetted, deterministic SQL query to answer it.
            - All available scenarios are listed in the "Scenarios" tab; the sidebar shows which scenario was matched and the certainty (1–10).
            - If certainty isn’t high, the app either asks you to confirm or can generate a one-off custom SQL query for you (Generic SQL).
            - After a Generic SQL answer, you can choose to save it as a scenario for future refinement and reuse.
            """
        )

        st.subheader("How to use")
        st.markdown(
            """
            - Go to the "Chat" tab and type your question (examples are in the sidebar).
            - When the system is confident, it answers using a deterministic scenario SQL.
            - When confidence is medium, you’ll see options: confirm the scenario, use Generic SQL, or reclassify.
            - Results are explained in plain language and, when relevant, shown as tables.
            """
        )

        st.markdown("---")
        st.subheader("Full README")
        try:
            with open("README.md", "r") as f:
                readme_text = f.read()
            st.markdown(readme_text)
        except FileNotFoundError:
            st.info("README.md not found.")

    # Tab 2: Chat interface
    with tab2:
        history = FlowBackend.convert_messages_for_display(state["messages"])
        followup_prompt = None
        for i, (role, content) in enumerate(history):
            with st.chat_message(role):
                st.markdown(content)

                # Display dataframe table after the last AI message if available
                if (
                    role == "assistant"
                    and i == len(history) - 1
                    and state.get("display_dataframe")
                ):
                    df_data = state["display_dataframe"]
                    if df_data and df_data.get("rows"):
                        columns = df_data.get("columns", [])
                        rows = df_data.get("rows", [])

                        # Create pandas DataFrame
                        df = pd.DataFrame(rows, columns=columns)

                        # Display as a nice table
                        st.dataframe(df, use_container_width=True)

        confirmation_action = None

        clarification_input = None
        if state.get("awaiting_clarification"):
            # If a scenario-specific hint exists, print it and use as placeholder
            placeholder_text = "e.g., July 2025 for Biscuits in Brazil…"
            try:
                scenarios = _load_scenarios()
                qt = state.get("query_type")
                hint = None
                if scenarios and qt:
                    for sc in scenarios:
                        if (
                            sc.get("question_example") == qt
                            or sc.get("question_type") == qt
                            or sc.get("question_id") == qt
                        ):
                            hint = sc.get("hint")
                            break
                if hint:
                    print(hint)
                    placeholder_text = hint
            except Exception:
                # Fallback silently if scenarios cannot be loaded/parsed
                pass

            clarification_input = st.text_input(
                "Provide missing details",
                placeholder=placeholder_text,
                key="clarification_input",
            )

        user_prompt = None
        if followup_prompt:
            user_prompt = followup_prompt
        elif confirmation_action:
            user_prompt = confirmation_action
        elif clarification_input:
            user_prompt = clarification_input
        else:
            user_prompt = st.chat_input("Type your question about market performance…")

        if user_prompt:
            with st.spinner("Thinking..."):
                result = backend.run_conversation(user_prompt, state)
                st.session_state["flow_state"] = result.state
                st.session_state["latest_new_messages"] = (
                    FlowBackend.convert_messages_for_display(result.new_messages)
                )
                state = st.session_state["flow_state"]
                if not state.get("awaiting_clarification"):
                    st.session_state.pop("clarification_input", None)
            st.rerun()

    # Tab 3: Scenarios
    with tab3:
        st.header("Available Scenarios")
        st.markdown(
            "Below are all the available query scenarios with their descriptions and SQL templates:"
        )

        scenarios = _load_scenarios()

        if scenarios:
            for scenario in scenarios:
                scenario_id = scenario.get("question_id", "N/A")
                example = scenario.get("question_example", "No example available")
                description = scenario.get(
                    "question_description", "No description available"
                )
                sql_template = scenario.get(
                    "question_sql_template", "No SQL template available"
                )
                question_inputs = scenario.get("question_inputs", [])

                with st.expander(f"**Scenario {scenario_id}: {example}**"):
                    st.markdown(f"**Description:** {description}")

                    # Display question inputs if available
                    if question_inputs:
                        st.markdown("**Required Inputs:**")
                        for input_param in question_inputs:
                            param_name = input_param.get("name", "N/A")
                            param_type = input_param.get("type", "N/A")
                            param_example = input_param.get("example", "N/A")
                            unique_values = input_param.get("unique_values", [])

                            st.markdown(f"- **{param_name}** ({param_type})")
                            st.markdown(f"  - Example: `{param_example}`")

                            if unique_values:
                                st.markdown(
                                    f"  - Unique values: {', '.join([f'`{v}`' for v in unique_values])}"
                                )

                    st.markdown("**SQL Template:**")
                    st.code(sql_template, language="sql")
        else:
            st.warning("No scenarios available to display.")

    # Tab 4: Source Data
    with tab4:
        st.header("Source Data")

        # Display data description if file exists
        try:
            with open("data_description.md", "r") as f:
                data_description = f.read()
            st.markdown(data_description)
        except FileNotFoundError:
            st.info("There will be inserted data description.")

        st.markdown("---")

        st.markdown("""
        ### Download Source Data

        You can download the source data from the following Google Sheets link:
        """)

        st.markdown(
            "[Click here to download source data](https://docs.google.com/spreadsheets/d/1nutv2ABuCygA2JvCHKrsAADTPitTLjIKpmVZFM56TAo/edit?usp=sharing)"
        )

        st.info(
            "The source data contains all the performance metrics used by the SQL Copilot."
        )


if __name__ == "__main__":
    main()
