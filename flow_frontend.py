"""
Streamlit frontend for the Market Query Assistant.

Relies on FlowBackend to handle all business logic while Streamlit manages UX.
"""

from __future__ import annotations

import json
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

        # LLM Model Selection
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
            help="Choose which LLM model to use for the conversation"
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
            st.info("Awaiting your confirmation (yes/no).")

        if state.get("awaiting_clarification"):
            missing_params = state.get("missing_params") or []
            if missing_params:
                st.warning("Missing parameters:")
                for param in missing_params:
                    st.write(f"- {param['name']} (e.g., {param['example']})")

        if st.button("Reset conversation"):
            st.session_state["flow_state"] = backend.create_initial_state()
            st.session_state["latest_new_messages"] = []
            st.rerun()

        st.markdown("---")
        st.markdown("Provide executive summary of performance for HQ")
        st.markdown("Provide executive summary of YTD performance by region for HQ")
        st.markdown(
            "Show top and bottom Gross Profit growth country and category combinations"
        )
        st.markdown("Show HQ performance for the Biscuits in July 2025")
        st.markdown("Show HQ Operational Income performance for the Biscuits in 2025")
        st.markdown(
            "What is the Chocolate Other sales volume performance for EU in June 2025 vs forecast?"
        )
        st.markdown(
            "Show me top 10 best and worst performing countries on Biscuit gross profit vs previous year, in terms of growth percentage."
        )
        st.markdown(
            "Show me top 20 countries in terms of L3M category volume growth vs previous year."
        )
        st.markdown("Show me historical category trend in Brazil Biscuits")
        st.markdown(
            "Show me actual Gross profit Top/Bottom 20 Country/Category Combinations by PY growth"
        )
        st.markdown("How was MDLZ performance for the Q2 2025?")
        st.markdown(
            "Provide Gross Profit till the end of June vs PY ($MM) Top/Bottom 10 Country/Brand Combinations"
        )
        st.markdown("What has changed vs Prior Forecast?")


def main() -> None:
    st.set_page_config(page_title="Lingaro SQL Copilot", layout="wide")

    # Password authentication
    """
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
    """
    st.title("📊 Lingaro - Mondelez GenSights Technical Proof of Concept")
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

        st.markdown("""
        This AI-powered Market Query Assistant transforms natural language questions into actionable business insights
        by intelligently matching queries to predefined analytical scenarios and automatically extracting relevant
        parameters. The system leverages conversational AI to understand business questions about performance metrics,
        execute SQL queries against Mondelez data, and deliver clear, data-driven answers with visual context.
        """)

        st.markdown("---")

        st.subheader("Scenario Matching")

        st.markdown("""
        This conversational interface allows you to **ask questions that route to answer scenarios**. Any other questions
        or vague questions will lead trigger clarification requests. This solution does not allow to go any other
        scenario other than predetermined and described in tab "Scenarios".
        
        You can **see which scenario was matched in the sidebar in the top left corner** of the application.
        
        Scenario *total performance of category in year-month* triggered by question *Provide executive summary of performance for HQ* triggers example of **buttons with follow-up questions**.

        **Examples of questions triggering scenarios are presented on the left tab of the application UI**. Users may ask
        questions with various examples and in various form. Conversational AI matches these questions with some level
        of certainty. If certainty in matching scenarios is low, the system asks a user for clarification.
        """)

        st.subheader("Parameter Extraction")

        st.markdown("""
        Answer **cenarios are always predetermined** queries that need to be filled with questions parameters, like:
        country, region, product category or a month. You can see these parameters in tab "Scenarios" in sql queries
        like in this example:
        """)

        st.code(
            """WHERE region = '{{region}}'
    AND sub_category_text = '{{category}}'
    AND year = {{year}}""",
            language="sql",
        )

        st.markdown("""
        In this example the role of conversational interface is to find out which region, category and year a user
        is interested in.
        """)

        st.subheader("Data Information")

        st.markdown("""
        Data presented in the interface is artificial, generated by authors of this POC. As such actual numbers may
        not be reasonable. Actual dataset used in this UI can be seen in tab "Source Data". Data covers periods
        **January - September 2025**.
        """)

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

                if (
                    role == "assistant"
                    and i == len(history) - 1
                    and state.get("query_type")
                    == "executive_summary_of_performance_for_hq"
                    and not state.get("awaiting_confirmation")
                    and not state.get("awaiting_clarification")
                ):
                    buttons = st.columns(2)
                    with buttons[0]:
                        if st.button(
                            "Regional breakdown",
                            key="regional_breakdown_button",
                            use_container_width=True,
                        ):
                            followup_prompt = "Provide executive summary of performance by region for 2025."
                    with buttons[1]:
                        if st.button(
                            "Region and country breakdown",
                            key="region_country_breakdown_button",
                            use_container_width=True,
                        ):
                            followup_prompt = "Show top 10 and bottom 10 gross profit growth country and category combinations for the Gross Profit (MM) KPI."

        confirmation_action = None
        if state.get("awaiting_confirmation"):
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes ✅", use_container_width=True):
                    confirmation_action = "yes"
            with col_no:
                if st.button("No ❌", use_container_width=True):
                    confirmation_action = "no"

        clarification_input = None
        if state.get("awaiting_clarification"):
            clarification_input = st.text_input(
                "Provide missing details",
                placeholder="e.g., July 2025 for Biscuits in Brazil…",
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
                description = scenario.get(
                    "question_description", "No description available"
                )
                sql_template = scenario.get(
                    "question_sql_template", "No SQL template available"
                )
                question_inputs = scenario.get("question_inputs", [])

                with st.expander(f"**Scenario {scenario_id}: {description}**"):
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
        st.markdown("""
        ### Download Source Data

        You can download the source data from the following Google Sheets link:
        """)

        st.markdown(
            "[Click here to download source data](https://docs.google.com/spreadsheets/d/1YosemXgGERFDHTkPSHj8YdM2Z2XUGw8J/edit?usp=drive_link&ouid=118397130291546262478&rtpof=true&sd=true)"
        )

        st.info(
            "The source data contains all the performance metrics used by the SQL Copilot."
        )


if __name__ == "__main__":
    main()
