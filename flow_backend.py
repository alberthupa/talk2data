"""
Backend logic for the Market Query Assistant conversational flow.

This module encapsulates the LangGraph pipeline, Azure OpenAI interactions,
and conversation state handling so that multiple frontends (CLI, Streamlit, etc.)
can reuse the same business logic.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

import json
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from openai import AzureOpenAI
from pydantic import BaseModel, Field, create_model


class ConversationState(TypedDict):
    """State for the conversational flow."""

    messages: Annotated[list, add_messages]
    query_type: str | None
    certainty: int | None
    awaiting_confirmation: bool
    extracted_params: dict | None
    missing_params: list | None
    awaiting_clarification: bool
    sql_query: str | None
    sql_results: dict | None
    display_dataframe: dict | None


@dataclass(slots=True)
class ConversationTurnResult:
    """Return structure for a processed user turn."""

    state: ConversationState
    new_messages: list[AIMessage | HumanMessage | SystemMessage]


class FlowBackend:
    """Encapsulates the conversation flow logic."""

    def __init__(
        self,
        *,
        env_path: str | os.PathLike[str] | None = ".env",
        scenarios_path: str | os.PathLike[str] = "scenarios.json",
        llm_client: AzureOpenAI | None = None,
        db_path: str | os.PathLike[str] = "sql_data.db",
    ) -> None:
        if env_path:
            load_dotenv(env_path, override=True)

        self._openai_api_version = os.environ.get("OPENAI_API_VERSION")
        self._azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        self._azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self._deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

        self._llm_client = llm_client or self._build_llm_client()
        self._scenarios = self._load_scenarios(Path(scenarios_path))
        self._question_descriptions = self._build_question_descriptions(self._scenarios)

        # Initialize SQLite database path (connections are opened per query)
        self._db_path = Path(db_path)
        self._init_database()

        self._graph_app = self._build_graph()

    #
    # Public API
    #
    def create_initial_state(self) -> ConversationState:
        """Return a fresh conversation state dictionary."""
        return {
            "messages": [],
            "query_type": None,
            "certainty": None,
            "awaiting_confirmation": False,
            "extracted_params": None,
            "missing_params": None,
            "awaiting_clarification": False,
            "sql_query": None,
            "sql_results": None,
            "display_dataframe": None,
        }

    def run_conversation(
        self,
        user_input: str,
        conversation_state: ConversationState | None = None,
    ) -> ConversationTurnResult:
        """
        Process a single user turn and return updated state plus new messages.
        """
        state = conversation_state or self.create_initial_state()
        previous_len = len(state["messages"])

        # Clear stale tabular data before processing the new turn
        state["display_dataframe"] = None
        state["sql_results"] = None

        if state.get("awaiting_clarification"):
            state["messages"].append(HumanMessage(content=user_input))
            print(
                "[CONVERSATION] User provided clarification, re-extracting parameters..."
            )

            param_result = self.parameter_extraction_node(state)
            state.update(param_result)

            if param_result.get("missing_params"):
                clarif_result = self.clarification_node(state)
                state["messages"].extend(clarif_result["messages"])
                state["awaiting_clarification"] = True
                new_messages = state["messages"][previous_len:]
                return ConversationTurnResult(state=state, new_messages=new_messages)

            sql_result = self.sql_node(state)
            state.update(sql_result)

            answer_result = self.answering_node(state)
            state.update(answer_result)
            state["awaiting_clarification"] = False
            new_messages = state["messages"][previous_len:]
            return ConversationTurnResult(state=state, new_messages=new_messages)

        if state.get("awaiting_confirmation"):
            state["messages"].append(HumanMessage(content=user_input))

            user_input_lower = user_input.lower()
            if any(
                word in user_input_lower
                for word in ["yes", "yeah", "correct", "right", "yep"]
            ):
                param_result = self.parameter_extraction_node(state)
                state.update(param_result)

                if param_result.get("missing_params"):
                    clarif_result = self.clarification_node(state)
                    state["messages"].extend(clarif_result["messages"])
                    state["awaiting_confirmation"] = False
                    state["awaiting_clarification"] = True
                    new_messages = state["messages"][previous_len:]
                    return ConversationTurnResult(
                        state=state, new_messages=new_messages
                    )

                sql_result = self.sql_node(state)
                state.update(sql_result)

                answer_result = self.answering_node(state)
                state.update(answer_result)
                state["awaiting_confirmation"] = False
                new_messages = state["messages"][previous_len:]
                return ConversationTurnResult(state=state, new_messages=new_messages)

            if any(
                word in user_input_lower
                for word in ["no", "nope", "wrong", "different"]
            ):
                print("[CONVERSATION] User denied, reclassifying...")
                state["awaiting_confirmation"] = False
                result_state = self._graph_app.invoke(state)
                state.update(result_state)
                new_messages = state["messages"][previous_len:]
                return ConversationTurnResult(state=state, new_messages=new_messages)

            state["messages"].append(
                AIMessage(content="Please answer with 'yes' or 'no'.")
            )
            new_messages = state["messages"][previous_len:]
            return ConversationTurnResult(state=state, new_messages=new_messages)

        state["messages"].append(HumanMessage(content=user_input))
        result_state = self._graph_app.invoke(state)
        state.update(result_state)

        new_messages = state["messages"][previous_len:]
        return ConversationTurnResult(state=state, new_messages=new_messages)

    #
    # Helpers exposed for frontends
    #
    @staticmethod
    def convert_messages_for_display(
        messages: list[AIMessage | HumanMessage | SystemMessage],
    ) -> list[tuple[str, str]]:
        """Convert LangChain messages to (role, content) tuples suitable for UIs."""
        display_items: list[tuple[str, str]] = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                role = "assistant"
            content = (
                message.content
                if isinstance(message.content, str)
                else str(message.content)
            )
            display_items.append((role, content))
        return display_items

    #
    # Internal helpers & LangGraph nodes
    #
    def _init_database(self) -> None:
        """Validate SQLite database file exists."""
        if not self._db_path.exists():
            raise FileNotFoundError(f"Database file not found at {self._db_path}")
        print(f"[DATABASE] SQLite database available at {self._db_path}")

    def _build_llm_client(self) -> AzureOpenAI:
        if not all(
            [self._azure_endpoint, self._azure_api_key, self._openai_api_version]
        ):
            raise RuntimeError(
                "Azure OpenAI environment variables are not fully configured. "
                "Expected AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION."
            )

        return AzureOpenAI(
            azure_endpoint=self._azure_endpoint,
            api_key=self._azure_api_key,
            api_version=self._openai_api_version,
        )

    def _load_scenarios(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found at {path}")

        with path.open("r") as json_file:
            return json.load(json_file)

    @staticmethod
    def _build_question_descriptions(scenarios: list[dict[str, Any]]) -> str:
        keys_to_filter = ["question_type", "question_description", "question_example"]
        question_descriptions = ""
        for scenario in scenarios:
            for key, value in scenario.items():
                if key in keys_to_filter:
                    question_descriptions += f'"{key}": "{value}"\n'
            question_descriptions += "###\n"
        return question_descriptions

    @staticmethod
    def _messages_to_string(
        messages: list[AIMessage | HumanMessage | SystemMessage],
    ) -> str:
        return "\n".join(
            [
                (
                    "User input"
                    if isinstance(message, (HumanMessage, SystemMessage))
                    else "Assistant message"
                )
                + f": {message.content}"
                for message in messages
            ]
        )

    def classifier_node(self, state: ConversationState) -> dict:
        """
        Classifies the user query with certainty level using LLM-based classification.
        """
        messages = state["messages"]
        recent_messages = messages[-6:] if len(messages) > 6 else messages
        messages_string = self._messages_to_string(recent_messages)
        last_user_message = next(
            (
                msg.content
                for msg in reversed(messages)
                if isinstance(msg, HumanMessage)
            ),
            "",
        )

        prompt = f"""You are classifying the latest user message in a conversation.

Conversation context (for reference only):
{messages_string}

Latest user message:
"{last_user_message}"

Decide whether the latest user message is about any of the following questions:
{self._question_descriptions}

Respond ONLY in the following JSON format:
{{
    "question_type": "...",
    "certainty": 1-10
}}

Set question_type to the most relevant scenario, or "OTHER" if none of them are relevant to the latest user message.

If a user asks you about any company that is not Mondelez, set question_type to "OTHER".

"""

        query_types: tuple[str, ...] = tuple(
            scenario["question_type"] for scenario in self._scenarios
        ) + ("OTHER",)

        QuestionClassification = create_model(
            "QuestionClassification",
            question_type=(str, Field(...)),
            certainty=(int, Field(..., ge=1, le=10)),
        )

        allowed_query_types = set(query_types)

        for attempt in range(5):
            try:
                completion = self._llm_client.beta.chat.completions.parse(
                    model=self._deployment_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=QuestionClassification,
                )
                init_output = completion.choices[0].message.parsed
                if init_output:
                    query_type = init_output.question_type
                    if query_type not in allowed_query_types:
                        query_type = "OTHER"
                    certainty = init_output.certainty

                    print(f"[CLASSIFIER] Query: '{messages_string[:50]}...'")
                    print(f"[CLASSIFIER] Type: {query_type}, Certainty: {certainty}/10")

                    return {
                        "query_type": query_type,
                        "certainty": certainty,
                        "awaiting_confirmation": False,
                    }

            except RuntimeError as exc:
                if attempt == 4:
                    raise
                print(f"[CLASSIFIER] Attempt {attempt + 1} failed: {exc}")

        return {
            "query_type": "unknown",
            "certainty": 1,
            "awaiting_confirmation": False,
        }

    def parameter_extraction_node(self, state: ConversationState) -> dict:
        """
        Extracts input parameters required for the SQL query template
        based on the classified query type.
        """
        query_type = state.get("query_type", "unknown")
        messages_string = self._messages_to_string(state["messages"])

        scenario = next(
            (
                scenario
                for scenario in self._scenarios
                if scenario.get("question_type") == query_type
            ),
            None,
        )

        if not scenario:
            print(
                f"[PARAMETER EXTRACTION] No scenario found for query type: {query_type}"
            )
            return {"extracted_params": {}}

        question_inputs = scenario.get("question_inputs", [])

        if not question_inputs:
            print(
                f"[PARAMETER EXTRACTION] No inputs required for query type: {query_type}"
            )
            return {"extracted_params": {}}

        fields: dict[str, tuple[type, Any]] = {}
        param_descriptions: list[str] = []

        for input_spec in question_inputs:
            param_name = input_spec["name"]
            param_type = input_spec["type"]
            param_example = input_spec["example"]
            unique_values = input_spec.get("unique_values", [])

            if param_type == "string":
                field_type = str
            elif param_type == "number":
                field_type = int
            else:
                field_type = str

            fields[param_name] = (
                field_type,
                Field(..., description=f"Example: {param_example}"),
            )

            # Build parameter description with unique values if available
            param_desc = f"- {param_name} ({param_type}): Example: {param_example}"
            if unique_values:
                # Format unique values as a comma-separated list
                values_str = ", ".join(str(v) for v in unique_values)
                param_desc += f". Valid values: [{values_str}]"

            param_descriptions.append(param_desc)

        DynamicParamModel = create_model("DynamicParamModel", **fields)

        param_desc_string = "\n".join(param_descriptions)
        prompt = f"""Based on the conversation: '{messages_string}',

extract the following parameters needed for answering the question about '{query_type}':

{param_desc_string}

ACT LIKE TODAY IS September 2025, i.e. 202509.

Year is always 2025.

IMPORTANT: Only extract parameter values that are explicitly mentioned or very clearly implied in the conversation.
If a parameter is not mentioned or cannot be confidently determined from the context, use the value "unspecified".
Do NOT make assumptions or guess parameter values.

The parameters must match the examples provided when they are specified.

Respond with the extracted parameter values.
"""

        for attempt in range(5):
            try:
                completion = self._llm_client.beta.chat.completions.parse(
                    model=self._deployment_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=DynamicParamModel,
                )
                parsed_output = completion.choices[0].message.parsed

                if parsed_output:
                    extracted_params = parsed_output.model_dump()

                    print(f"[PARAMETER EXTRACTION] Query type: {query_type}")
                    print(f"[PARAMETER EXTRACTION] Extracted parameters:")
                    for key, value in extracted_params.items():
                        print(f"  - {key}: {value}")

                    missing_params: list[dict[str, Any]] = []
                    uncertain_indicators = [
                        "unspecified",
                        "unknown",
                        "not mentioned",
                        "unclear",
                        "not provided",
                        "n/a",
                        "na",
                    ]

                    for input_spec in question_inputs:
                        param_name = input_spec["name"]
                        param_value = extracted_params.get(param_name)

                        is_uncertain = False
                        if param_value is None or param_value == "":
                            is_uncertain = True
                        elif isinstance(param_value, str) and any(
                            indicator in param_value.lower()
                            for indicator in uncertain_indicators
                        ):
                            is_uncertain = True

                        if is_uncertain:
                            missing_params.append(
                                {
                                    "name": param_name,
                                    "type": input_spec["type"],
                                    "example": input_spec["example"],
                                }
                            )

                    if missing_params:
                        print(
                            "[PARAMETER EXTRACTION] Missing or uncertain parameters detected: "
                            f"{[param['name'] for param in missing_params]}"
                        )
                        return {
                            "extracted_params": extracted_params,
                            "missing_params": missing_params,
                            "awaiting_clarification": True,
                        }

                    print(
                        "[PARAMETER EXTRACTION] All parameters successfully extracted"
                    )
                    return {
                        "extracted_params": extracted_params,
                        "missing_params": None,
                        "awaiting_clarification": False,
                    }

            except Exception as exc:
                if attempt == 4:
                    print(f"[PARAMETER EXTRACTION] Failed after 5 attempts: {exc}")
                    return {"extracted_params": {}}
                print(f"[PARAMETER EXTRACTION] Attempt {attempt + 1} failed: {exc}")

        return {"extracted_params": {}}

    def sql_node(self, state: ConversationState) -> dict:
        """
        Generates SQL query by filling the template with extracted parameters,
        executes it against the SQLite database, and prints the results.
        """
        query_type = state.get("query_type", "unknown")
        extracted_params = state.get("extracted_params", {})

        scenario = next(
            (
                scenario
                for scenario in self._scenarios
                if scenario.get("question_type") == query_type
            ),
            None,
        )

        if not scenario:
            print(f"[SQL NODE] No scenario found for query type: {query_type}")
            return {"sql_query": None}

        sql_template = scenario.get("question_sql_template", "")

        if not sql_template:
            print(f"[SQL NODE] No SQL template found for query type: {query_type}")
            return {"sql_query": None}

        sql_query = sql_template
        for param_name, param_value in extracted_params.items():
            placeholder = f"{{{{{param_name}}}}}"
            sql_query = sql_query.replace(placeholder, str(param_value))

        print("[SQL NODE] Generated SQL Query:")
        print(sql_query)
        print()

        # Execute the SQL query and store results
        try:
            with sqlite3.connect(str(self._db_path)) as connection:
                cursor = connection.cursor()
                cursor.execute(sql_query)

                # Fetch column names
                column_names = (
                    [description[0] for description in cursor.description]
                    if cursor.description
                    else []
                )

                # Fetch all results for processing
                all_results = cursor.fetchall()

                # Store results in state
                sql_results = {
                    "columns": column_names,
                    "rows": [list(row) for row in all_results],
                }

                # Print first 5 rows for debugging
                if all_results:
                    print("[SQL NODE] Query Results (first 5 rows):")
                    print("-" * 80)

                    # Print column headers
                    if column_names:
                        print(" | ".join(column_names))
                        print("-" * 80)

                    # Print first 5 rows
                    for row in all_results[:5]:
                        print(" | ".join(str(value) for value in row))

                    print("-" * 80)
                    print(f"[SQL NODE] Total rows fetched: {len(all_results)}")
                    print()
                else:
                    print("[SQL NODE] Query returned no results.")
                    print()
                    sql_results = {"columns": [], "rows": []}

        except sqlite3.Error as e:
            print(f"[SQL NODE] Error executing SQL query: {e}")
            print()
            sql_results = {"columns": [], "rows": []}

        return {"sql_query": sql_query, "sql_results": sql_results}

    def answering_node(self, state: ConversationState) -> dict:
        """
        Generates a natural language response based on SQL results using Azure LLM.
        Uses the answer_template from scenarios as a style guide.
        """
        query_type = state.get("query_type", "unknown")
        sql_results = state.get("sql_results", {"columns": [], "rows": []})
        messages = state.get("messages", [])

        # Find the scenario for this query type
        scenario = next(
            (
                scenario
                for scenario in self._scenarios
                if scenario.get("question_type") == query_type
            ),
            None,
        )

        if not scenario:
            print(f"[ANSWERING NODE] No scenario found for query type: {query_type}")
            fallback_response = "I've executed your query, but I'm not sure how to interpret the results."
            return {
                "messages": [AIMessage(content=fallback_response)],
                "awaiting_confirmation": False,
            }

        # Get the answer template
        answer_template = scenario.get("answer_template", "")

        # Get the user's original question
        user_question = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_question = msg.content
                break

        # Format SQL results as a readable string
        if not sql_results or not sql_results.get("rows"):
            print("[ANSWERING NODE] No SQL results available")
            response = "No data has been retrieved for this sql query"
            return {
                "messages": [AIMessage(content=response)],
                "awaiting_confirmation": False,
            }

        # Create a table representation of the results
        columns = sql_results.get("columns", [])
        rows = sql_results.get("rows", [])

        results_text = "SQL Query Results:\n\n"
        if columns:
            results_text += " | ".join(columns) + "\n"
            results_text += "-" * (len(" | ".join(columns))) + "\n"

        # Include all rows (or limit if too many)
        max_rows = 100  # Limit to prevent token overflow
        for row in rows[:max_rows]:
            results_text += " | ".join(str(value) for value in row) + "\n"

        if len(rows) > max_rows:
            results_text += f"\n... ({len(rows) - max_rows} more rows)\n"

        # Build prompt for Azure OpenAI
        prompt = f"""You are a business analyst generating insights from data.

User's question: "{user_question}"

{results_text}

Based on the SQL results above, generate a clear, professional, and concise business summary that answers the user's question.

Use the following answer template as a style guide for formatting and tone (but use the actual SQL data provided above, not the template data):

{answer_template}

Important:
- The latest closed month is September 2025
- Use the ACTUAL data from the SQL results provided
- Follow the style and structure of the template
- Be specific with numbers and percentages
- Keep the response professional and concise
- If comparing periods, clearly state the metrics and differences

Generate the response now:"""

        # Call Azure OpenAI
        try:
            print("[ANSWERING NODE] Generating response with Azure OpenAI...")
            completion = self._llm_client.chat.completions.create(
                model=self._deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a business analyst expert in financial reporting and data analysis.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1000,
            )

            generated_response = completion.choices[0].message.content

            if not generated_response:
                generated_response = "I apologize, but I couldn't generate a proper response. Here's the raw data summary."

            print(
                f"[ANSWERING NODE] Generated response ({len(generated_response)} chars)"
            )
            print(f"[ANSWERING NODE] Preview: {generated_response[:100]}...")

            return {
                "messages": [AIMessage(content=generated_response)],
                "awaiting_confirmation": False,
                "display_dataframe": sql_results,
            }

        except Exception as e:
            print(f"[ANSWERING NODE] Error generating response: {e}")
            # Fallback to basic summary
            fallback_response = f"I found {len(rows)} result(s) for your query, but encountered an error formatting the response."
            return {
                "messages": [AIMessage(content=fallback_response)],
                "awaiting_confirmation": False,
                "display_dataframe": sql_results,
            }

    def confirmation_node(self, state: ConversationState) -> dict:
        """Asks the user to confirm the classified query when certainty is medium."""
        query_type = state.get("query_type", "unknown")

        confirmation_msg = (
            f"I think you're asking about: **{query_type.replace('_', ' ')}**. "
            f"Is that correct? (yes/no)"
        )

        print(f"[CONFIRMATION] Asking for confirmation on: {query_type}")

        return {
            "messages": [AIMessage(content=confirmation_msg)],
            "awaiting_confirmation": True,
        }

    def low_certainty_node(self, state: ConversationState) -> dict:
        """Handles low certainty by asking user to start over with more details."""
        msg = (
            "I'm not quite sure what you're asking about.\n"
            "Could you please rephrase your question with more details? \n"
            "For example, you can ask about: \n"
            "- Executive summary of performance for HQ \n"
            "- Executive summary of YTD performance by region for HQ \n"
            "- Show top and bottom Gross Profit growth country and category combination \n"
        )

        print("[LOW CERTAINTY] Asking user to clarify")

        return {
            "messages": [AIMessage(content=msg)],
            "awaiting_confirmation": False,
        }

    def clarification_node(self, state: ConversationState) -> dict:
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
            f"[CLARIFICATION] Asking for missing parameters: {[param['name'] for param in missing_params]}"
        )

        return {
            "messages": [AIMessage(content=clarification_msg)],
            "awaiting_clarification": True,
        }

    def route_after_classification(
        self, state: ConversationState
    ) -> Literal[
        "parameter_extraction_node", "confirmation_node", "low_certainty_node"
    ]:
        """Routes based on certainty level."""
        query_type = state.get("query_type")

        if query_type == "OTHER":
            print("[ROUTER] Query classified as OTHER → low certainty handler")
            return "low_certainty_node"

        certainty = state.get("certainty", 1)

        if certainty >= 8:
            print(f"[ROUTER] High certainty ({certainty}) → parameter extraction")
            return "parameter_extraction_node"
        if certainty >= 4:
            print(f"[ROUTER] Mid certainty ({certainty}) → confirmation")
            return "confirmation_node"
        print(f"[ROUTER] Low certainty ({certainty}) → low certainty handler")
        return "low_certainty_node"

    def route_after_confirmation(
        self, state: ConversationState
    ) -> Literal["parameter_extraction_node", "classifier_node", END]:
        """Routes after confirmation node based on the user response."""
        messages = state["messages"]
        awaiting = state.get("awaiting_confirmation", False)

        if not awaiting:
            return END

        last_message = None
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                last_message = message.content.lower()
                break

        if not last_message:
            return END

        if any(
            word in last_message for word in ["yes", "yeah", "correct", "right", "yep"]
        ):
            print("[CONFIRMATION ROUTER] User confirmed → parameter extraction")
            return "parameter_extraction_node"
        if any(word in last_message for word in ["no", "nope", "wrong", "different"]):
            print("[CONFIRMATION ROUTER] User denied → reclassify")
            return "classifier_node"
        print("[CONFIRMATION ROUTER] Unclear response → END")
        return END

    def route_after_parameter_extraction(
        self, state: ConversationState
    ) -> Literal["sql_node", "clarification_node"]:
        """Routes after parameter extraction."""
        missing_params = state.get("missing_params")

        if missing_params:
            print("[PARAM ROUTER] Missing parameters detected → clarification")
            return "clarification_node"
        print("[PARAM ROUTER] All parameters extracted → sql_node")
        return "sql_node"

    def _build_graph(self):
        graph = StateGraph(ConversationState)

        graph.add_node("classifier_node", self.classifier_node)
        graph.add_node("parameter_extraction_node", self.parameter_extraction_node)
        graph.add_node("sql_node", self.sql_node)
        graph.add_node("answering_node", self.answering_node)
        graph.add_node("confirmation_node", self.confirmation_node)
        graph.add_node("low_certainty_node", self.low_certainty_node)
        graph.add_node("clarification_node", self.clarification_node)

        graph.add_edge(START, "classifier_node")

        graph.add_conditional_edges(
            "classifier_node",
            self.route_after_classification,
            {
                "parameter_extraction_node": "parameter_extraction_node",
                "confirmation_node": "confirmation_node",
                "low_certainty_node": "low_certainty_node",
            },
        )

        graph.add_conditional_edges(
            "parameter_extraction_node",
            self.route_after_parameter_extraction,
            {
                "sql_node": "sql_node",
                "clarification_node": "clarification_node",
            },
        )

        graph.add_edge("sql_node", "answering_node")
        graph.add_edge("answering_node", END)
        graph.add_edge("low_certainty_node", END)
        graph.add_edge("confirmation_node", END)
        graph.add_edge("clarification_node", END)

        return graph.compile()


def get_backend(env_path: str | os.PathLike[str] | None = ".env") -> FlowBackend:
    """
    Convenience factory that returns a cached backend instance.
    Useful for frontends that want a singleton per process.
    """
    # Basic module-level cache
    global _BACKEND_SINGLETON  # type: ignore[global-var-not-assigned]
    try:
        backend = _BACKEND_SINGLETON  # type: ignore[name-defined]
    except NameError:
        backend = FlowBackend(env_path=env_path)
        _BACKEND_SINGLETON = backend  # type: ignore[name-defined]
    return backend


__all__ = [
    "ConversationState",
    "ConversationTurnResult",
    "FlowBackend",
    "get_backend",
]
