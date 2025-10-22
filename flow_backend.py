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
from functools import partial
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from llms.basic_agent import BasicAgent
from llms.llm_clients import create_llm_client
from nodes import (
    answering_node as answering_node_impl,
    clarification_node as clarification_node_impl,
    classifier_node as classifier_node_impl,
    confirmation_node as confirmation_node_impl,
    generic_sql_node as generic_sql_node_impl,
    low_certainty_node as low_certainty_node_impl,
    parameter_extraction_node as parameter_extraction_node_impl,
    sql_node as sql_node_impl,
)


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
    # LLM client management
    llm_model_input: str | None
    llm_client: Any | None
    llm_model_location: str | None
    llm_model_name: str | None
    # Generic SQL support
    confirmation_mode: str | None
    awaiting_generic_choice: bool
    generic_sql_attempted: bool
    generic_sql_error: str | None


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
        llm_model_input: str = "gpt-4o",
        db_path: str | os.PathLike[str] = "sql_data.db",
        preinit_scenario_vectorizer: bool = False,
    ) -> None:
        if env_path:
            load_dotenv(env_path, override=True)

        # Store default LLM model
        self._default_llm_model = llm_model_input

        # Initialize BasicAgent for text responses
        self._llm_agent = BasicAgent()

        # Load scenarios and build descriptions
        self._scenarios = self._load_scenarios(Path(scenarios_path))
        self._question_descriptions = self._build_question_descriptions(self._scenarios)

        # Optionally initialize the TF-IDF vectorizer at startup
        try:
            from scenario_detector import initialize as scenario_init

            if preinit_scenario_vectorizer:
                scenario_init(self._scenarios, enabled=True)
        except Exception:
            # Best effort; detector stays lazy if import fails
            pass

        # Initialize SQLite database path (connections are opened per query)
        self._db_path = Path(db_path)
        self._init_database()

        # Cache for database schema
        self._cached_schema: str | None = None

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
            "llm_model_input": None,
            "llm_client": None,
            "llm_model_location": None,
            "llm_model_name": None,
            "confirmation_mode": None,
            "awaiting_generic_choice": False,
            "generic_sql_attempted": False,
            "generic_sql_error": None,
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

        # Initialize LLM client if needed
        client, location, model_name = self._initialize_llm_client(state)
        if client is None:
            # Failed to initialize client
            error_msg = AIMessage(
                content="Failed to initialize LLM client. Please check your configuration."
            )
            state["messages"].append(error_msg)
            return ConversationTurnResult(state=state, new_messages=[error_msg])

        # Clear stale tabular data before processing the new turn
        state["display_dataframe"] = None
        state["sql_results"] = None

        if state.get("awaiting_clarification"):
            state["messages"].append(HumanMessage(content=user_input))
            print(
                "[CONVERSATION] User provided clarification, re-extracting parameters..."
            )

            param_result = parameter_extraction_node_impl(self, state)
            state.update(param_result)

            if param_result.get("missing_params"):
                clarif_result = clarification_node_impl(self, state)
                state["messages"].extend(clarif_result["messages"])
                state["awaiting_clarification"] = True
                new_messages = state["messages"][previous_len:]
                return ConversationTurnResult(state=state, new_messages=new_messages)

            sql_result = sql_node_impl(self, state)
            state.update(sql_result)

            answer_result = answering_node_impl(self, state)
            state.update(answer_result)
            state["awaiting_clarification"] = False
            new_messages = state["messages"][previous_len:]
            return ConversationTurnResult(state=state, new_messages=new_messages)

        if state.get("awaiting_confirmation"):
            state["messages"].append(HumanMessage(content=user_input))

            user_input_lower = user_input.strip().lower()

            # Option 1: Confirm the scenario
            if user_input_lower == "1" or any(
                word in user_input_lower
                for word in ["yes", "yeah", "correct", "right", "yep", "confirm"]
            ):
                print(
                    "[CONVERSATION] User confirmed scenario, extracting parameters..."
                )
                param_result = parameter_extraction_node_impl(self, state)
                state.update(param_result)

                if param_result.get("missing_params"):
                    clarif_result = clarification_node_impl(self, state)
                    state["messages"].extend(clarif_result["messages"])
                    state["awaiting_confirmation"] = False
                    state["awaiting_generic_choice"] = False
                    state["awaiting_clarification"] = True
                    new_messages = state["messages"][previous_len:]
                    return ConversationTurnResult(
                        state=state, new_messages=new_messages
                    )

                sql_result = sql_node_impl(self, state)
                state.update(sql_result)

                answer_result = answering_node_impl(self, state)
                state.update(answer_result)
                state["awaiting_confirmation"] = False
                state["awaiting_generic_choice"] = False
                new_messages = state["messages"][previous_len:]
                return ConversationTurnResult(state=state, new_messages=new_messages)

            # Option 2: Generic SQL path
            if user_input_lower == "2" or any(
                phrase in user_input_lower
                for phrase in ["generic", "write the query", "craft", "custom sql"]
            ):
                print("[CONVERSATION] User chose generic SQL path...")

                # Invoke generic SQL node
                generic_result = generic_sql_node_impl(self, state)
                state.update(generic_result)

                # If SQL was successfully generated and executed, run answering node
                if generic_result.get("sql_results") and generic_result[
                    "sql_results"
                ].get("rows"):
                    answer_result = answering_node_impl(self, state)
                    state.update(answer_result)

                new_messages = state["messages"][previous_len:]
                return ConversationTurnResult(state=state, new_messages=new_messages)

            # Option 3: Deny and reclassify
            if (
                user_input_lower == "3"
                or user_input_lower == "no"
                or any(
                    word in user_input_lower for word in ["nope", "wrong", "different"]
                )
            ):
                print("[CONVERSATION] User denied, reclassifying...")
                state["awaiting_confirmation"] = False
                state["awaiting_generic_choice"] = False
                result_state = self._graph_app.invoke(state)
                state.update(result_state)
                new_messages = state["messages"][previous_len:]
                return ConversationTurnResult(state=state, new_messages=new_messages)

            # Invalid input
            state["messages"].append(
                AIMessage(
                    content="Please reply with '1' to confirm, '2' for generic SQL, or 'no' to reclassify."
                )
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

    def _initialize_llm_client(
        self, state: ConversationState
    ) -> tuple[Any, str, str] | tuple[None, None, None]:
        """
        Initialize or retrieve LLM client from state.

        Args:
            state: Current conversation state

        Returns:
            Tuple of (client, location, model_name) or (None, None, None) if initialization fails
        """
        # Determine which model to use (state overrides default)
        llm_model_input = state.get("llm_model_input") or self._default_llm_model

        # Check if we need to (re)initialize
        current_model_input = state.get("llm_model_input")
        if state.get("llm_client") is None or current_model_input != llm_model_input:
            print(f"[LLM CLIENT] Initializing client for: {llm_model_input}")

            client, location, model_name = create_llm_client(
                llm_model_input, self._llm_agent.llm_model_dict
            )

            if client is None:
                print(f"[LLM CLIENT] Failed to create client for {llm_model_input}")
                return None, None, None

            # Update state with new client info
            state["llm_client"] = client
            state["llm_model_location"] = location
            state["llm_model_name"] = model_name
            state["llm_model_input"] = llm_model_input

            print(f"[LLM CLIENT] Initialized {location}:{model_name}")

            return client, location, model_name
        else:
            # Return existing client from state
            return (
                state["llm_client"],
                state["llm_model_location"],
                state["llm_model_name"],
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

    def _get_database_schema(self) -> str:
        """
        Retrieves and caches the database schema information.
        Returns a text summary of tables, columns, and types.
        """
        if self._cached_schema is not None:
            return self._cached_schema

        try:
            with sqlite3.connect(str(self._db_path)) as connection:
                cursor = connection.cursor()

                # Get all table names
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
                tables = [row[0] for row in cursor.fetchall()]

                schema_parts = ["Database Schema:\n"]

                for table_name in tables:
                    schema_parts.append(f"\nTable: {table_name}")

                    # Get column information
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()

                    schema_parts.append("Columns:")
                    for col in columns:
                        col_name = col[1]
                        col_type = col[2]
                        is_pk = " (PRIMARY KEY)" if col[5] else ""
                        schema_parts.append(f"  - {col_name}: {col_type}{is_pk}")

                self._cached_schema = "\n".join(schema_parts)
                return self._cached_schema

        except sqlite3.Error as error:
            print(f"[SCHEMA] Error retrieving database schema: {error}")
            return "Error: Unable to retrieve database schema"

    def get_schema_prompt(self) -> str:
        """
        Public wrapper for schema retrieval, suitable for use in prompts.
        """
        return self._get_database_schema()

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
        if certainty >= 2:
            print(f"[ROUTER] Mid certainty ({certainty}) → confirmation")
            return "confirmation_node"
        print(f"[ROUTER] Low certainty ({certainty}) → low certainty handler")
        return "low_certainty_node"

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

        graph.add_node("classifier_node", partial(classifier_node_impl, self))
        graph.add_node(
            "parameter_extraction_node", partial(parameter_extraction_node_impl, self)
        )
        graph.add_node("sql_node", partial(sql_node_impl, self))
        graph.add_node("answering_node", partial(answering_node_impl, self))
        graph.add_node("confirmation_node", partial(confirmation_node_impl, self))
        graph.add_node("low_certainty_node", partial(low_certainty_node_impl, self))
        graph.add_node("clarification_node", partial(clarification_node_impl, self))
        graph.add_node("generic_sql_node", partial(generic_sql_node_impl, self))

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
