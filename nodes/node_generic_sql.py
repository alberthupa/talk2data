from __future__ import annotations

import json
import re
import sqlite3
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from flow_backend import ConversationState, FlowBackend


def generic_sql_node(backend: "FlowBackend", state: "ConversationState") -> dict:
    """
    Generates and executes a generic SQL query using LLM when user opts for
    generic path during medium-certainty confirmation.
    """
    print("[GENERIC SQL NODE] Starting generic SQL generation...")

    # Build context for the LLM
    conversation_history = backend._messages_to_string(state.get("messages", []))

    # Get classifier guess if available
    query_type = state.get("query_type", "unknown")
    classifier_context = (
        f"The system initially classified this as: {query_type.replace('_', ' ')}\n"
        if query_type and query_type != "unknown"
        else ""
    )

    # Get extracted params if any
    extracted_params = state.get("extracted_params", {})
    params_context = ""
    if extracted_params:
        params_context = f"Extracted parameters: {json.dumps(extracted_params, indent=2)}\n"

    # Get database schema
    schema = backend.get_schema_prompt()

    # Build the prompt for SQL generation
    prompt = f"""You are a SQL expert helping a user analyze market performance data.

Conversation history:
{conversation_history}

{classifier_context}{params_context}

{schema}

Based on the user's question and the database schema above, generate a SQL query that answers their question.

IMPORTANT INSTRUCTIONS:
1. Return ONLY valid SQLite syntax
2. Use proper table and column names from the schema
3. Include appropriate WHERE clauses, GROUP BY, ORDER BY as needed
4. Limit results to a reasonable number (use LIMIT if appropriate)
5. Format your response as JSON with two keys:
   - "sql": the SQL query as a string
   - "rationale": brief explanation of what the query does (1-2 sentences)

Example response format:
{{
  "sql": "SELECT category, SUM(net_revenue) as total_revenue FROM sales WHERE year = 2025 GROUP BY category ORDER BY total_revenue DESC LIMIT 10",
  "rationale": "This query calculates total net revenue by category for 2025, ordered by revenue descending, limited to top 10."
}}

Generate the SQL query now:"""

    try:
        print("[GENERIC SQL NODE] Requesting SQL generation from LLM...")

        # Get LLM client info from state
        llm_model_input = state.get("llm_model_input", backend._default_llm_model)

        response_dict = backend._llm_agent.get_text_response_from_llm(
            llm_model_input=llm_model_input,
            messages=[
                {
                    "role": "system",
                    "content": "You are a SQL expert. Always respond with valid JSON containing 'sql' and 'rationale' keys.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        generated_text = response_dict.get("text_response", "")

        if not generated_text:
            error_msg = (
                "I apologize, but I couldn't generate a SQL query. "
                "Please try rephrasing your question or select a predefined scenario."
            )
            return {
                "messages": [AIMessage(content=error_msg)],
                "generic_sql_attempted": True,
                "generic_sql_error": "Empty LLM response",
                "awaiting_confirmation": False,
                "awaiting_generic_choice": False,
            }

        # Parse the JSON response
        try:
            # Try to extract JSON from the response (handle markdown code blocks)
            json_match = re.search(r'```json\s*(.*?)\s*```', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
                json_str = json_match.group(0) if json_match else generated_text

            parsed_response = json.loads(json_str)
            sql_query = parsed_response.get("sql", "").strip()
            rationale = parsed_response.get("rationale", "")

        except (json.JSONDecodeError, AttributeError) as parse_error:
            print(f"[GENERIC SQL NODE] Failed to parse JSON response: {parse_error}")
            print(f"[GENERIC SQL NODE] Raw response: {generated_text[:200]}...")

            error_msg = (
                "I generated a query but had trouble parsing it. "
                "Please try rephrasing your question or select a predefined scenario."
            )
            return {
                "messages": [AIMessage(content=error_msg)],
                "generic_sql_attempted": True,
                "generic_sql_error": f"JSON parse error: {parse_error}",
                "awaiting_confirmation": False,
                "awaiting_generic_choice": False,
            }

        if not sql_query:
            error_msg = (
                "I couldn't generate a valid SQL query for your question. "
                "Please try rephrasing or select a predefined scenario."
            )
            return {
                "messages": [AIMessage(content=error_msg)],
                "generic_sql_attempted": True,
                "generic_sql_error": "Empty SQL query",
                "awaiting_confirmation": False,
                "awaiting_generic_choice": False,
            }

        print(f"[GENERIC SQL NODE] Generated SQL: {sql_query}")
        print(f"[GENERIC SQL NODE] Rationale: {rationale}")

        # Execute the SQL query
        try:
            with sqlite3.connect(str(backend._db_path)) as connection:
                cursor = connection.cursor()
                cursor.execute(sql_query)

                column_names = (
                    [description[0] for description in cursor.description]
                    if cursor.description
                    else []
                )

                all_results = cursor.fetchall()

                sql_results = {
                    "columns": column_names,
                    "rows": [list(row) for row in all_results],
                }

                if all_results:
                    print(f"[GENERIC SQL NODE] Query returned {len(all_results)} rows")
                    print("[GENERIC SQL NODE] Query Results (first 5 rows):")
                    print("-" * 80)

                    if column_names:
                        print(" | ".join(column_names))
                        print("-" * 80)

                    for row in all_results[:5]:
                        print(" | ".join(str(value) for value in row))

                    print("-" * 80)
                else:
                    print("[GENERIC SQL NODE] Query returned no results")
                    sql_results = {"columns": [], "rows": []}

                # Create a message with the rationale
                rationale_msg = f"Generated query: {rationale}" if rationale else "Query executed successfully."

                return {
                    "sql_query": sql_query,
                    "sql_results": sql_results,
                    "messages": [AIMessage(content=rationale_msg)],
                    "generic_sql_attempted": True,
                    "generic_sql_error": None,
                    "awaiting_confirmation": False,
                    "awaiting_generic_choice": False,
                }

        except sqlite3.Error as sql_error:
            print(f"[GENERIC SQL NODE] SQL execution error: {sql_error}")

            error_msg = (
                f"I generated a SQL query but encountered an error executing it:\n\n"
                f"**Error:** {sql_error}\n\n"
                f"**Query:** `{sql_query}`\n\n"
                f"Please try rephrasing your question or select a predefined scenario."
            )

            return {
                "messages": [AIMessage(content=error_msg)],
                "sql_query": sql_query,
                "generic_sql_attempted": True,
                "generic_sql_error": str(sql_error),
                "awaiting_confirmation": False,
                "awaiting_generic_choice": False,
            }

    except Exception as error:
        print(f"[GENERIC SQL NODE] Unexpected error: {error}")

        error_msg = (
            "I encountered an unexpected error while trying to generate a SQL query. "
            "Please try again or select a predefined scenario."
        )

        return {
            "messages": [AIMessage(content=error_msg)],
            "generic_sql_attempted": True,
            "generic_sql_error": str(error),
            "awaiting_confirmation": False,
            "awaiting_generic_choice": False,
        }
