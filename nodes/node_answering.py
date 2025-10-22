from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage

if TYPE_CHECKING:
    from flow_backend import ConversationState, FlowBackend


def answering_node(backend: "FlowBackend", state: "ConversationState") -> dict:
    """
    Generates a natural language response based on SQL results using Azure LLM.
    Uses the answer_template from scenarios as a style guide.
    """
    query_type = state.get("query_type", "unknown")
    sql_results = state.get("sql_results", {"columns": [], "rows": []})
    messages = state.get("messages", [])

    scenario = next(
        (
            scenario
            for scenario in backend._scenarios
            if scenario.get("question_example") == query_type
            or scenario.get("question_type") == query_type
        ),
        None,
    )

    if not scenario:
        print(f"[ANSWERING NODE] No scenario found for query type: {query_type}")
        fallback_response = (
            "I've executed your query, but I'm not sure how to interpret the results."
        )
        return {
            "messages": [AIMessage(content=fallback_response)],
            "awaiting_confirmation": False,
        }

    answer_template = scenario.get("answer_template", "")

    user_question = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    if not sql_results or not sql_results.get("rows"):
        print("[ANSWERING NODE] No SQL results available")
        response = "No data has been retrieved for this sql query"
        return {
            "messages": [AIMessage(content=response)],
            "awaiting_confirmation": False,
        }

    columns = sql_results.get("columns", [])
    rows = sql_results.get("rows", [])

    results_text = "SQL Query Results:\n\n"
    if columns:
        results_text += " | ".join(columns) + "\n"
        results_text += "-" * (len(" | ".join(columns))) + "\n"

    max_rows = 100
    for row in rows[:max_rows]:
        results_text += " | ".join(str(value) for value in row) + "\n"

    if len(rows) > max_rows:
        results_text += f"\n... ({len(rows) - max_rows} more rows)\n"

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

    try:
        print("[ANSWERING NODE] Generating response with LLM...")

        # Get LLM client info from state
        llm_model_input = state.get("llm_model_input", backend._default_llm_model)

        response_dict = backend._llm_agent.get_text_response_from_llm(
            llm_model_input=llm_model_input,
            messages=[
                {
                    "role": "system",
                    "content": "You are a business analyst expert in financial reporting and data analysis.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        generated_response = response_dict.get("text_response")

        if not generated_response:
            generated_response = (
                "I apologize, but I couldn't generate a proper response. "
                "Here's the raw data summary."
            )

        print(f"[ANSWERING NODE] Generated response ({len(generated_response)} chars)")
        print(f"[ANSWERING NODE] Preview: {generated_response[:100]}...")

        return {
            "messages": [AIMessage(content=generated_response)],
            "awaiting_confirmation": False,
            "display_dataframe": sql_results,
        }

    except Exception as error:
        print(f"[ANSWERING NODE] Error generating response: {error}")
        fallback_response = (
            f"I found {len(rows)} result(s) for your query, "
            "but encountered an error formatting the response."
        )
        return {
            "messages": [AIMessage(content=fallback_response)],
            "awaiting_confirmation": False,
            "display_dataframe": sql_results,
        }
