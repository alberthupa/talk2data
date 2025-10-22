from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flow_backend import ConversationState, FlowBackend


def sql_node(backend: "FlowBackend", state: "ConversationState") -> dict:
    """
    Generates SQL query by filling the template with extracted parameters,
    executes it against the SQLite database, and prints the results.
    """
    query_type = state.get("query_type", "unknown")
    extracted_params = state.get("extracted_params", {})

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
                print("[SQL NODE] Query Results (first 5 rows):")
                print("-" * 80)

                if column_names:
                    print(" | ".join(column_names))
                    print("-" * 80)

                for row in all_results[:5]:
                    print(" | ".join(str(value) for value in row))

                print("-" * 80)
                print(f"[SQL NODE] Total rows fetched: {len(all_results)}")
                print()
            else:
                print("[SQL NODE] Query returned no results.")
                print()
                sql_results = {"columns": [], "rows": []}

    except sqlite3.Error as error:
        print(f"[SQL NODE] Error executing SQL query: {error}")
        print()
        sql_results = {"columns": [], "rows": []}

    return {"sql_query": sql_query, "sql_results": sql_results}
