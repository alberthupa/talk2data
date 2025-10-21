from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field, create_model
from llms.structured_output import get_structured_response

if TYPE_CHECKING:
    from flow_backend import ConversationState, FlowBackend


def parameter_extraction_node(
    backend: "FlowBackend", state: "ConversationState"
) -> dict:
    """
    Extracts input parameters required for the SQL query template
    based on the classified query type.
    """
    query_type = state.get("query_type", "unknown")
    messages_string = backend._messages_to_string(state["messages"])

    scenario = next(
        (
            scenario
            for scenario in backend._scenarios
            if scenario.get("question_type") == query_type
        ),
        None,
    )

    if not scenario:
        print(f"[PARAMETER EXTRACTION] No scenario found for query type: {query_type}")
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

        param_desc = f"- {param_name} ({param_type}): Example: {param_example}"
        if unique_values:
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

    # Get LLM client from state
    client = state.get("llm_client")
    model_location = state.get("llm_model_location")
    model_name = state.get("llm_model_name")

    if not client or not model_location or not model_name:
        print("[PARAMETER EXTRACTION] LLM client not initialized in state")
        return {"extracted_params": {}}

    # Use structured output module
    parsed_output = get_structured_response(
        client=client,
        model_location=model_location,
        model_name=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_model=DynamicParamModel,
        max_retries=5,
    )

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

        print("[PARAMETER EXTRACTION] All parameters successfully extracted")
        return {
            "extracted_params": extracted_params,
            "missing_params": None,
            "awaiting_clarification": False,
        }

    # Fallback if structured output failed
    print("[PARAMETER EXTRACTION] Failed to extract parameters")
    return {"extracted_params": {}}
