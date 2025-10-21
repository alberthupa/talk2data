"""
Unified structured output module for different LLM providers.

Handles structured outputs using:
- Native structured outputs for Azure OpenAI/OpenAI (beta.chat.completions.parse)
- JSON mode with validation for other providers
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from openai import AzureOpenAI, OpenAI
    from groq import Groq

T = TypeVar("T", bound=BaseModel)


def get_structured_response(
    client: "AzureOpenAI | OpenAI | Groq | Any",
    model_location: str,
    model_name: str,
    messages: list[dict[str, str]],
    response_model: type[T],
    max_retries: int = 5,
) -> T | None:
    """
    Get structured output from an LLM using the appropriate method.

    Args:
        client: The LLM client instance
        model_location: Provider location (e.g., 'azure_openai', 'groq', etc.)
        model_name: Model name/deployment name
        messages: List of message dicts with 'role' and 'content' keys
        response_model: Pydantic model class for structured output
        max_retries: Maximum number of retry attempts

    Returns:
        Parsed Pydantic model instance or None if all retries fail
    """
    # Providers that support native structured outputs via beta API
    native_structured_providers = ["azure_openai", "priv_openai"]

    for attempt in range(max_retries):
        try:
            if model_location in native_structured_providers:
                # Use native structured output (Azure OpenAI / OpenAI)
                return _get_native_structured_response(
                    client, model_name, messages, response_model
                )
            else:
                # Use JSON mode fallback for other providers
                return _get_json_mode_structured_response(
                    client, model_location, model_name, messages, response_model
                )

        except (RuntimeError, ValidationError, json.JSONDecodeError) as exc:
            if attempt == max_retries - 1:
                print(
                    f"[STRUCTURED OUTPUT] Failed after {max_retries} attempts: {exc}"
                )
                return None
            print(f"[STRUCTURED OUTPUT] Attempt {attempt + 1} failed: {exc}")

    return None


def _get_native_structured_response(
    client: "AzureOpenAI | OpenAI",
    model_name: str,
    messages: list[dict[str, str]],
    response_model: type[T],
) -> T:
    """
    Get structured output using native Azure OpenAI/OpenAI beta API.
    """
    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=messages,
        response_format=response_model,
    )

    parsed = completion.choices[0].message.parsed
    if not parsed:
        raise RuntimeError("No parsed output received from native structured API")

    return parsed


def _get_json_mode_structured_response(
    client: Any,
    model_location: str,
    model_name: str,
    messages: list[dict[str, str]],
    response_model: type[T],
) -> T:
    """
    Get structured output using JSON mode + validation for providers without native support.

    This works by:
    1. Adding instructions to return valid JSON matching the schema
    2. Using response_format: "json_object" if supported
    3. Parsing the JSON response and validating against Pydantic model
    """
    # Get JSON schema from Pydantic model
    schema = response_model.model_json_schema()
    schema_str = json.dumps(schema, indent=2)

    # Enhance the last user message with JSON schema instructions
    enhanced_messages = messages.copy()
    last_message = enhanced_messages[-1]["content"]

    json_instruction = f"""
{last_message}

IMPORTANT: Respond ONLY with valid JSON that matches this exact schema:
{schema_str}

Do not include any explanatory text, markdown formatting, or code blocks.
Return only the raw JSON object.
"""
    enhanced_messages[-1] = {"role": "user", "content": json_instruction}

    # Call the LLM
    # Some providers support response_format, others don't
    if model_location in ["groq", "deepseek", "openrouter", "dbrx"]:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=enhanced_messages,
                response_format={"type": "json_object"},
            )
        except Exception:
            # Fallback if response_format not supported
            completion = client.chat.completions.create(
                model=model_name,
                messages=enhanced_messages,
            )
    else:
        # For providers without response_format support (Gemini, etc.)
        # This would need special handling - for now use BasicAgent pattern
        completion = client.chat.completions.create(
            model=model_name,
            messages=enhanced_messages,
        )

    # Extract and parse JSON response
    response_text = completion.choices[0].message.content

    # Try to extract JSON from response (in case it's wrapped in code blocks)
    response_text = _extract_json_from_text(response_text)

    # Parse JSON
    response_dict = json.loads(response_text)

    # Validate against Pydantic model
    validated_model = response_model.model_validate(response_dict)

    return validated_model


def _extract_json_from_text(text: str) -> str:
    """
    Extract JSON from text that might be wrapped in markdown code blocks.
    """
    text = text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```json"):
        text = text[7:]  # Remove ```json
    elif text.startswith("```"):
        text = text[3:]  # Remove ```

    if text.endswith("```"):
        text = text[:-3]  # Remove trailing ```

    return text.strip()
