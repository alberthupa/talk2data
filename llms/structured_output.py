"""
Unified structured output module for different LLM providers.

Handles structured outputs using:
- Native structured outputs for Azure OpenAI/OpenAI (beta.chat.completions.parse)
- JSON mode with validation for other providers
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError
from google.genai import types as genai_types

from llms.basic_agent import translate_messages_from_openai_to_gemini

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
    property_keys = sorted(schema.get("properties", {}).keys())
    keys_text = ", ".join(property_keys) if property_keys else "the required fields"
    example_obj = {key: f"VALUE_FOR_{key}" for key in property_keys}
    example_str = json.dumps(example_obj, indent=2) if property_keys else "{}"

    enhanced_messages = [msg.copy() for msg in messages]
    last_message = enhanced_messages[-1]["content"]

    json_instruction = f"""
{last_message}

IMPORTANT: Respond ONLY with a valid JSON object that matches this exact schema:
{schema_str}

The JSON must contain the keys: {keys_text}.
Populate each key with the best extracted value from the conversation.
If a value is unknown, set it to null.

Example format (replace VALUE_FOR_* with actual values):
{example_str}

Do not include any explanatory text, markdown formatting, or code blocks.
Do not repeat or describe the schema or the instructions.
Return only the raw JSON object.
"""
    enhanced_messages[-1] = {"role": "user", "content": json_instruction}

    # Call the LLM
    # Some providers support response_format, others don't
    if model_location == "openrouter":
        openrouter_messages = [
            {
                "role": "system",
                "content": (
                    "You must reply with a single JSON object that matches the provided schema. "
                    f"The JSON must contain the keys: {keys_text}. "
                    "Populate each key with the extracted value from the conversation, or null if unknown. "
                    f"Example format (replace VALUE_FOR_* with actual values):\n{example_str}\n"
                    "Do not include explanations, markdown, repeated schema text, or <think> sections."
                ),
            },
            *enhanced_messages,
        ]
        return _get_openrouter_structured_response(
            client=client,
            model_name=model_name,
            messages=openrouter_messages,
            response_model=response_model,
        )

    if model_location == "google_ai_studio":
        gemini_messages, system_instruction = translate_messages_from_openai_to_gemini(
            enhanced_messages
        )
        if not gemini_messages:
            raise RuntimeError(
                "Gemini request has no user/model messages after translation."
            )

        config_kwargs: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": response_model,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        config = genai_types.GenerateContentConfig(**config_kwargs)
        response = client.models.generate_content(
            model=model_name,
            contents=gemini_messages,
            config=config,
        )

        parsed = getattr(response, "parsed", None)
        try:
            structured = _coerce_gemini_parsed(parsed, response_model)
        except ValidationError as exc:
            raise RuntimeError(f"Gemini structured output validation failed: {exc}") from exc
        if structured is not None:
            return structured

        response_text = getattr(response, "text", None)
        if not response_text and hasattr(response, "parts"):
            parts = getattr(response, "parts")
            collected_parts: list[str] = []
            for part in parts or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    collected_parts.append(part_text)
            if collected_parts:
                response_text = "\n".join(collected_parts)

        if not response_text:
            raise RuntimeError("Gemini response did not contain text output.")

    elif model_location in ["groq", "deepseek", "dbrx"]:
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
        response_text = completion.choices[0].message.content
    else:
        # For providers without response_format support that still use OpenAI-compatible clients
        completion = client.chat.completions.create(
            model=model_name,
            messages=enhanced_messages,
        )
        response_text = completion.choices[0].message.content

    # Try to extract JSON from response (in case it's wrapped in code blocks)
    response_text = _extract_json_from_text(response_text)

    # Parse JSON
    response_dict = json.loads(response_text)

    # Validate against Pydantic model
    validated_model = response_model.model_validate(response_dict)

    return validated_model


def _coerce_gemini_parsed(
    parsed: Any,
    response_model: type[T],
) -> T | None:
    """
    Convert Gemini structured output payloads into the expected Pydantic model.
    """
    if parsed is None:
        return None

    candidate = parsed
    if isinstance(candidate, list):
        if not candidate:
            return None
        # If Gemini returns a list of items, pick the first structured response.
        candidate = candidate[0]

    if isinstance(candidate, response_model):
        return candidate

    if hasattr(candidate, "model_dump"):
        candidate = candidate.model_dump()

    if isinstance(candidate, dict):
        return response_model.model_validate(candidate)

    return None


def _get_openrouter_structured_response(
    client: Any,
    model_name: str,
    messages: list[dict[str, str]],
    response_model: type[T],
) -> T:
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
        )
    except Exception:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )

    message = completion.choices[0].message
    response_text = _normalize_openrouter_message_content(message)
    response_text = _strip_openrouter_wrappers(response_text)
    response_text = _extract_json_from_text(response_text)

    response_dict = json.loads(response_text)
    return response_model.model_validate(response_dict)


def _normalize_openrouter_message_content(message: Any) -> str:
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in (None, "output_text", "text"):
                    text_value = item.get("text") or item.get("content") or ""
                    parts.append(text_value)
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)

    if isinstance(content, str):
        return content

    if content is None:
        return ""

    return str(content)


def _strip_openrouter_wrappers(text: str) -> str:
    if not text:
        return text

    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


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
