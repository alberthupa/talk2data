# import json
import re
import typing
import yaml
import os
# import re

from tenacity import retry, wait_random_exponential, stop_after_attempt, wait_fixed
from google.genai import types as genai_types

from llms.llm_clients import create_llm_client


def translate_messages_from_openai_to_gemini(
    messages_to_change: list[dict[str, typing.Any]],
) -> tuple[list[dict[str, typing.Any]], typing.Optional[str]]:
    """
    Convert OpenAI chat messages to the Google AI Studio format.
    Aggregates system prompts into a single instruction string so it can be
    passed via the `system_instruction` field on the request.
    """
    gemini_messages: list[dict[str, typing.Any]] = []
    system_instruction_segments: list[str] = []

    for message in messages_to_change:
        role = message.get("role")
        content = message.get("content")

        if isinstance(content, list):
            # OpenAI-compatible content can arrive as a list of parts; extract text pieces.
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text" and part.get("text"):
                        text_parts.append(part["text"])
                    elif part.get("text"):
                        text_parts.append(str(part["text"]))
                elif isinstance(part, str):
                    text_parts.append(part)
                else:
                    text_parts.append(str(part))
            content = "\n".join(p for p in text_parts if p)
        elif content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)

        content = content.strip()
        if not content:
            continue

        if role == "system":
            system_instruction_segments.append(content)
            continue

        mapped_role = "model" if role == "assistant" else "user"
        gemini_messages.append(
            {
                "role": mapped_role,
                "parts": [{"text": content}],
            }
        )

    system_instruction = (
        "\n\n".join(system_instruction_segments) if system_instruction_segments else None
    )

    return gemini_messages, system_instruction


class BasicAgent:
    def __init__(self):
        """Initializes the BasicAgent, loading configuration."""
        config_path = "llm_config.yaml"
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            self.llm_model_dict = config.get("llm_location", {})
            if not self.llm_model_dict:
                print(f"Warning: 'llm_location' not found or empty in {config_path}")
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {config_path}")
            self.llm_model_dict = {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML configuration file {config_path}: {e}")
            self.llm_model_dict = {}

        self.llm_client = None
        self.model_location = None
        self.llm_model_name = None
        self._current_llm_input = ()

    @retry(
        wait=wait_fixed(2) + wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
    )
    def get_text_response_from_llm(
        self,
        llm_model_input: str,
        messages: typing.Union[str, list[dict[str, str]]],
        code_tag: str = None,
    ) -> dict:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Check if the client needs to be (re)initialized
        if self.llm_client is None or llm_model_input != self._current_llm_input:
            print(f"Initializing LLM client for: {llm_model_input}")
            client, location, resolved_name = create_llm_client(
                llm_model_input, self.llm_model_dict
            )
            if client is None:
                print(
                    f"Failed to create LLM client for {llm_model_input}. Aborting request."
                )
                return {"text_response": None, "reasoning_content": None}

            self.llm_client = client
            self.model_location = location
            self.llm_model_name = resolved_name
            self._current_llm_input = (
                llm_model_input  # Store the input that created this client
            )
        else:
            print(f"Using existing LLM client for: {self._current_llm_input}")

        llm_client = self.llm_client
        model_location = self.model_location
        llm_model_name_resolved = self.llm_model_name

        reasoning_content = None
        text_content = None

        try:
            if model_location in [
                "azure_openai",
                "dbrx",
                "groq",
                "openrouter",
                "priv_openai",
                "deepseek",
                "lingaro",
            ]:
                my_response = llm_client.chat.completions.create(
                    model=llm_model_name_resolved,
                    messages=messages,
                )
                text_content = my_response.choices[0].message.content
                if hasattr(my_response.choices[0].message, "reasoning_content"):
                    reasoning_content = my_response.choices[0].message.reasoning_content

            elif model_location == "google_ai_studio":
                gemini_messages, system_instruction = (
                    translate_messages_from_openai_to_gemini(messages)
                )
                if not gemini_messages:
                    print(
                        "Warning: Gemini request has no user/model messages after translation."
                    )
                    text_content = None
                else:
                    request_kwargs = {
                        "model": llm_model_name_resolved,
                        "contents": gemini_messages,
                    }
                    if system_instruction:
                        request_kwargs["config"] = genai_types.GenerateContentConfig(
                            system_instruction=system_instruction
                        )

                    response = llm_client.models.generate_content(**request_kwargs)
                    text_content = getattr(response, "text", None)
                    if not text_content and hasattr(response, "parts"):
                        parts = getattr(response, "parts")
                        if parts:
                            collected_parts = []
                            for part in parts:
                                part_text = getattr(part, "text", None)
                                if part_text:
                                    collected_parts.append(part_text)
                            if collected_parts:
                                text_content = "\n".join(collected_parts)

            else:
                print(
                    f"Error: Unsupported model location '{model_location}' encountered in get_text_response."
                )
                return {"text_response": None, "reasoning_content": None}

        except Exception as e:
            print(
                f"Error during LLM API call for {model_location} ({llm_model_name_resolved}): {e}"
            )
            return {"text_response": None, "reasoning_content": None}

        if text_content is not None and code_tag is not None:
            tool_escaping_pattern = rf"```\s?{code_tag}\s?(.*?)```"
            match = re.search(tool_escaping_pattern, text_content, re.DOTALL)
            if match:
                extracted_content = match.group(1).strip()
                return {
                    "text_response": extracted_content,
                    "reasoning_content": None,
                }
            else:
                print(
                    f"Warning: Code tag '{code_tag}' specified but not found in the response."
                )

        return {
            "text_response": text_content,
            "reasoning_content": reasoning_content,
        }
