# import json
import re
import typing
import yaml
import os
# import re

from tenacity import retry, wait_random_exponential, stop_after_attempt, wait_fixed

from llms.llm_clients import create_llm_client


def translate_messages_from_openai_to_gemini(
    messages_to_change: list[dict[str, str]],
) -> str:
    last_message = messages_to_change[-1]["content"]
    if len(messages_to_change) == 1:
        gemini_messages = []
    else:
        prev_messages = messages_to_change[:-1]
        gemini_messages = []
        for message in prev_messages:
            role = message["role"]
            content = message["content"]
            if role == "assistant":
                role = "model"

            gemini_messages.append({"role": role, "parts": [content]})

    return gemini_messages, last_message


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
                gemini_messages, last_message = (
                    translate_messages_from_openai_to_gemini(messages)
                )
                if hasattr(llm_client, "start_chat"):
                    chat_session = llm_client.start_chat(history=gemini_messages)
                    response = chat_session.send_message(last_message)
                    text_content = response.text

                else:
                    print("Error: Gemini client does not have 'start_chat' method.")
                    text_content = None

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
