from dotenv import load_dotenv
import os
import yaml
from openai import AzureOpenAI, OpenAI
from groq import Groq
from google import genai

load_dotenv()


def create_llm_client(llm_model_input: str, llm_model_dict: dict):
    if ":" in llm_model_input:
        try:
            model_location, resolved_model_name = llm_model_input.split(":", 1)
            if model_location not in llm_model_dict:
                print(
                    f"Warning: Explicit location '{model_location}' not found in config keys. Proceeding anyway."
                )
        except ValueError:
            print(
                f"Error: Invalid model input format '{llm_model_input}'. Expected 'location:model_name' or 'model_name'."
            )
            return None, None, None
    else:
        resolved_model_name = llm_model_input
        llm_locations = [
            key
            for key, models in llm_model_dict.items()
            if resolved_model_name in models
        ]
        if not llm_locations:
            print(
                f"Error: Model '{resolved_model_name}' not found in any location in the config."
            )
            return None, None, None
        model_location = llm_locations[0]  # Take the first location found

    print(
        f"Attempting to activate client for: location='{model_location}', model='{resolved_model_name}'"
    )

    # 2. Instantiate the client based on location
    client = None
    try:
        if model_location == "azure_openai":
            client = AzureOpenAI(
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                api_version=os.environ.get("OPENAI_API_VERSION"),
            )
        elif model_location == "priv_openai":
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
        elif model_location == "lingaro":
            client = OpenAI(
                base_url="https://llm.lingarogroup.com",
                api_key=os.environ.get("LINGARO_API_KEY"),
            )
        elif model_location == "dbrx":
            # Assuming temperature is a default, can be overridden in the call if needed
            client = OpenAI(
                api_key=os.environ.get("DATABRICKS_TOKEN"),
                base_url=os.environ.get("DATABRICKS_ENDPOINT"),
            )
        elif model_location == "openrouter":
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
            )
        elif model_location == "deepseek":
            client = OpenAI(
                base_url="https://api.deepseek.com/",
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
            )
        elif model_location == "groq":
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        elif model_location == "google_ai_studio":
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set.")
            client = genai.Client(
                api_key=api_key,
            )
        else:
            print(f"Error: Unknown model location '{model_location}'.")
            return None, None, None

        if client is None:
            # This might happen if an env var is missing for a specific client type
            print(
                f"Error: Client creation failed for location '{model_location}'. Check environment variables."
            )
            return None, None, None

    except Exception as e:
        print(f"Error during client instantiation for {model_location}: {e}")
        return None, None, None

    return client, model_location, resolved_model_name


def get_and_print_openai_models():
    """
    Gets available OpenAI models, sorts them by creation date (newest first),
    and prints their IDs along with human-readable creation dates.
    """
    import datetime  # Add this import to the top of the file if not already present

    config_path = os.path.join("src", "llms", "llm_config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    llm_model_dict = config.get("llm_location", {})

    priv_openai_client = create_llm_client("priv_openai:gpt-4o", llm_model_dict)
    if priv_openai_client is not None:
        print("Private OpenAI client created successfully.")
        openai_models = priv_openai_client[0].models.list()

        # Sort models by creation date (newest first)
        sorted_models = sorted(
            openai_models.data, key=lambda x: x.created, reverse=True
        )

        print("Available models (newest to oldest):")
        for model in sorted_models:
            # Convert Unix timestamp to readable date format
            created_date = datetime.datetime.fromtimestamp(model.created).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            print(f"id='{model.id}', created='{created_date}'")
