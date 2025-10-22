# Talk2Data - Market Query Assistant

An AI-powered conversational interface that transforms natural language questions about Mondelez market performance into actionable business insights using LangGraph and multiple LLM providers.

## Overview

This project is a conversational assistant that takes user questions, matches them to predefined scenarios, executes SQL queries, and returns natural language answers. The system supports multiple LLM providers with dynamic model switching, allowing flexibility in cost, performance, and provider selection.

## Code Structure

### Core Application Files

- **`flow_backend.py`** - Bundles the shared conversation engine. It loads environment settings and scenarios, builds the LangGraph pipeline, and exposes helpers (`FlowBackend`, `ConversationTurnResult`) so any interface can process a user turn or initialize state without duplicating business logic.

- **`flow_frontend.py`** - Implements the Streamlit shell. It keeps state in `st.session_state`, renders the chat history and status sidebar, and forwards every user submission to the backend, reacting to `awaiting_confirmation` and `awaiting_clarification` flags with tailored UI prompts. Includes LLM model selector dropdown.

- **`flow.py`** - CLI wrapper that mirrors the original terminal experience. It creates a `FlowBackend` instance, runs the main input loop, and prints new `AIMessage` responses from the backend after each turn. Supports command-line model selection.

- **`scenarios.json`** - Stores scenarios of questions and answers, including question types, descriptions, SQL templates, required parameters, and answer templates.

### LLM Infrastructure (`llms/` directory)

- **`llms/llm_clients.py`** - Multi-provider LLM client factory. Creates client instances for Azure OpenAI, OpenAI, Groq, Google AI Studio (Gemini), Databricks, OpenRouter, DeepSeek, and Lingaro based on configuration.

- **`llms/basic_agent.py`** - Text response handler with retry logic. Manages LLM client lifecycle, handles provider-specific message formatting (especially for Gemini), and provides a unified interface for getting text completions.

- **`llms/structured_output.py`** - Unified structured output module. Uses native structured outputs (beta.chat.completions.parse) for Azure/OpenAI, and JSON mode fallback with Pydantic validation for other providers.

### Nodes (`nodes/` directory)

LangGraph pipeline nodes that handle different stages of conversation:

- **`node_classifier.py`** - Classifies user queries into predefined question types with certainty scoring. Uses structured output to return classification and certainty level (1-10).

- **`node_parameter_extraction.py`** - Extracts required parameters (region, category, year-month, etc.) from the conversation. Dynamically builds Pydantic models based on scenario requirements.

- **`node_confirmation.py`** - Asks user to confirm classification when certainty is medium (4-7).

- **`node_low_certainty.py`** - Handles queries that don't match any scenario or have very low certainty (<4).

- **`node_clarification.py`** - Requests missing parameters from the user with examples.

- **`node_sql.py`** - Generates SQL queries from templates by injecting extracted parameters, executes them against SQLite database, and returns results.

- **`node_answering.py`** - Generates natural language responses from SQL results using the LLM, following the answer template style from scenarios.

### Configuration Files

- **`llm_config.yaml`** - Defines available LLM models organized by provider. Used to populate model selection dropdowns and validate model inputs.

- **`.env`** - Contains API keys and endpoints for all LLM providers (not in repo, see `.env.example`).

- **`pyproject.toml`** - Python project configuration with dependencies managed by `uv`.

### Data Files

- **`sql_data.db`** - SQLite database containing market performance data (artificial data for POC).

### Documentation & Testing

- **`LLM_MIGRATION_SUMMARY.md`** - Complete technical overview of the LLM migration architecture.

- **`QUICK_START_LLM.md`** - User-friendly quick start guide for using different LLM models.

- **`test_llm_migration.py`** - Comprehensive test suite verifying backend initialization, state management, client initialization, conversation flow, and model switching.

- **`CLAUDE.md`** - Instructions for Claude Code when working with this repository.

## Architecture

### Conversation Flow

```
User Input
    |
    v
FlowBackend.run_conversation()
    |
    v
Initialize/Retrieve LLM Client (from state)
    |
    v
LangGraph Pipeline:
    |
    +---> Classifier Node (classify query type)
    |       |
    |       +---> High certainty (>=8) ---> Parameter Extraction
    |       |
    |       +---> Medium certainty (4-7) ---> Confirmation
    |       |
    |       +---> Low certainty (<4) ---> Low Certainty Handler
    |
    +---> Parameter Extraction Node (extract required params)
    |       |
    |       +---> All params found ---> SQL Node
    |       |
    |       +---> Missing params ---> Clarification
    |
    +---> SQL Node (generate & execute query)
    |       |
    |       +---> Answering Node
    |
    +---> Answering Node (generate natural language response)
            |
            +---> Return to User
```

### LLM Client Architecture

```
FlowBackend
  |
  +---> BasicAgent (for text responses)
  |       |
  |       +---> create_llm_client() ---> Provider-specific client
  |
  +---> ConversationState
          |
          +---> llm_client (cached client instance)
          +---> llm_model_location (e.g., "azure_openai", "groq")
          +---> llm_model_name (e.g., "gpt-4o")
          +---> llm_model_input (e.g., "azure_openai:gpt-4o")

Nodes communicate with LLM via:
  +---> Text responses: BasicAgent.get_text_response_from_llm()
  +---> Structured outputs: get_structured_response()
```

## Supported LLM Providers

The application supports multiple LLM providers with seamless switching:

| Provider | Models | Structured Output | Notes |
|----------|--------|-------------------|-------|
| **Azure OpenAI** | gpt-4o, gpt-4o-mini | Native | Recommended for production |
| **OpenAI** | gpt-4o, gpt-4o-mini | Native | Standard OpenAI API |
| **Groq** | llama-3.1-8b-instant, llama-3.3-70b-versatile | JSON fallback | Fast inference |
| **Google AI Studio** | gemini-2.0-flash, gemini-2.5-pro-exp | JSON fallback | Google's models |
| **Databricks** | dbrx, llama-3.1-70b/405b | JSON fallback | Enterprise deployment |
| **OpenRouter** | deepseek-chat, command-r7b | JSON fallback | Multi-model access |
| **DeepSeek** | deepseek-chat, deepseek-reasoner | JSON fallback | Specialized models |
| **Lingaro** | gpt-4o-mini, grok-3 | JSON fallback | Internal gateway |

## Setup & Installation

### Prerequisites

- Python >=3.10, <3.12
- `uv` package manager (or pip)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd talk2data

# Install dependencies with uv
uv sync

# Or with pip
pip install -r requirements.txt
```

### Configuration

1. **Configure API Keys**: Create a `.env` file with your API credentials:

```env
# Azure OpenAI (Required for default behavior)
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
OPENAI_API_VERSION=2024-08-01-preview

# Optional: Other providers
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AI...
DATABRICKS_TOKEN=dapi...
DATABRICKS_ENDPOINT=https://...
OPENROUTER_API_KEY=sk-or-...
DEEPSEEK_API_KEY=sk-...
LINGARO_API_KEY=...
```

2. **Verify Configuration**: Check that `llm_config.yaml` lists your desired models.

3. **Database**: Ensure `sql_data.db` is present (contains market performance data).

## Running the Application

### Option 1: CLI (Terminal Interface)

```bash
# Use default model (gpt-4o)
uv run flow.py

# Specify a specific model
uv run flow.py azure_openai:gpt-4o-mini
uv run flow.py groq:llama-3.1-8b-instant
uv run flow.py google_ai_studio:gemini-2.0-flash

# Or with python directly
python flow.py azure_openai:gpt-4o
```

**Example CLI Session:**

```
============================================================
MARKET QUERY ASSISTANT - Interactive Chat
Using LLM: azure_openai:gpt-4o
============================================================

Welcome! Ask me about:
  - Provide executive summary of performance for HQ
  - Show HQ performance for the Buscuits in July 2025
  ...

You: Show HQ performance for Biscuits in July 2025

[CLASSIFIER] Type: total_performance_category_year_month, Certainty: 9/10
Assistant: Net Revenue reached 50.97 MM EUR, above last year (50.03 MM EUR, +1.9%)...
```

### Option 2: Streamlit (Web Interface)

```bash
# Start Streamlit app
streamlit run flow_frontend.py
```

**Features:**
- **Settings Sidebar**: Select LLM model from dropdown
- **Chat Interface**: Interactive conversation with message history
- **Data Tables**: SQL results displayed as formatted tables
- **Follow-up Buttons**: Context-aware quick actions
- **Scenarios Tab**: View all available query scenarios
- **Source Data Tab**: Access to source data documentation

**Model Selection:**
1. Open sidebar (Settings icon)
2. Choose model from "Select LLM Model" dropdown
3. Model switches immediately - works mid-conversation

### Option 3: Programmatic Usage

```python
from flow_backend import FlowBackend

# Create backend with specific model
backend = FlowBackend(llm_model_input="groq:llama-3.3-70b-versatile")

# Initialize conversation state
state = backend.create_initial_state()

# Process user query
result = backend.run_conversation(
    "Show HQ performance for Biscuits in July 2025",
    state
)

# Access response
for message in result.new_messages:
    print(message.content)

# Continue conversation with updated state
state = result.state
result2 = backend.run_conversation("What about August?", state)
```

### Scenario Detector Pre‑Init (Optional)

- Purpose: Pre-fit the TF‑IDF vectorizer for scenario detection at startup to avoid first‑request latency.
- How to enable: pass `preinit_scenario_vectorizer=True` when constructing `FlowBackend`.

```python
from flow_backend import FlowBackend

# Pre-initialize TF-IDF vectorizer using scenarios.json at startup
backend = FlowBackend(preinit_scenario_vectorizer=True)
```

- Default behavior: If not enabled, the vectorizer initializes lazily on the first classification.
- Dependencies: Uses scikit-learn. If unavailable, detection gracefully falls back to `OTHER` with low certainty.
- Implementation: See `scenario_detector.py` for the detector and cache, and `nodes/node_classifier.py` for the thin caller.

## Model Selection Guide

### Format

Models can be specified in two ways:

1. **Short form**: `gpt-4o`
   - Uses first provider that has this model in `llm_config.yaml`

2. **Full form**: `provider:model`
   - Example: `azure_openai:gpt-4o`
   - Example: `groq:llama-3.1-8b-instant`

### Recommendations by Use Case

**Best Overall Performance:**
```bash
uv run flow.py azure_openai:gpt-4o
```

**Cost-Effective:**
```bash
uv run flow.py azure_openai:gpt-4o-mini
uv run flow.py groq:llama-3.1-8b-instant
```

**Fastest Inference:**
```bash
uv run flow.py groq:llama-3.3-70b-versatile
uv run flow.py google_ai_studio:gemini-2.0-flash
```

**Experimental/Advanced:**
```bash
uv run python flow.py deepseek:deepseek-reasoner
uv run flow.py google_ai_studio:gemini-2.5-pro-exp-03-25
```

**Tests verify:**
- ✓ Backend initialization with multiple models
- ✓ State creation with LLM fields
- ✓ Client initialization and caching
- ✓ Full conversation flow (classification -> extraction -> SQL -> answer)
- ✓ Model switching mid-conversation

## Project Structure

```
talk2data/
├── flow_backend.py          # Core conversation engine
├── flow_frontend.py         # Streamlit web interface
├── flow.py                  # CLI terminal interface
├── scenario_detector.py     # TF-IDF scenario detection & cache
│
├── llms/                    # LLM infrastructure
│   ├── llm_clients.py       # Multi-provider client factory
│   ├── basic_agent.py       # Text response handler
│   └── structured_output.py # Unified structured outputs
│
├── nodes/                   # LangGraph pipeline nodes
│   ├── node_classifier.py
│   ├── node_parameter_extraction.py
│   ├── node_confirmation.py
│   ├── node_clarification.py
│   ├── node_low_certainty.py
│   ├── node_sql.py
│   └── node_answering.py
│
├── scenarios.json           # Question scenarios & SQL templates
├── llm_config.yaml          # Available LLM models by provider
├── sql_data.db              # SQLite database (market data)
│
├── QUICK_START_LLM.md       # User quick start guide
├── CLAUDE.md                # Claude Code instructions
├── .env                     # API keys (not in repo)
└── pyproject.toml           # Dependencies & config
```

## Key Features

### Multi-Provider Support
- Switch between 8+ LLM providers without code changes
- Each provider configured with API keys in `.env`
- Automatic client initialization and caching

### Dynamic Model Switching
- Change models via CLI arguments or Streamlit dropdown
- Switch mid-conversation without restarting
- Client automatically reinitializes when model changes

### Structured Outputs
- **Native structured outputs** for Azure OpenAI/OpenAI using `beta.chat.completions.parse()`
- **JSON mode fallback** for other providers with Pydantic validation
- Automatic retry logic with configurable attempts

### Scenario Matching
- Predefined analytical scenarios in `scenarios.json`
- Classification with certainty scoring (1-10)
- Confirmation flow for medium-certainty matches
- Clarification requests for missing parameters

### SQL Query Generation
- Template-based SQL generation with parameter injection
- SQLite database execution
- Results formatted as pandas-compatible structures

### Natural Language Responses
- LLM generates business-friendly answers from SQL results
- Follows answer template style from scenarios
- Includes comparative analysis (vs previous year, etc.)

## Example Questions

The system supports various types of market performance queries:

- **Executive summaries**: "Provide executive summary of performance for HQ"
- **Category performance**: "Show HQ performance for Biscuits in July 2025"
- **Regional breakdowns**: "Provide executive summary of YTD performance by region"
- **Top/Bottom analysis**: "Show top 10 and bottom 10 gross profit growth country and category combinations"
- **Trend analysis**: "Show me historical category trend in Brazil Biscuits"
- **Forecast comparisons**: "What is the Chocolate sales volume performance for EU in June 2025 vs forecast?"

All available scenarios are documented in the **Scenarios** tab of the Streamlit interface.

## Data Information

- **Data Period**: January - September 2025
- **Data Type**: Artificial data generated for POC purposes
- **Metrics**: Net Revenue, Gross Profit, Operating Income, Volume
- **Dimensions**: Region, Country, Category, Sub-category, Brand
- **Source**: Available in Streamlit "Source Data" tab

## Troubleshooting

### "Failed to create LLM client"
**Cause**: Missing API key for the selected provider
**Solution**: Add the required API key to your `.env` file

### "Model not found in config"
**Cause**: Model not listed in `llm_config.yaml`
**Solution**: Add the model to the appropriate provider section in the config file

### Structured output errors with non-OpenAI providers
**Cause**: JSON fallback mode is less reliable than native structured outputs
**Solution**: Use Azure OpenAI or OpenAI for best structured output reliability

### Rate limit errors
**Cause**: Provider rate limits exceeded
**Solution**: Switch to a different provider or implement rate limiting in your code

## Development

### Adding a New LLM Provider

1. Add provider configuration to `llms/llm_clients.py` in `create_llm_client()`
2. Add models to `llm_config.yaml` under new provider section
3. Add API key environment variable to `.env`
4. Update `llms/structured_output.py` if provider has unique structured output capabilities
5. Test with `test_llm_migration.py`

### Adding a New Scenario

1. Define scenario in `scenarios.json` with:
   - `question_type`, `question_description`, `question_example`
   - `question_inputs` (required parameters with types and examples)
   - `question_sql_template` (SQL with `{{parameter}}` placeholders)
   - `answer_template` (example response format)
2. Test classification with various phrasings
3. Verify parameter extraction and SQL generation
