# Talk2Data - Market Query Assistant

An AI-powered conversational interface that transforms natural language questions about Mondelez market performance into actionable business insights using LangGraph and multiple LLM providers.

## Overview

This project is a conversational assistant that takes user questions, matches them to predefined scenarios, executes SQL queries, and returns natural language answers. The system supports multiple LLM providers with dynamic model switching, and includes a flexible **Generic SQL** feature for ad-hoc queries that don't match predefined scenarios.

## Key Features

### 🤖 Multi-Provider LLM Support
- Switch between 8+ LLM providers without code changes
- Each provider configured with API keys in `.env`
- Automatic client initialization and caching
- Dynamic model switching mid-conversation

### 🎯 Intelligent Query Classification
- Predefined analytical scenarios in `scenarios.json`
- Classification with certainty scoring (1-10)
- Three-tier routing system:
  - **High certainty (≥8)**: Direct to parameter extraction
  - **Medium certainty (2-7)**: Confirmation with options
  - **Low certainty (<2)**: Clarification request
 - Non-first turn with low/mid certainty: lightweight LLM intent check over recent messages to detect one of:
   - **QUESTION** → keep detected scenario and proceed with mid certainty (to confirmation/generic SQL)
   - **FOLLOWUP** → keep previous scenario type and jump directly to parameter extraction
   - **SAVING_SCENARIO** → route to adding_scenario_node (acknowledge save)

### ⚡ Generic SQL Generation (NEW)
- **LLM-powered custom queries** when predefined scenarios don't match perfectly
- Available during medium-certainty confirmation flow (certainty 2-7)
- Uses database schema introspection to guide SQL generation
- Three-option confirmation:
  1. **Confirm** - Use the suggested predefined scenario
  2. **Generic SQL** - Generate custom SQL query on-the-fly
  3. **No** - Reject and reclassify
- Bypasses parameter extraction for ad-hoc queries
- Full error handling with user-friendly messages
 - After answering, the assistant asks whether to save the Q&A as a scenario; replying yes triggers the `SAVING_SCENARIO` path handled by `adding_scenario_node`

### 📊 Structured Outputs
- **Native structured outputs** for Azure OpenAI/OpenAI using `beta.chat.completions.parse()`
- **JSON mode fallback** for other providers with Pydantic validation
- Automatic retry logic with configurable attempts

### 💬 Natural Language Responses
- LLM generates business-friendly answers from SQL results
- Follows answer template style from scenarios
- Includes comparative analysis (vs previous year, etc.)

## Code Structure

### Core Application Files

- **`flow_backend.py`** - Shared conversation engine. Loads environment settings and scenarios, builds the LangGraph pipeline, exposes `FlowBackend` and `ConversationTurnResult` helpers. Contains database schema introspection and routing logic.

- **`flow_frontend.py`** - Streamlit web interface. Manages state in `st.session_state`, renders chat history and status sidebar, displays generic SQL options and errors.

- **`flow.py`** - CLI terminal interface. Creates `FlowBackend` instance, runs main input loop, prints AI responses. Supports command-line model selection.

- **`scenarios.json`** - Predefined question scenarios with types, descriptions, SQL templates, required parameters, and answer templates.

- **`scenario_detector.py`** - TF-IDF based scenario detection with optional pre-initialization for faster first-query response.

### LLM Infrastructure (`llms/` directory)

- **`llms/llm_clients.py`** - Multi-provider LLM client factory. Creates client instances for Azure OpenAI, OpenAI, Groq, Google AI Studio (Gemini), Databricks, OpenRouter, DeepSeek, and Lingaro.

- **`llms/basic_agent.py`** - Text response handler with retry logic. Manages LLM client lifecycle, handles provider-specific message formatting.

- **`llms/structured_output.py`** - Unified structured output module. Native structured outputs for Azure/OpenAI, JSON mode fallback for others.

### Nodes (`nodes/` directory)

LangGraph pipeline nodes that handle different stages of conversation:

- **`node_classifier.py`** - Classifies user queries into predefined question types with certainty scoring (1-10). Uses structured output and optional TF-IDF detection.

- **`node_parameter_extraction.py`** - Extracts required parameters (region, category, year-month, etc.) from conversation. Dynamically builds Pydantic models based on scenario requirements.

- **`node_confirmation.py`** - Asks user to confirm classification when certainty is medium (2-7). Offers three options: confirm scenario, use generic SQL, or reclassify.

- **`node_generic_sql.py`** - **NEW** - Generates custom SQL queries using LLM and database schema when user opts for generic path. Executes query and returns results, bypassing predefined scenarios.

- **`node_clarification.py`** - Requests missing parameters from the user with examples.

- **`node_sql.py`** - Generates SQL queries from templates by injecting extracted parameters, executes them against SQLite database, and returns results.

- **`node_answering.py`** - Generates natural language responses from SQL results using the LLM, following the answer template style from scenarios.

- **`node_low_certainty.py`** - Handles queries that don't match any scenario or have very low certainty (<2).
- **`node_adding_scenario.py`** - Handles the `SAVING_SCENARIO` intent by acknowledging scenario storage (future: persist to library).

### Configuration Files

- **`llm_config.yaml`** - Defines available LLM models organized by provider. Used to populate model selection dropdowns and validate model inputs.

- **`.env`** - Contains API keys and endpoints for all LLM providers (not in repo, see `.env.example`).

- **`pyproject.toml`** - Python project configuration with dependencies managed by `uv`.

### Data Files

- **`sql_data.db`** - SQLite database containing market performance data (artificial data for POC).

### Documentation & Testing

- **`LLM_MIGRATION_SUMMARY.md`** - Complete technical overview of the LLM migration architecture.

- **`QUICK_START_LLM.md`** - User-friendly quick start guide for using different LLM models.

- **`GENERIC_SQL_IMPLEMENTATION.md`** - Complete implementation details for the Generic SQL feature.

- **`test_llm_migration.py`** - Comprehensive test suite verifying backend initialization, state management, client initialization, conversation flow, model switching, and generic SQL state.

- **`CLAUDE.md`** - Instructions for Claude Code when working with this repository.

- **`generic_sql_update.md`** - Original technical plan for Generic SQL enhancement.

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
    |       +---> Medium certainty (2-7) ---> Confirmation
    |       |       |
    |       |       +---> Option 1: Confirm ---> Parameter Extraction
    |       |       |                                |
    |       |       |                                +---> SQL Node
    |       |       |
    |       |       +---> Option 2: Generic SQL ---> Generic SQL Node
    |       |       |                                   |
    |       |       |                                   +---> Answering Node (asks to save)
    |       |       |
    |       |       +---> Option 3: No ---> Reclassify
    |       |
    |       +---> Low certainty (<2) ---> Low Certainty Handler
    |       |
    |       +---> Non-first turn & low/mid certainty → LLM intent:
    |               - QUESTION → Confirmation
    |               - FOLLOWUP → Parameter Extraction (keep previous scenario)
    |               - SAVING_SCENARIO → Adding Scenario Node
    |
    +---> Parameter Extraction Node (extract required params)
    |       |
    |       +---> All params found ---> SQL Node
    |       |
    |       +---> Missing params ---> Clarification
    |
    +---> SQL Node (generate & execute template-based query)
    |       |
    |       +---> Answering Node
    |
    +---> Generic SQL Node (LLM-generated custom query)
    |       |
    |       +---> Answering Node (if successful)
    |
    +---> Adding Scenario Node (acknowledge saving)
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
          +---> confirmation_mode (e.g., "scenario")
          +---> awaiting_generic_choice (bool)
          +---> generic_sql_attempted (bool)
          +---> generic_sql_error (str | None)

Nodes communicate with LLM via:
  +---> Text responses: BasicAgent.get_text_response_from_llm()
  +---> Structured outputs: get_structured_response()
```

### Database Schema Introspection

```
FlowBackend._get_database_schema()
    |
    +---> PRAGMA table_info() for each table
    |
    +---> Cache in self._cached_schema
    |
    +---> Return formatted text summary
            |
            +---> Used by Generic SQL Node for prompt context
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
  - Show HQ performance for the Biscuits in July 2025
  ...

You: Show HQ performance for Biscuits in July 2025

[CLASSIFIER] Type: total_performance_category_year_month, Certainty: 9/10
Assistant: Net Revenue reached 50.97 MM EUR, above last year (50.03 MM EUR, +1.9%)...
```

**Example with Generic SQL:**

```
You: what product categories do i have

[CLASSIFIER] Type: What is sell-out by category and region?, Certainty: 3/10
[ROUTER] Mid certainty (3) → confirmationAssistant: I think you're asking about: What is sell-out by category and region?

Assistant: I think you're asking about: What is sell-out by category and region?

Please choose an option:
1. Confirm - Yes, that's correct
2. Generic SQL - Let me craft a custom SQL query for you
3. No - That's wrong, try again

Reply with '1', '2', or 'no'.

You: 2

[GENERIC SQL NODE] Starting generic SQL generation...
[GENERIC SQL NODE] Generated SQL: SELECT DISTINCT category FROM sales ORDER BY category
Assistant: Generated query: This query retrieves all unique product categories...
          
Results:
- Biscuits
- Chocolate  
- Gum & Candy
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
- **Generic SQL Support**: See confirmation options and error messages
- **Follow-up Buttons**: Context-aware quick actions
- **Scenarios Tab**: View all available query scenarios
- **Source Data Tab**: Access to source data documentation

**Model Selection:**
1. Open sidebar (Settings icon)
2. Choose model from "Select LLM Model" dropdown
3. Model switches immediately - works mid-conversation

**Using Generic SQL in Streamlit:**
- When medium certainty is detected, sidebar shows: "Reply with '1' to confirm, '2' for generic SQL, or 'no' to reclassify"
- Type "2" in the chat input to activate generic SQL
- Any errors are displayed in the sidebar with `st.error()`

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

### Scenario Detector Pre-Init (Optional)

- Purpose: Pre-fit the TF-IDF vectorizer for scenario detection at startup to avoid first-request latency.
- How to enable: pass `preinit_scenario_vectorizer=True` when constructing `FlowBackend`.

```python
from flow_backend import FlowBackend

# Pre-initialize TF-IDF vectorizer using scenarios.json at startup
backend = FlowBackend(preinit_scenario_vectorizer=True)
```

- Default behavior: If not enabled, the vectorizer initializes lazily on the first classification.
- Dependencies: Uses scikit-learn. If unavailable, detection gracefully falls back to `OTHER` with low certainty.
- Implementation: See `scenario_detector.py` for the detector and cache, and `nodes/node_classifier.py` for the thin caller.

## Using Generic SQL

### When to Use

The Generic SQL feature is available during medium-certainty classification (certainty score 2-7). Use it when:

- Your question is slightly different from predefined scenarios
- You want ad-hoc analysis not covered by templates
- The suggested scenario is close but not exactly what you need
- You want to explore data without parameter constraints

### Confirmation Flow

When the system has medium certainty (2-7), you'll see three options:

```
I think you're asking about: [scenario name]

Please choose an option:
1. Confirm - Yes, that's correct
2. Generic SQL - Let me craft a custom SQL query for you
3. No - That's wrong, try again

Reply with '1', '2', or 'no'.
```

### How It Works

**Option 2: Generic SQL** triggers the following process:

1. **Context Building**:
   - Analyzes conversation history
   - Considers classifier's suggested scenario
   - Reviews any extracted parameters

2. **Schema Introspection**:
   - Retrieves database schema (tables, columns, types)
   - Caches schema for performance
   - Includes schema in LLM prompt

3. **SQL Generation**:
   - LLM generates custom SQL query
   - Returns JSON with `sql` and `rationale` keys
   - Validates and parses response

4. **Execution**:
   - Executes query against SQLite database
   - Captures results or errors
   - Handles errors with user-friendly messages

5. **Response**:
   - If successful: passes results to answering node
   - Generates natural language summary
   - Returns to user with data table

### Example Workflow

```
User: Show me total revenue by brand for chocolate in Q2 2025
Assistant: I think you're asking about: category performance year month
           Please choose: 1) Confirm, 2) Generic SQL, 3) No

User: 2
Assistant: [Generates custom SQL based on schema and query]
           [Executes query and returns results with natural language summary]
```

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
uv run flow.py deepseek:deepseek-reasoner
uv run flow.py google_ai_studio:gemini-2.5-pro-exp-03-25
```

## Testing

Run the comprehensive test suite:

```bash
python test_llm_migration.py
```

**Tests verify:**
- ✓ Backend initialization with multiple models
- ✓ State creation with LLM fields
- ✓ Client initialization and caching
- ✓ Full conversation flow (classification → extraction → SQL → answer)
- ✓ Model switching mid-conversation
- ✓ Generic SQL state fields initialization

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
│   ├── node_generic_sql.py  # NEW: Generic SQL generation
│   ├── node_clarification.py
│   ├── node_low_certainty.py
│   ├── node_sql.py
│   └── node_answering.py
│
├── scenarios.json           # Question scenarios & SQL templates
├── llm_config.yaml          # Available LLM models by provider
├── sql_data.db              # SQLite database (market data)
├── data_description.md      # Complete dataset schema & KPI documentation
│
├── QUICK_START_LLM.md       # User quick start guide
├── CLAUDE.md                # Claude Code instructions
├── GENERIC_SQL_IMPLEMENTATION.md  # Generic SQL feature docs
├── generic_sql_update.md    # Technical implementation plan
├── test_llm_migration.py    # Test suite
├── .env                     # API keys (not in repo)
└── pyproject.toml           # Dependencies & config
```

## Example Questions

The system supports various types of market performance queries:

### Predefined Scenarios

- **Executive summaries**: "Provide executive summary of performance for HQ"
- **Category performance**: "Show HQ performance for Biscuits in July 2025"
- **Regional breakdowns**: "Provide executive summary of YTD performance by region"
- **Top/Bottom analysis**: "Show top 10 and bottom 10 gross profit growth country and category combinations"
- **Trend analysis**: "Show me historical category trend in Brazil Biscuits"
- **Forecast comparisons**: "What is the Chocolate sales volume performance for EU in June 2025 vs forecast?"

### Generic SQL Examples

- **Data exploration**: "what product categories do i have"
- **Custom aggregations**: "total revenue by brand for chocolate"
- **Ad-hoc queries**: "show me all countries with declining sales"
- **Schema queries**: "what metrics are available in the database"

All available scenarios are documented in the **Scenarios** tab of the Streamlit interface.

## Data Information

- **Data Period**: January - September 2025 (explicitly 2024-01 to 2024-09 and 2025-01 to 2025-09)
- **Data Type**: Synthetic beverages dataset for POC purposes covering Point-of-Customer (POC) analytics
- **Domain Coverage**: POC Sell-In, POC Sell-Out, Trade Inventory, Market Share, Promotions
- **Geography**: Mexico (MX) with region/city and Nielsen market IDs
- **Grain**: One row per period × customer_salesarea_sk (POC) × product_salesarea_sk
- **Metrics**: Net Revenue, Gross Profit, Operating Income, Volume, ASP, Returns Rate, Days of Supply, Market Share
- **Dimensions**: Region, Country, Category (CSD, JUICE, WATER), Product Flavour, Packaging, Distribution Channel
- **Detailed Schema**: See `data_description.md` for complete column definitions, KPIs, generation logic, and topic mapping
- **Source**: Available in Streamlit "Source Data" tab with full documentation

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

### Generic SQL errors
**Cause**: LLM generated invalid SQL or query failed  
**Solution**: 
- Check `generic_sql_error` in state for details
- Try rephrasing your question
- Use option 1 to confirm a predefined scenario instead
- Check database schema matches expectations

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

### Adjusting Certainty Thresholds

Current thresholds in `flow_backend.py:route_after_classification()`:
- High certainty: `>= 8`
- Medium certainty: `>= 2` (triggers confirmation with generic SQL option)
- Low certainty: `< 2`

To adjust:
```python
def route_after_classification(self, state: ConversationState):
    certainty = state.get("certainty", 1)
    
    if certainty >= 8:
        return "parameter_extraction_node"
    if certainty >= 2:  # Adjust this threshold
        return "confirmation_node"
    return "low_certainty_node"
```

## License

[Add your license here]

## Contributors

[Add contributors here]

## Acknowledgments

- Built with LangGraph for conversation flow management
- Supports multiple LLM providers for flexibility
- Uses SQLite for data storage
- Streamlit for web interface
- TF-IDF for scenario detection optimization
