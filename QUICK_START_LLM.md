# Quick Start: Using Different LLM Models

## Overview

The talk2data application now supports multiple LLM providers. You can switch between models easily using the CLI or Streamlit interface.

## Available Providers

- **Azure OpenAI** - Recommended for production
- **OpenAI** - Standard OpenAI API
- **Groq** - Fast inference
- **Google AI Studio (Gemini)** - Google's models
- **Databricks** - DBRX and Llama models
- **OpenRouter** - Access to multiple models
- **DeepSeek** - DeepSeek models
- **Lingaro** - Internal LLM gateway

## Setup

### 1. Configure API Keys

Add your API keys to `.env` file:

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

### 2. Check Available Models

See `llm_config.yaml` for all available models, or view them in the Streamlit sidebar.

## Usage

### CLI (Terminal)

```bash
# Use default model (gpt-4o)
python flow.py

# Specify a model explicitly
python flow.py azure_openai:gpt-4o-mini
python flow.py groq:llama-3.1-8b-instant
python flow.py google_ai_studio:gemini-2.0-flash

# Or use uv
uv run flow.py groq:llama-3.1-8b-instant
```

### Streamlit (Web UI)

```bash
# Start Streamlit
streamlit run flow_frontend.py
```

Then:
1. Look in the sidebar under "⚙️ Settings"
2. Select your preferred model from the dropdown
3. Start chatting - the selected model will be used
4. You can switch models anytime mid-conversation

### Programmatic

```python
from flow_backend import FlowBackend

# Create backend with specific model
backend = FlowBackend(llm_model_input="groq:llama-3.3-70b-versatile")

# Process conversations
state = backend.create_initial_state()
result = backend.run_conversation("Your query here", state)
```

## Model Format

Models can be specified in two ways:

1. **Short form**: `gpt-4o`
   - Uses the first provider that has this model in `llm_config.yaml`

2. **Full form**: `provider:model`
   - Example: `azure_openai:gpt-4o`
   - Example: `groq:llama-3.1-8b-instant`
   - Explicitly specifies which provider to use

## Recommended Models by Use Case

### Best Overall Performance
```bash
python flow.py azure_openai:gpt-4o
```

### Cost-Effective
```bash
python flow.py azure_openai:gpt-4o-mini
python flow.py groq:llama-3.1-8b-instant
```

### Fastest Inference
```bash
python flow.py groq:llama-3.3-70b-versatile
python flow.py google_ai_studio:gemini-2.0-flash
```

### Experimental/Advanced
```bash
python flow.py deepseek:deepseek-reasoner
python flow.py google_ai_studio:gemini-2.5-pro-exp-03-25
```

## Troubleshooting

### Issue: "Failed to create LLM client"
**Solution**: Check that the API key for that provider is in your `.env` file

### Issue: "Model not found in config"
**Solution**: Check `llm_config.yaml` - the model might not be listed

### Issue: Structured output errors with non-OpenAI providers
**Solution**: This is expected - some providers use JSON fallback which may be less reliable. Try Azure OpenAI or OpenAI for best results.

### Issue: Rate limit errors
**Solution**: Different providers have different rate limits. Switch to a different provider or wait a moment.

## Testing

Run the test suite to verify everything works:

```bash
python test_llm_migration.py
```

This will test:
- Backend initialization
- State management
- Client initialization
- Full conversation flow
- Model switching

## Getting Help

- Check `LLM_MIGRATION_SUMMARY.md` for detailed architecture info
- Look at `llms/` directory for implementation details
- Review `llm_config.yaml` for available models
- Check console logs - they show which model is being used

## Example Session

```bash
$ python flow.py groq:llama-3.1-8b-instant

[INFO] Using LLM model from command line: groq:llama-3.1-8b-instant

============================================================
MARKET QUERY ASSISTANT - Interactive Chat
Using LLM: groq:llama-3.1-8b-instant
============================================================

Welcome! Ask me about:
  • Provide executive summary of performance for HQ
  • Show HQ performance for the Biscuits in July 2025
  ...

💬 You: Show HQ performance for Biscuits in July 2025

[LLM CLIENT] Initializing client for: groq:llama-3.1-8b-instant
[CLASSIFIER] Type: total_performance_category_year_month, Certainty: 9/10
...
🤖 Assistant: Net Revenue reached 50.97 MM EUR, above last year...
```

Enjoy using multiple LLM providers with talk2data!
