# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project using LangGraph for agent workflows and Streamlit for UI. The project uses `uv` for dependency management.

## Essential Commands

- **Run backend application through terminal**: `uv run flow.py`
- **Run Streamlit demo**: `streamlit run flow_frontend.py`
- **Add dependencies**: `uv add <package>`
- **Sync dependencies**: `uv sync`

## Python Version

Project requires Python >=3.10, <3.12 (specified in pyproject.toml:6)


This project is an assistant that takes users questions, matches best scenario, runs sql query for this scenario and returns an answer to the user. Questions and answears are in streamlit.

## Code Structure
- `flow_backend.py` bundles the shared conversation engine. It loads environment settings and scenarios, builds the LangGraph pipeline, and exposes helpers (`FlowBackend`, `ConversationTurnResult`) so any interface can process a user turn or initialize state without duplicating business logic.
- `flow_frontend.py` implements the Streamlit shell. It keeps state in `st.session_state`, renders the chat history and status sidebar, and forwards every user submission to the backend singleton (`get_backend()`), reacting to `awaiting_confirmation` and `awaiting_clarification` flags with tailored UI prompts.
- `scenario.json` stores scenarios of questions and answers.
- `flow.py` is the CLI wrapper that mirrors the original terminal experience. It creates a `FlowBackend` instance, runs the main input loop, and prints new `AIMessage` responses from the backend after each turn.



source data for scenarios is in file scenarios.json which is a list of dicts where each dict has a scenario - question and answer description in a following format:

[
    {
        "question_id": 1,
        "question_complexity": "...",
        "question_type": "...",
        "question_description": "...",
        "question_example": "...",
        "question_sql": "...",
        "answer_template": "...",
        "question_sql_template": "...",
        "question_inputs": [
            {
                "name": "...",
                "type": "...",
                "example": "...",
                "unique_values": [...]
            },
            {
                "name": "...",
                "type": "...",
                "example": "...",
                "unique_values": [...]
            },
            {
                "name": "...",
                "type": "...",
                "example": "...",
                "unique_values": [...]
            }
        ]
    }, 
  ...
]