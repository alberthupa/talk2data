# Generic SQL Medium Certainty Enhancement

**Status**: ✅ IMPLEMENTED (2025-10-22)
**Threshold Update**: Changed from 4-7 to 2-7 to allow more queries access to generic SQL

## Functional Background
- Current flow: a classifier routes medium-certainty results (score 2-7) to `confirmation_node`, which emits a yes/no prompt and sets `awaiting_confirmation`. The follow-up handling in `FlowBackend.run_conversation` only supports binary confirmation or denial before continuing to parameter extraction or reclassification.
- Goal: keep the classifier suggestion but offer a second track where the assistant attempts a best-effort SQL query without a pre-defined scenario, leveraging schema knowledge. The user should be able to opt in when the classifier is unsure, instead of being forced to confirm/deny.
- Expected user journey:
  1. Classifier returns medium certainty.
  2. Confirmation message presents two choices: (a) confirm the detected scenario; (b) let the agent craft a generic SQL query.
  3. User reply determines the path: confirmed scenario continues with parameter extraction; denial triggers reclassification; generic path invokes the new `node_generic_sql` flow.
  4. Generic flow still culminates in the existing `answering_node` so the UI always receives an interpreted answer and tabular results.

## Technical Implementation Plan

1. Conversation state extensions (`flow_backend.py`)
   1.1 Add fields to `ConversationState`: `confirmation_mode` (e.g., "scenario" vs "generic"), `awaiting_generic_choice` (bool), `generic_sql_attempted` (bool), and optionally `generic_sql_error` for surfacing failures.
   1.2 Initialize the new keys in `create_initial_state` and clear them alongside `awaiting_confirmation` and `awaiting_clarification` transitions.
   1.3 Ensure `FlowBackend.convert_messages_for_display` and any state copying logic tolerate the extra fields (no direct changes expected beyond initialization).

2. Confirmation prompt update (`nodes/node_confirmation.py`)
   2.1 Adjust the message to include both options, e.g., "I think you're asking about X. Reply '1' to confirm, '2' to let me craft a SQL query, or 'no' if this is wrong." Use clear option labels that can be parsed.
   2.2 Return additional state markers: set `awaiting_confirmation=True`, `confirmation_mode="scenario"`, and `awaiting_generic_choice=True` so downstream logic knows which branch is pending.

3. New generic SQL node (`nodes/node_generic_sql.py`)
   3.1 Create a module exposing `generic_sql_node(backend, state)`.
   3.2 Responsibilities:
       - Build a prompt using: (a) current conversation transcript (`backend._messages_to_string(state["messages"])`), (b) classifier guess (if any), (c) extracted params (may be `None`), and (d) a serialized schema of the SQLite DB.
       - Fetch schema via helper (see Step 4) returning table/column names, primary keys, and available metrics; cache in backend to avoid repeated PRAGMA calls.
       - Ask the LLM (likely via `backend._llm_agent.get_text_response_from_llm`) for a SQL query plus short rationale. Define a structured output schema or parsing strategy to safely separate SQL from explanation.
       - Execute the generated SQL using the same pattern as `node_sql` (with try/except, result packaging, and logging). On failure, capture the error in `generic_sql_error` and craft an AI message summarizing the issue, inviting user to rephrase or return to scenarios.
       - Return dict with `sql_query`, `sql_results`, optional `messages` (for rationale or error), `generic_sql_attempted=True`, and ensure `awaiting_confirmation=False`, `awaiting_generic_choice=False`.

4. Backend helpers for schema introspection (`flow_backend.py`)
   4.1 Add a private method like `_get_database_schema()` that caches table/column metadata in `self._cached_schema`.
   4.2 Use PRAGMA statements to retrieve table names, column names, types, and optionally sample values or foreign keys. Serialize into a concise plain-text summary suitable for the LLM prompt (limit size to avoid token bloat).
   4.3 Expose a utility on backend so `node_generic_sql` can call `backend.get_schema_prompt()` (wrapper that builds friendly text with caching and error handling).

5. `FlowBackend.run_conversation` branching
   5.1 When `awaiting_confirmation` is True:
       - Normalise user response (strip, lower).
       - Handle new options: detect "1", "yes", or synonyms → existing parameter extraction path.
       - Detect "2", "generic", or phrase like "write the query" → invoke `generic_sql_node_impl(self, state)` followed by `answering_node_impl` if the SQL execution succeeded (available rows or not). Append any messages from the generic node to state.
       - Maintain existing denial branch for "no"/"wrong".
       - For unrecognised input, re-prompt with both options.
   5.2 Ensure all paths reset `awaiting_confirmation`/`awaiting_generic_choice` appropriately to avoid repeated prompts.
   5.3 After a successful generic run, skip parameter extraction entirely and proceed straight to answering node so the user receives a narrative reply.

6. LangGraph registration
   6.1 Update `nodes/__init__.py` `__all__` export list to include `generic_sql_node`.
   6.2 In `_build_graph`, add `graph.add_node("generic_sql_node", partial(generic_sql_node_impl, self))`. Even if the node is only triggered manually from `run_conversation`, registering keeps the compiled graph consistent and allows future routing if needed.
   6.3 No new conditional edges are required immediately, but document in comments that medium-certainty routing is handled imperatively after confirmation.

7. Frontend adjustments (`flow_frontend.py`)
   7.1 Update status sidebar to communicate the new waiting state: if `awaiting_generic_choice` show something like "Reply 1 to confirm, 2 for generic SQL, or no to reclassify." Consider reusing the Streamlit info box.
   7.2 Optionally display `generic_sql_error` or rationale in the chat by ensuring messages returned from the generic node are appended before rendering.

8. Testing strategy
   8.1 Add unit/integration coverage in `test_llm_migration.py` (or create dedicated tests) to simulate medium-certainty classification and confirm:
       - Option "1" flows to parameter extraction as before.
       - Option "2" triggers generic node, populates `sql_query`, and calls answering node.
       - Option "no" re-runs classifier.
   8.2 Mock the LLM + SQLite to deterministic outputs; ensure schema helper caching is exercised.
   8.3 Verify regression: high-certainty and low-certainty paths remain unaffected.

9. Documentation updates
   9.1 Amend `README.md` conversation flow diagram and narrative to mention the new generic SQL fallback.
   9.2 Document how the generic path works and any limitations (e.g., may return empty results, relies on DB schema) in a suitable guide (potentially `main_idea.md` or a new troubleshooting section).

## Assumptions & Open Points
- The LLM used for generic SQL can handle schema up to current DB size; if token limits become an issue, investigate summarising schema or on-demand table descriptions.
- User inputs for options are assumed to be short textual commands; consider adding button-support in UI later but out of scope for backend plan.
- Existing answer templates may not perfectly fit generic queries; rely on `answering_node` to generate flexible narratives when template context is missing.
