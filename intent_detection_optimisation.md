Context
We are enhancing the classifier stage of the conversational flow to better detect user intent beyond pure scenario classification. Currently, the classifier uses semantic (TF‑IDF) scenario detection on the last user message and returns a question_type and certainty. This works for first‑turn questions but fails to distinguish between:
- A brand new question (“QUESTION”)
- A follow‑up to the previous topic (“FOLLOWUP”)
- A user consenting to save a Generic SQL scenario that the assistant proposed (“SAVING_SCENARIO”)

Goal
Improve intent detection and routing by combining the existing scenario detector with a lightweight LLM classifier over recent conversation turns when confidence is not high. This enables:
- First message: keep current fast path (no extra LLM call)
- Subsequent messages: if TF‑IDF certainty is high, keep current path; otherwise invoke LLM to classify as QUESTION | FOLLOWUP | SAVING_SCENARIO
- FOLLOWUP: keep current query_type from state and jump directly to parameter extraction
- SAVING_SCENARIO: route to a new adding_scenario_node that acknowledges saving
- QUESTION: carry on with mid certainty so confirmation/generic SQL can be used

Design Overview
1) Classifier node blending
   - Always run TF‑IDF detector on the latest user message.
   - If it’s the first user message → return as today (no LLM).
   - If not first message and certainty ≥ 8 → return as today.
   - If not first message and certainty < 8 → build a 3‑message window (latest 3 turns), stringify with backend._messages_to_string(), and ask the LLM to classify into QUESTION | FOLLOWUP | SAVING_SCENARIO (single token response).

2) Routing semantics
   - QUESTION → retain TF‑IDF chosen label and keep/massage certainty into mid (≥2 and <8) so the confirmation node is used and Generic SQL remains an option.
   - FOLLOWUP → do not change query_type (take existing from state), set certainty high (≥8) to route directly to parameter extraction.
   - SAVING_SCENARIO → set query_type='SAVING_SCENARIO' and route to a new node that acknowledges storing the scenario.

3) New node: adding_scenario_node
   - For now, only returns the message: "ok, i added it to scenario library".
   - In future, it can persist scenario templates to scenarios.json.

4) Graph & router changes
   - Add adding_scenario_node to the graph and export from nodes package.
   - Update route_after_classification to short‑circuit when query_type == 'SAVING_SCENARIO' and route to the new node.
   - Add an edge from adding_scenario_node to END.

5) Prompting strategy
   - Keep it deterministic and short. Instruct the LLM to reply with a single token from {QUESTION, FOLLOWUP, SAVING_SCENARIO}. Accept minor variations (e.g., "FOLLOW UP", "FOLLOW-UP").
   - Use state["llm_model_input"] (or default) via BasicAgent to avoid bespoke client handling inside nodes.

6) Thresholds & definitions
   - High certainty: ≥ 8 (unchanged).
   - Mid certainty: 2–7 (unchanged). If TF‑IDF < 2 but LLM says QUESTION, bump to 2 to enter confirmation.
   - Low certainty: < 2 (unchanged). If LLM says QUESTION we still ensure certainty ≥ 2 to allow confirmation; otherwise the low‑certainty fallback remains available when detector returns OTHER and LLM doesn’t override.

7) Edge cases
   - If previous query_type is missing on FOLLOWUP, fall back to detector’s label but keep high certainty to continue flow (best effort).
   - Robust parsing of LLM response to handle whitespace/case and common variants.

Implementation Plan
1. Create plan file (this document).
2. Update nodes/node_classifier.py:
   - Keep current TF‑IDF detection and parsing.
   - Detect first‑message by counting HumanMessages.
   - On low/mid certainty for non‑first message, call LLM with last 3 messages → classify into QUESTION | FOLLOWUP | SAVING_SCENARIO.
   - Implement branching as per routing semantics above.
3. Add nodes/node_adding_scenario.py with adding_scenario_node that returns the acknowledgement message.
4. Update nodes/__init__.py to export adding_scenario_node.
5. Update flow_backend.py:
   - Import adding_scenario_node.
   - Extend route_after_classification to handle query_type == 'SAVING_SCENARIO'.
   - Register the new node in _build_graph() and add edge to END.
6. Light manual validation
   - First message: behaves as before.
   - Multi‑turn: try a follow‑up like "make it for Europe only" → should reuse previous query_type and jump to parameter extraction.
   - After a Generic SQL answer, reply "yes, save it" → should route to adding_scenario_node and return the confirmation message.
7. Future work (not in this change):
   - Persist saved scenarios to scenarios.json with de‑duplication and validation.
   - Add tests for the new routing and classifier behavior.

