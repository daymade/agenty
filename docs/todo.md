# PPA New Business AI Agent - Step-by-Step Implementation Plan

**Objective:** Implement the LangGraph agent for PPA New Business PoC as described in `docs/system_design.md`.

**Target File:** `ppa_agent.py` (or similar)

---

## Phase 1: Setup and Foundation

1.  **[x] Imports:**
    *   Add `TypedDict`, `List`, `Dict`, `Optional` from `typing`.
    *   Add `StateGraph`, `END` from `langgraph.graph`.
    *   Add `ChatOpenAI` from `langchain_openai`.
    *   Add `ChatPromptTemplate` from `langchain_core.prompts`.
    *   Add `JsonOutputParser`, `StrOutputParser` from `langchain_core.output_parsers`.
    *   Add `json`.
    *   Add `os`.
    *   Add `logging`.

2.  **[X] Configuration & Constants:**
    *   Setup basic logging: `logging.basicConfig(...)`.
    *   Define `OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")` (or similar via config).
    *   Define `LLM_MODEL_NAME = "gpt-4-turbo"` (or similar via config).
    *   Define `PPA_REQUIREMENTS = [...]` (list of required fields).

3.  **[X] Agent State Definition:**
    *   Define the `AgentState` class using `TypedDict` as per Section 4 of `system_design.md`.

4.  **[X] LLM Initialization:**
    *   Instantiate `llm = ChatOpenAI(...)` (or provider like `GeminiProvider`).

## Phase 2: Implement Core Nodes

*(Each node function takes `state: AgentState` and returns a `dict` for conditional routing or the modified `state`)*

5.  **[X] Node: `identify_intention`**
    *   Define `identify_intention(state: AgentState) -> dict`.
    *   Create `ChatPromptTemplate` for intent classification (JSON output: `{"intent": ...}`).
    *   Build chain: `prompt | llm | JsonOutputParser()`.
    *   Invoke chain with `try-except` for parsing.
    *   Log intent, return result `dict`.

6.  **[X] Node: `identify_line_of_business` (Implemented as `identify_lob`)**
    *   Define `identify_line_of_business(state: AgentState) -> dict`.
    *   Create `ChatPromptTemplate` for LoB classification (JSON output: `{"lob": ...}`).
    *   Build chain: `prompt | llm | JsonOutputParser()`.
    *   Invoke chain with `try-except` for parsing.
    *   Log LoB, return result `dict`.

7.  **[X] Node: `analyze_information`**
    *   Define `analyze_information(state: AgentState) -> dict`.
    *   Set `state['ppa_requirements'] = PPA_REQUIREMENTS`.
    *   Create `ChatPromptTemplate` for information extraction based on `PPA_REQUIREMENTS` (JSON output).
    *   Build chain: `prompt | llm | JsonOutputParser()`.
    *   Invoke chain with `try-except` for parsing.
    *   Update `state['customer_info']`.
    *   Calculate and update `state['missing_info']`.
    *   Log extracted/missing info.
    *   Determine `status` (`info_incomplete` or `info_complete`).
    *   Return `{"status": status}`.

8.  **[X] Node: `generate_info_request`**
    *   Define `generate_info_request(state: AgentState) -> AgentState`.
    *   Create `ChatPromptTemplate` to generate email asking for `state['missing_info']`.
    *   Build chain: `prompt | llm | StrOutputParser()`.
    *   Invoke chain with `try-except` for LLM errors.
    *   Format message `dict` (`role`, `content`, `type`, `requires_review: True`).
    *   Append message to `state['messages']`.
    *   Set `state['requires_review'] = True`.
    *   Log generation.
    *   Return updated `state`.

9.  **[X] Node: `ask_for_discount_proof` (Implemented as `check_for_discounts`)**
    *   Define `ask_for_discount_proof(state: AgentState) -> AgentState`.
    *   Create `ChatPromptTemplate` to generate email asking about discount proofs.
    *   Build chain: `prompt | llm | StrOutputParser()`.
    *   Invoke chain with `try-except`.
    *   Format message `dict` (`type: discount_proof_request`, `requires_review: True`).
    *   Append message to `state['messages']`.
    *   Set `state['requires_review'] = True`.
    *   Log generation.
    *   Return updated `state`.

10. **[X] Node: `generate_quote` (Logic combined into `prepare_agency_review`)**
    *   Define `generate_quote(state: AgentState) -> AgentState`.
    *   Implement **mock** quote logic based on `state['customer_info']` and `state['proof_of_discount']`.
    *   Update `state['quote_data']` and `state['quote_ready'] = True`.
    *   Create `ChatPromptTemplate` to generate quote summary email.
    *   Build chain: `prompt | llm | StrOutputParser()`.
    *   Invoke chain with `try-except`.
    *   Format message `dict` (`type: quote`, `requires_review: True`).
    *   Append message to `state['messages']`.
    *   Set `state['requires_review'] = True`.
    *   Log generation.
    *   Return updated `state`.

## Phase 3: Implement HITL Node (Simulation)

11. **[X] Node: `agency_review` (Implemented as `prepare_agency_review`)**
    *   Define `agency_review(state: AgentState) -> AgentState`.
    *   Check `state['requires_review']`.
    *   If True, find the last message needing review.
    *   Simulate approval: Update message (`reviewed`, `requires_review`), update state (`requires_review = False`).
    *   Log simulation.
    *   Return updated `state`.

## Phase 4: Construct the Graph

12. **[X] Instantiate Graph:**
    *   `workflow = StateGraph(AgentState)`

13. **[X] Add Nodes:**
    *   Add all defined node functions to `workflow`.
    *   (Optional) Add placeholder for `process_customer_response`.

14. **[X] Set Entry Point:**
    *   `workflow.set_entry_point("identify_intention")`

15. **[X] Add Conditional Edges:**
    *   From `identify_intention` based on `intent` -> `identify_line_of_business` or `END`.
    *   From `identify_line_of_business` based on `lob` -> `analyze_information` or `END`.
    *   From `analyze_information` based on `status` -> `generate_info_request` or `ask_for_discount_proof`.

16. **[X] Add Direct & Conditional Edges for Review Flow:**
    *   `workflow.add_edge("generate_info_request", "agency_review")`
    *   `workflow.add_edge("ask_for_discount_proof", "agency_review")`
    *   `workflow.add_edge("generate_quote", "agency_review")`
    *   Add **conditional edge** from `agency_review`:
        *   Check type of last reviewed message.
        *   Map: If `ask_for_discount_proof` -> `generate_quote`, else -> `END`.

## Phase 5: Compile and Test

17. **[X] Compile Graph:**
    *   `app = workflow.compile()`

18. **[X] Prepare Test Case:**
    *   Create an `initial_state` dictionary with sample `customer_email` (Done in `visualize_agent.py`).

19. **[X] Run and Inspect:**
    *   Add `if __name__ == "__main__":` block.
    *   Invoke `app.invoke(initial_state)` (Done in `visualize_agent.py`).
    *   Print the final state using `json.dumps` (Done in `visualize_agent.py`).

## Phase 6: Refinements

20. **[~] Error Handling:** Review all nodes, ensure `try-except` blocks cover LLM calls and parsing.
21. **[X] Logging:** Ensure sufficient `logging.info` calls exist for tracing execution flow.
22. **[~] Docstrings:** Add docstrings to node functions.
23. **[X] Clarity:** Review code for style (PEP 8) and clarity.

## Phase 7: Multi-Turn Conversation Handling

24. **[X] **#24 Refine AgentState for Multi-Turn Context:** Enhance `AgentState` to store conversation history (`email_thread`) and a unique identifier (`thread_id`) for each conversation thread.
25. **[X] **#25 Modify `process_email` Node for Thread Management:** Update `PPAAgent.process_email` to accept a `thread_id`. Load existing state for that `thread_id` or create a new one. Append new emails to the `email_thread` in the state.
26. **[X] **#26 Adapt Nodes and Prompts for Conversation History:** Modify existing nodes and prompts to utilize the conversation history (`email_thread`) from the `AgentState` for context. Ensure prompts clearly instruct the LLM to consider past interactions.
27. **[X] **#27 Implement `process_customer_response` Node:** Create a new node specifically designed to handle incoming customer responses. This node should analyze the response in the context of the conversation history and update the `customer_info`.
28. **[ ] **#28 Modify Graph Routing for Multi-Turn Flow (Refine logic for waiting/question answering):** Update the `LangGraph` routing logic to handle multi-turn conversations.
  - [X] Integrate the `process_customer_response` node.
  - [ ] Implement logic to determine if the agent is waiting for a response or needs to ask clarifying questions based on the `customer_info` and `email_thread`.
  - [ ] Decide when to end the conversation (e.g., all required info gathered).
- [ ] **#29 Implement True State Persistence (Basic customer info persistence added, needs robust solution):** Implement a robust mechanism to persist `AgentState` (including `customer_info` and `email_thread`) between `process_email` calls for the *same* `thread_id`. This might involve saving/loading state to a file or database keyed by `thread_id`. (Basic `customer_info` saving/loading added, needs full `AgentState` persistence).
29. **[X] **#30 Update Tests for Multi-Turn Scenarios:** Create new integration tests that simulate multi-turn conversations, verifying that the agent correctly maintains context and updates state across turns.

## Phase 8: Advanced Features & Refinements

30. **[ ] Full State Persistence:** Implement robust state saving/loading between turns (e.g., using a database or file per thread) for all relevant state fields.
31. **[ ] Question Handling:** Implement a node and routing to explicitly handle customer questions identified by `process_customer_response`.
32. **[ ] Error Handling:** Improve error handling within nodes and graph routing.
33. **[ ] Configuration:** Centralize more configuration (log levels, API endpoints, etc.) in `config.py`.
34. **[ ] Prompt Management:** Move prompts from `nodes.py` to `prompts.py` for better organization.
35. **[ ] Code Quality:** Add linting (e.g., `ruff`, `mypy`) and formatting (`ruff format`).
36. **[ ] Documentation:** Expand README, add docstrings, explain project structure.
37. **[ ] Deployment:** Consider containerization (Dockerfile) and deployment options.

## Future Ideas

- Add support for attachments.
- Integrate with CRM/external systems.
- Implement policy change handling.
- Add more sophisticated discount logic.
- Fine-tune prompts based on real-world examples.
- Human-in-the-loop for review steps.

## Phase 8: Production Readiness & Deployment (Future)
