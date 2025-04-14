# PPA New Business AI Agent - Step-by-Step Implementation Plan

**Objective:** Implement the LangGraph agent for PPA New Business PoC as described in `docs/system_design.md`.

**Target File:** `ppa_agent.py` (or similar)

---

## Phase 1: Setup and Foundation

1.  **[ ] Imports:**
    *   Add `TypedDict`, `List`, `Dict`, `Optional` from `typing`.
    *   Add `StateGraph`, `END` from `langgraph.graph`.
    *   Add `ChatOpenAI` from `langchain_openai`.
    *   Add `ChatPromptTemplate` from `langchain_core.prompts`.
    *   Add `JsonOutputParser`, `StrOutputParser` from `langchain_core.output_parsers`.
    *   Add `json`.
    *   Add `os`.
    *   Add `logging`.

2.  **[ ] Configuration & Constants:**
    *   Setup basic logging: `logging.basicConfig(...)`.
    *   Define `OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")`.
    *   Define `LLM_MODEL_NAME = "gpt-4-turbo"`.
    *   Define `PPA_REQUIREMENTS = [...]` (list of required fields).

3.  **[ ] Agent State Definition:**
    *   Define the `AgentState` class using `TypedDict` as per Section 4 of `system_design.md`.

4.  **[ ] LLM Initialization:**
    *   Instantiate `llm = ChatOpenAI(...)`.

## Phase 2: Implement Core Nodes

*(Each node function takes `state: AgentState` and returns a `dict` for conditional routing or the modified `state`)*

5.  **[ ] Node: `identify_intention`**
    *   Define `identify_intention(state: AgentState) -> dict`.
    *   Create `ChatPromptTemplate` for intent classification (JSON output: `{"intent": ...}`).
    *   Build chain: `prompt | llm | JsonOutputParser()`.
    *   Invoke chain with `try-except` for parsing.
    *   Log intent, return result `dict`.

6.  **[ ] Node: `identify_line_of_business`**
    *   Define `identify_line_of_business(state: AgentState) -> dict`.
    *   Create `ChatPromptTemplate` for LoB classification (JSON output: `{"lob": ...}`).
    *   Build chain: `prompt | llm | JsonOutputParser()`.
    *   Invoke chain with `try-except` for parsing.
    *   Log LoB, return result `dict`.

7.  **[ ] Node: `analyze_information`**
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

8.  **[ ] Node: `generate_info_request`**
    *   Define `generate_info_request(state: AgentState) -> AgentState`.
    *   Create `ChatPromptTemplate` to generate email asking for `state['missing_info']`.
    *   Build chain: `prompt | llm | StrOutputParser()`.
    *   Invoke chain with `try-except` for LLM errors.
    *   Format message `dict` (`role`, `content`, `type`, `requires_review: True`).
    *   Append message to `state['messages']`.
    *   Set `state['requires_review'] = True`.
    *   Log generation.
    *   Return updated `state`.

9.  **[ ] Node: `ask_for_discount_proof`**
    *   Define `ask_for_discount_proof(state: AgentState) -> AgentState`.
    *   Create `ChatPromptTemplate` to generate email asking about discount proofs.
    *   Build chain: `prompt | llm | StrOutputParser()`.
    *   Invoke chain with `try-except`.
    *   Format message `dict` (`type: discount_proof_request`, `requires_review: True`).
    *   Append message to `state['messages']`.
    *   Set `state['requires_review'] = True`.
    *   Log generation.
    *   Return updated `state`.

10. **[ ] Node: `generate_quote`**
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

11. **[ ] Node: `agency_review`**
    *   Define `agency_review(state: AgentState) -> AgentState`.
    *   Check `state['requires_review']`.
    *   If True, find the last message needing review.
    *   Simulate approval: Update message (`reviewed`, `requires_review`), update state (`requires_review = False`).
    *   Log simulation.
    *   Return updated `state`.

## Phase 4: Construct the Graph

12. **[ ] Instantiate Graph:**
    *   `workflow = StateGraph(AgentState)`

13. **[ ] Add Nodes:**
    *   Add all defined node functions to `workflow`.
    *   (Optional) Add placeholder for `process_customer_response`.

14. **[ ] Set Entry Point:**
    *   `workflow.set_entry_point("identify_intention")`

15. **[ ] Add Conditional Edges:**
    *   From `identify_intention` based on `intent` -> `identify_line_of_business` or `END`.
    *   From `identify_line_of_business` based on `lob` -> `analyze_information` or `END`.
    *   From `analyze_information` based on `status` -> `generate_info_request` or `ask_for_discount_proof`.

16. **[ ] Add Direct & Conditional Edges for Review Flow:**
    *   `workflow.add_edge("generate_info_request", "agency_review")`
    *   `workflow.add_edge("ask_for_discount_proof", "agency_review")`
    *   `workflow.add_edge("generate_quote", "agency_review")`
    *   Add **conditional edge** from `agency_review`:
        *   Check type of last reviewed message.
        *   Map: If `ask_for_discount_proof` -> `generate_quote`, else -> `END`.

## Phase 5: Compile and Test

17. **[ ] Compile Graph:**
    *   `app = workflow.compile()`

18. **[ ] Prepare Test Case:**
    *   Create an `initial_state` dictionary with sample `customer_email`.

19. **[ ] Run and Inspect:**
    *   Add `if __name__ == "__main__":` block.
    *   Invoke `app.invoke(initial_state)`.
    *   Print the final state using `json.dumps`.

## Phase 6: Refinements

20. **[ ] Error Handling:** Review all nodes, ensure `try-except` blocks cover LLM calls and parsing.
21. **[ ] Logging:** Ensure sufficient `logging.info` calls exist for tracing execution flow.
22. **[ ] Docstrings:** Add docstrings to node functions.
23. **[ ] Clarity:** Review code for style (PEP 8) and clarity. 