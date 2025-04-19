# Implementation Plan: Milestones 5 & 6 - HITL Integration

This document outlines the detailed steps required to implement Milestones 5 (Agency Review HITL) and 6 (Customer Clarification HITL) as described in `Milestone 5-6.md`.

## Milestone 5: Implement Agency Review HITL

**Goal:** Modify the agent graph to introduce a pause point where a human agent can review the planner's intended tool execution (`planned_tool_inputs`). The agent should wait for approval before proceeding or loop back to the planner with feedback upon rejection.

**Detailed Tasks:**

1.  **State Verification (`src/ppa_agentic_v2/state.py`):**
    *   Confirm the presence and correct definition of `requires_agency_review: bool`, `human_feedback: Optional[Dict]`, and `planned_tool_inputs: Optional[Dict]` in the `AgentState` model. (These were likely added conceptually in M4's code but ensure they match the requirements).

2.  **Prompt Update V3 (`src/ppa_agentic_v2/prompts.py`):**
    *   Define `PLANNER_PROMPT_TEMPLATE_V3` incorporating sections for `human_feedback_str`.
    *   Instruct the LLM within the prompt to:
        *   Process `human_feedback` if present and adjust its plan accordingly.
        *   Decide if the *next* planned action requires review and set the `requires_review` flag (boolean) in its JSON output.
    *   Implement `format_planner_prompt` function (V3) to correctly format the state, including `human_feedback_str`, for this new template.

3.  **Planner Node Update (`src/ppa_agentic_v2/agent.py`):**
    *   Modify `planner_node` to:
        *   Use `format_planner_prompt` V3.
        *   Parse the LLM's JSON response, specifically extracting the `requires_review` boolean flag.
        *   Update the `AgentState` delta dictionary with `planned_tool_inputs`, `agent_scratchpad` (thought), and the parsed `requires_agency_review` flag.
        *   Crucially, clear `human_feedback` from the state update dictionary after it has been processed by the prompt formatter.
        *   Implement robust JSON parsing for the LLM response, potentially forcing review on parsing errors.
        *   Clear `last_tool_outputs` in the state update *before* returning the delta, as the planner has now consumed it.

4.  **Executor Node Wrapper (`src/ppa_agentic_v2/agent.py`):**
    *   Implement `executor_node_wrapper` function.
    *   This wrapper calls the underlying `ToolNode`.
    *   It handles potential errors during tool execution.
    *   It ensures the result is stored correctly in `last_tool_outputs` within the state update dictionary.
    *   It clears `planned_tool_inputs` after successful or failed execution.

5.  **Conditional Logic (`src/ppa_agentic_v2/agent.py`):**
    *   Implement the `check_agency_review` conditional edge function. This function checks `state.requires_agency_review` and `state.planned_tool_inputs` to decide the next step:
        *   `review_needed`: If review is true and a tool is planned.
        *   `execute_tool`: If review is false and a tool is planned.
        *   `END`: If no tool is planned.

6.  **Graph Modification (`src/ppa_agentic_v2/agent.py`):**
    *   Modify `build_agent_graph`:
        *   Import `InterruptForHumanApproval`.
        *   Define a constant for the review node name (e.g., `AGENCY_REVIEW_NODE_NAME = "agency_review_pause"`).
        *   Add the `executor_node_wrapper` node.
        *   Add the conditional edge from `planner` using `check_agency_review`, routing to `AGENCY_REVIEW_NODE_NAME`, `executor`, or `END`.
        *   Add an edge from `AGENCY_REVIEW_NODE_NAME` to `executor` (this path is taken upon resume/approval).
        *   Add the edge from `executor` back to `planner`.
        *   Compile the graph using `interrupt_before=[AGENCY_REVIEW_NODE_NAME]`.

7.  **Agent Runner Update (`src/ppa_agentic_v2/agent.py` / potentially `run_agent.py`):**
    *   Update the agent execution logic (e.g., in `PPAAgentRunner` or the run script):
        *   Use `app.stream` or `app.invoke` and check the graph state using `app.get_state`.
        *   Detect when `current_graph_state.next == (AGENCY_REVIEW_NODE_NAME,)`.
        *   Log the pause event.
        *   **(Initial Testing):** Simulate the review process:
            *   Create sample `HumanFeedback` (e.g., `{"approved": True}` or `{"approved": False, "comment": "..."}`).
            *   Update the paused state using `app.update_state` with the feedback.
            *   Resume execution using `app.stream/invoke` again with `None` as input.

**Testing Milestone 5:**
*   Trigger a planner decision that sets `requires_review=True`.
*   Verify the agent pauses.
*   Simulate approval: provide `{"approved": True}` feedback, resume, verify tool executes, and loops to planner.
*   Simulate rejection: provide `{"approved": False, "comment": "Try again"}` feedback, resume, verify it returns to the planner, and check that the planner prompt received the feedback.

## Milestone 6: Implement Customer Clarification HITL

**Goal:** Enable the agent to pause and request specific missing information from the customer using a dedicated tool (`ask_customer_tool`), then wait for the customer's reply before proceeding.

**Detailed Tasks:**

1.  **Tool Definition (`src/ppa_agentic_v2/tools.py`):**
    *   Define and implement the `ask_customer_tool`.
    *   This tool should:
        *   Accept `missing_fields: List[str]` as input.
        *   Use an LLM (potentially a simpler/cheaper one) with `GENERATE_INFO_REQUEST_PROMPT_TEMPLATE` (defined in `prompts.py`) to generate the clarification question text based on `missing_fields`.
        *   Return a structured dictionary, e.g., `{"status": "success", "message_content": "Generated question text...", "awaiting_customer_reply": True}`. *Crucially*, the tool itself indicates that the agent should now wait.

2.  **Prompt Update V4 (`src/ppa_agentic_v2/prompts.py`):**
    *   Define `PLANNER_PROMPT_TEMPLATE_V4` (building on V3).
    *   Instruct the LLM within the prompt:
        *   If information is missing and needed, plan to use `ask_customer_tool` with the specific `missing_fields`.
        *   Set a new flag `awaiting_customer_reply` (boolean) to `true` in its JSON output *only* when it plans to use `ask_customer_tool`.
    *   Implement `format_planner_prompt` function (V4).

3.  **Planner Node Update (`src/ppa_agentic_v2/agent.py`):**
    *   Modify `planner_node` to:
        *   Use `format_planner_prompt` V4.
        *   Parse the LLM's JSON response, extracting the `awaiting_customer_reply` boolean flag.
        *   Update the `AgentState` delta dictionary with this flag.

4.  **Conditional Logic (`src/ppa_agentic_v2/agent.py`):**
    *   Implement a new `check_customer_reply` conditional edge function. This function primarily checks if `state.awaiting_customer_reply` is true (set by the *output* of the `ask_customer_tool`).
        *   `wait_for_customer`: If `awaiting_customer_reply` is true.
        *   `continue_to_planner`: If `awaiting_customer_reply` is false.
    *   Modify `check_agency_review`: It should *not* route to END if `awaiting_customer_reply` is True after the planner runs (the planner might decide *not* to call a tool but *knows* it's waiting). This needs careful thought - maybe the check is simpler: Planner -> ReviewCheck -> (Interrupt | Executor). Executor -> CustomerWaitCheck -> (Interrupt | Planner).

5.  **Graph Modification (`src/ppa_agentic_v2/agent.py`):**
    *   Modify `build_agent_graph`:
        *   Define a constant for the customer wait node name (e.g., `CUSTOMER_WAIT_NODE_NAME = "customer_wait_pause"`).
        *   Modify the edge *from* `executor` node. Instead of going directly to `planner`, it should go to the new `check_customer_reply` conditional edge.
        *   Add conditional edges from `check_customer_reply`:
            *   `wait_for_customer` routes to `CUSTOMER_WAIT_NODE_NAME`.
            *   `continue_to_planner` routes to `planner`.
        *   Add an edge from `CUSTOMER_WAIT_NODE_NAME` back to `planner` (this path is taken upon resume after customer provides input).
        *   Compile the graph, adding `CUSTOMER_WAIT_NODE_NAME` to the `interrupt_before` list: `interrupt_before=[AGENCY_REVIEW_NODE_NAME, CUSTOMER_WAIT_NODE_NAME]`.

6.  **Agent Runner Update (`src/ppa_agentic_v2/agent.py` / `run_agent.py`):**
    *   Update the agent execution logic to also detect pauses for customer replies (`current_graph_state.next == (CUSTOMER_WAIT_NODE_NAME,)`).
    *   Log the pause event clearly indicating it's waiting for the customer.
    *   **(Testing):** When paused for the customer, simulate receiving the customer's reply (`HumanMessage`). Update the state using `app.update_state` (just adding the message). Resume execution with `None` input.

**Testing Milestone 6:**
*   Create a scenario where the planner should decide to use `ask_customer_tool`.
*   Verify the planner sets `awaiting_customer_reply=True`.
*   Verify `ask_customer_tool` executes and returns `awaiting_customer_reply=True` in its output.
*   Verify the agent pauses at the `CUSTOMER_WAIT_NODE_NAME` interrupt.
*   Simulate customer input: provide a `HumanMessage`, update state, resume.
*   Verify the agent loops back to the planner, and the planner now has the new customer message.

---

This plan provides a structured approach to implementing the HITL features. Each step involves modifying specific files and functions as outlined in the source markdown. Remember to test each milestone incrementally.
