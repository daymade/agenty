Okay, let's proceed with the detailed implementation, picking up from where Milestone 4 left off (real planner, core tools, persistence).

**Milestone 5: Implement Agency Review HITL**

This milestone integrates the first Human-in-the-Loop interrupt point, allowing an agency representative to review and approve/reject the agent's planned action before it's executed.

**Goal:** Modify the graph to pause before executing a tool if the planner flags it for review. Implement the logic for handling approval (continue execution) and rejection (loop back to planner with feedback).

**Code Implementation:**

1.  **`src/ppa_agent/state.py`**
    *   No changes are strictly needed here, as the required fields (`requires_agency_review`, `human_feedback`, `planned_tool_inputs`) were defined in Milestone 4's code example, preparing for this step. Ensure they are present as defined previously.

2.  **`src/ppa_agent/prompts.py`**
    *   Update the planner prompt template to V3, explicitly instructing the LLM to process `human_feedback` and decide on the `requires_review` flag for its *next* planned action.

    ```python
    # src/ppa_agent/prompts.py
    import json
    from typing import List, Any
    # Assuming AgentState is importable for type hint - use forward reference if needed
    # from typing import TYPE_CHECKING
    # if TYPE_CHECKING:
    #    from .state import AgentState

    # --- Planner Prompt V3 (Adds Human Feedback Handling & Review Decision) ---
    PLANNER_PROMPT_TEMPLATE_V3 = """
You are a PPA Insurance Quoting Agent assistant. Your primary goal is: {goal}.
You interact with the customer via messages and use internal tools to call Mercury Insurance APIs.

**Current Conversation State:**

*   **Conversation History (Newest Last):**
{messages_str}
*   **Current Customer Information Extracted:**
{customer_info_str}
*   **Mercury API Session Context (e.g., Quote ID):**
{mercury_session_str}
*   **Result of the VERY LAST Action Taken:**
{last_tool_outputs_str}
*   **Feedback from Last Human Review (if any):**
{human_feedback_str}

**Your Task:**

Based on the current state, conversation, AND HUMAN FEEDBACK (if provided), decide the single next best action.

1.  **Process Human Feedback:** If feedback exists (not 'None'), **you MUST address it**. Re-evaluate your previous plan based on the comment or use the edited inputs if provided. Do NOT repeat the rejected action without changes unless the feedback allows it.
2.  **Analyze State & Goal:** Review history, current data, goal, and last action result. What's the next logical step in the quoting sequence (e.g., Initiate -> Add Driver -> Add Vehicle -> Rate -> Summarize)? Handle errors from the last step.
3.  **Determine Next Step & Formulate Plan:**
    *   If ready to call an API or summarize: Choose the tool and generate exact `args`.
    *   If customer information is needed: Plan to use `ask_customer_tool` with specific `missing_fields`.
    *   If stuck or error unrecoverable: Plan `request_human_review_tool` (if available and necessary).
    *   If goal met: Plan `prepare_final_summary_tool` or indicate completion (`tool_name: null`).
4.  **Decide on Agency Review:** Set `requires_review` to `true` if the PLANNED action needs human review before execution (e.g., policy dictates review for `ask_customer_tool`, `rate_quote_tool`, or if you are uncertain). Otherwise, set `requires_review` to `false`.

**Available Tools:**

{tools_str}

**Output Format:**

Respond ONLY with a JSON object:
```json
{{
  "thought": "Detailed reasoning, including how you processed human feedback (if any) and why you chose this action and review requirement.",
  "requires_review": true | false, // Does THIS planned action need agency review?
  "tool_name": "Name of the tool to use OR null.",
  "args": {{ ... }} // Args for the tool OR null.
}}
```

**Example (Processing Rejection):**
```json
{{
  "thought": "The human agent rejected the previous plan to call 'add_vehicle_tool' with VIN '123', commenting 'VIN is incorrect'. I must ask the customer for the correct VIN instead.",
  "requires_review": true, // Asking the customer always requires review by policy
  "tool_name": "ask_customer_tool",
  "args": {{ "missing_fields": ["vehicle_vin"] }}
}}
```

Analyze the current state and provide your decision in the specified JSON format ONLY.
"""

    def format_planner_prompt(state: 'AgentState', tools: List[Any]) -> str:
        """Formats the prompt for the planner LLM (V3 - includes feedback)."""
        from .state import AgentState # Import locally for type hint if needed
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage # For type checking

        messages = state.messages or []
        # Format messages including role
        msg_parts = []
        for m in messages:
             role = "Unknown"
             if isinstance(m, HumanMessage): role = "Human"
             elif isinstance(m, AIMessage): role = "AI"
             elif isinstance(m, ToolMessage): role = f"Tool ({m.tool_call_id})" # Or just 'Tool Result'
             msg_parts.append(f"{role}: {m.content}")
        messages_str = "\n".join(msg_parts)
        if not messages_str: messages_str = "No messages yet."

        customer_info_str = json.dumps(state.customer_info or {}, indent=2, default=str)
        mercury_session_str = json.dumps(state.mercury_session or {}, indent=2, default=str)
        last_tool_outputs_str = json.dumps(state.last_tool_outputs or {"status": "N/A"}, indent=2, default=str)

        # Format human feedback clearly
        human_feedback_str = "None"
        if state.human_feedback:
            human_feedback_str = json.dumps(state.human_feedback, indent=2, default=str)
            if not state.human_feedback.get('approved'):
                 human_feedback_str = f"IMPORTANT: Last plan was REJECTED.\nFeedback: {human_feedback_str}"
            else:
                 human_feedback_str = f"Last plan was APPROVED.\nDetails: {human_feedback_str}"


        tools_str = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])

        prompt = PLANNER_PROMPT_TEMPLATE_V3.format(
            goal=state.goal,
            messages_str=messages_str,
            customer_info_str=customer_info_str,
            mercury_session_str=mercury_session_str,
            last_tool_outputs_str=last_tool_outputs_str,
            human_feedback_str=human_feedback_str, # <-- Include formatted feedback
            tools_str=tools_str
        )
        return prompt

    # Template for generating info request (needed by ask_customer_tool)
    GENERATE_INFO_REQUEST_PROMPT_TEMPLATE = """
    Generate a polite email asking the customer to provide these specific details: {missing_info_list}.
    Keep the email concise and clear. Respond ONLY with the body of the email.
    """
    ```

3.  **`src/ppa_agent/tools.py`**
    *   No changes needed in the tool definitions themselves for this milestone. Ensure they correctly return `{"status": "success", ...}` or `{"status": "error", ...}`.

4.  **`src/ppa_agent/agent.py`**
    *   Update `planner_node` to use the V3 prompt formatter and correctly parse the `requires_review` flag from the LLM response.
    *   Implement the `check_agency_review` conditional edge function.
    *   Modify `build_agent_graph` to incorporate the new conditional edge and the `InterruptForHumanApproval` node.
    *   Enhance `PPAAgentRunner` to detect the pause, simulate review, and resume.

    ```python
    # src/ppa_agent/agent.py
    import logging
    import json
    from typing import List, Dict, Any, Optional

    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.interrupt import InterruptForHumanApproval # <-- Import interrupt
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
    # Use BaseException for broader catch if needed for JSON parsing robustness
    from json import JSONDecodeError

    from .state import AgentState, HumanFeedback # Import updated state if needed
    from .tools import all_tools # Import the list of tools
    # Use V3 prompt formatter
    from .prompts import format_planner_prompt # <-- Use V3 formatter
    from .llm import get_llm_client
    from .config import SQLITE_DB_NAME, logger # Use configured logger

    # --- Agent Nodes ---

    def planner_node(state: AgentState) -> Dict[str, Any]:
        """
        Planner node (V3). Considers human feedback, decides action & review need.
        Returns ONLY the delta (changes) to the state.
        """
        logger.info("--- Entering Planner Node (V3) ---")

        # --- Prepare State Update Dictionary ---
        # Start with clearing previous cycle's transient state.
        # IMPORTANT: We return ONLY the changes. LangGraph merges these.
        # Crucially, clear human_feedback now that the planner is seeing it.
        update: Dict[str, Any] = {
            "planned_tool_inputs": None,
            "requires_agency_review": False,
            "awaiting_customer_reply": False, # Keep false unless specifically set
            "human_feedback": None, # Processed now, clear it for next state
            "agent_scratchpad": None # Will be set by LLM thought
        }
        # Note: We DO NOT clear last_tool_outputs here; the planner needs it.
        # It should be cleared *after* the executor runs or if planner decides no tool.

        # --- Prepare & Call LLM ---
        prompt = format_planner_prompt(state, all_tools) # Uses V3 template
        # logger.debug(f"Planner Prompt V3:\n------\n{prompt}\n------")
        llm = get_llm_client()
        logger.info("Invoking Planner LLM...")
        llm_decision = None
        try:
            ai_msg: AIMessage = llm.invoke(prompt)
            response_content = ai_msg.content
            logger.debug(f"Planner LLM Raw Response Content: {response_content}")

            # Robust JSON Parsing
            content_to_parse = response_content.strip()
            if content_to_parse.startswith("```json"):
                content_to_parse = content_to_parse[7:]
            if content_to_parse.endswith("```"):
                content_to_parse = content_to_parse[:-3]
            content_to_parse = content_to_parse.strip()
            if not content_to_parse:
                 raise ValueError("LLM returned empty content after stripping.")

            llm_decision = json.loads(content_to_parse)
            logger.info(f"Planner LLM Decided Action: {llm_decision}")

            # Validate expected keys might be good here
            if not isinstance(llm_decision, dict):
                 raise TypeError("LLM response is not a dictionary.")

        except (JSONDecodeError, ValueError, TypeError) as e:
             logger.error(f"Planner LLM response parsing error: {e}\nRaw Response:\n{response_content}", exc_info=True)
             update["agent_scratchpad"] = f"ERROR: LLM output parse error: {e}.\nRaw Response:\n{response_content}"
             # Decide error handling - plan no tool, maybe request review itself?
             llm_decision = {"tool_name": None, "args": None, "requires_review": True, "thought": f"Error processing LLM response: {e}. Requesting review."} # Force review on parse error
             update["requires_agency_review"] = True # Force review
        except Exception as e:
            logger.error(f"Unexpected error invoking planner LLM: {e}", exc_info=True)
            update["agent_scratchpad"] = f"ERROR: LLM invocation error: {e}"
            llm_decision = {"tool_name": None, "args": None, "requires_review": True, "thought": f"LLM invocation error: {e}. Requesting review."} # Force review
            update["requires_agency_review"] = True # Force review

        # --- Update State Based on LLM Decision ---
        # Clear previous tool output *before* planning the next input
        update["last_tool_outputs"] = None

        tool_name = llm_decision.get("tool_name")
        if tool_name:
            # TODO: Add validation: check if tool_name exists in all_tools?
            update["planned_tool_inputs"] = {
                "tool_name": tool_name,
                "args": llm_decision.get("args", {}) or {} # Ensure args is a dict
            }
            logger.info(f"Planner setting planned_tool_inputs: {update['planned_tool_inputs']}")
        else:
            # No tool planned (completion or handled error)
            logger.info("Planner decided no tool to execute.")
            # Keep planned_tool_inputs as None (already set in update init)

        # Set review flag based on LLM output
        update["requires_agency_review"] = llm_decision.get("requires_review", False)
        logger.info(f"Planner setting requires_agency_review: {update['requires_agency_review']}")

        # Update scratchpad (overwrite previous if needed)
        if "thought" in llm_decision:
            update["agent_scratchpad"] = llm_decision['thought']

        logger.info("--- Exiting Planner Node ---")
        # Return ONLY the changes
        return update

    # Use prebuilt ToolNode for execution
    execute_tool_node = ToolNode(all_tools)
    # We need to wrap the ToolNode to update last_tool_outputs correctly
    def executor_node_wrapper(state: AgentState) -> Dict[str, Any]:
         logger.info("--- Entering Executor Node ---")
         planned_inputs = state.planned_tool_inputs
         if not planned_inputs:
              logger.warning("Executor called without planned_tool_inputs. Skipping.")
              return {"last_tool_outputs": {"status": "skipped", "message": "No tool planned."}}

         logger.info(f"Executing Tool: {planned_inputs.get('tool_name')}")
         # logger.debug(f"Tool Args: {planned_inputs.get('args')}")

         # The ToolNode expects the tool invocation to be in the state,
         # LangGraph puts the result of the node into the state.
         # We need to map planned_tool_inputs -> ToolInvocation -> execute -> ToolOutput -> last_tool_outputs

         # Simulate passing invocation to ToolNode (actual mechanism might differ slightly)
         # In recent LangGraph versions, ToolNode might directly read a specific state key.
         # Let's assume execute_tool_node processes the state somehow.
         # The important part is the output format.
         try:
             # This simulates the execution and getting the raw output dict from the tool
             tool_output_dict = execute_tool_node.invoke(state) # This needs correct wiring in graph later
             # Ensure the output is structured as expected
             if not isinstance(tool_output_dict, dict) or 'status' not in tool_output_dict:
                  logger.error(f"Tool executed but returned unexpected format: {tool_output_dict}")
                  tool_output_dict = {"status": "error", "message": "Tool returned invalid format."}

         except Exception as e:
              logger.error(f"Error during tool execution: {e}", exc_info=True)
              tool_output_dict = {"status": "error", "message": f"Execution failed: {e}"}


         logger.info(f"Tool Execution Result: Status={tool_output_dict.get('status')}")
         logger.info("--- Exiting Executor Node ---")

         # Return the update dict: clear the plan, set the output
         return {
             "planned_tool_inputs": None, # Clear the executed plan
             "last_tool_outputs": tool_output_dict
             }

    # --- Conditional Edges ---

    def check_agency_review(state: AgentState) -> str:
        """Routes based on whether agency review is needed."""
        if state.planned_tool_inputs and state.requires_agency_review:
            logger.info("Routing to: agency_review_interrupt (Review Required)")
            return "review_needed"
        elif state.planned_tool_inputs:
             # Tool planned, but no review needed
            logger.info("Routing to: executor (No Review Required)")
            return "execute_tool"
        else:
             # No tool planned by planner (e.g., completion or error handled by planner)
             logger.info("Routing to: END (No Tool Planned)")
             return END


    # --- Graph Definition ---
    def build_agent_graph():
        """Builds the LangGraph StateGraph with Agency Review HITL."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("planner", planner_node)
        workflow.add_node("executor", executor_node_wrapper) # Use the wrapper
        # Define the interrupt node name used in compile()
        AGENCY_REVIEW_NODE_NAME = "agency_review_pause"
        # The interrupt node itself doesn't need custom logic beyond pausing

        workflow.set_entry_point("planner")

        # Planner -> Check Review -> (Interrupt | Executor)
        workflow.add_conditional_edges(
            "planner",
            check_agency_review,
            {
                "review_needed": AGENCY_REVIEW_NODE_NAME, # Route to the interrupt point
                "execute_tool": "executor",
                END: END
            }
        )

        # After review INTERRUPT (if approved), proceed to executor
        # LangGraph handles resuming *after* the interrupt point.
        # So the edge should go FROM the interrupt name TO the next step.
        workflow.add_edge(AGENCY_REVIEW_NODE_NAME, "executor")

        # After execution, always loop back to planner
        workflow.add_edge("executor", "planner")

        # Compile with checkpointer and interrupt config
        memory = SqliteSaver.sqlite(db_name=SQLITE_DB_NAME)
        app = workflow.compile(
            checkpointer=memory,
            # Interrupt *before* the node named AGENCY_REVIEW_NODE_NAME is entered
            interrupt_before=[AGENCY_REVIEW_NODE_NAME]
        )
        logger.info("Agent graph compiled (Milestone 5 - Agency Review HITL Added).")
        return app

    # --- Main Agent Class ---
    class PPAAgentRunner:
        def __init__(self):
            self.app = build_agent_graph()

        def run_turn(self, thread_id: str, user_input: Optional[str] = None) -> Optional[Dict[str, Any]]:
            """Runs agent turns, handling potential pauses for agency review."""
            logger.info(f"--- Running Agent Turn (Thread: {thread_id}) ---")
            if user_input: logger.info(f"Input: '{user_input}'")

            input_dict = {"messages": []}
            if user_input:
                input_dict["messages"].append(HumanMessage(content=user_input))

            config = {"configurable": {"thread_id": thread_id}}
            final_state_this_run = None
            paused = False

            try:
                # Stream returns intermediate states
                for step_output in self.app.stream(input_dict, config=config, stream_mode="values"):
                    # step_output is the full state dict after each node
                    final_state_this_run = step_output
                    step_keys = list(step_output.keys()) # Get keys to see which node ran last if needed
                    logger.info(f"Graph Step Completed. Current State Keys: {len(step_keys)}") # Simplified log

                    # Check if the *next* step is the interrupt node
                    current_graph_state = self.app.get_state(config)
                    if current_graph_state and current_graph_state.next == (AGENCY_REVIEW_NODE_NAME,):
                        logger.warning(f"--- Agent Paused for Agency Review (Thread: {thread_id}) ---")
                        paused = True
                        break # Exit the loop, state is saved by checkpointer

                if not paused:
                    logger.info(f"--- Agent Turn Finished Naturally (Thread: {thread_id}) ---")

                # Return the latest state achieved in this execution run
                # Fetch fresh state as stream might not yield *absolute* final if ended
                final_state_persisted = self.app.get_state(config)
                return final_state_persisted.values if final_state_persisted else None

            except Exception as e:
                 logger.error(f"Error during agent run (Thread: {thread_id}): {e}", exc_info=True)
                 # Attempt to return last known state before error
                 return final_state_this_run if final_state_this_run else {"error": str(e)}


        def update_state_and_resume(self, thread_id: str, update_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
             """Updates state for agency review feedback and resumes."""
             config = {"configurable": {"thread_id": thread_id}}
             logger.info(f"--- Resuming Agent (Thread: {thread_id}) after Agency Action ---")

             try:
                  # Update the state *before* resuming
                  # This includes setting human_feedback
                  logger.debug(f"Applying state update: {update_dict}")
                  self.app.update_state(config, update_dict)
                  logger.info("State updated successfully.")
             except Exception as e:
                   logger.error(f"Failed to update state for thread {thread_id}: {e}")
                   return {"error": f"Failed to update state: {e}"}

             # Resume execution by invoking stream with None input
             final_state_this_resume = None
             paused = False
             try:
                for step_output in self.app.stream(None, config=config, stream_mode="values"):
                    final_state_this_resume = step_output
                    step_keys = list(step_output.keys())
                    logger.info(f"Step after resume completed. State Keys: {len(step_keys)}")

                    # Check if paused AGAIN (e.g., review -> execute -> needs review again)
                    current_graph_state = self.app.get_state(config)
                    if current_graph_state and current_graph_state.next == (AGENCY_REVIEW_NODE_NAME,):
                         logger.warning(f"--- Agent Paused AGAIN for Agency Review (Thread: {thread_id}) ---")
                         paused = True
                         break

                if not paused:
                    logger.info(f"--- Agent Resume Finished Naturally (Thread: {thread_id}) ---")

                final_state_persisted = self.app.get_state(config)
                return final_state_persisted.values if final_state_persisted else None

             except Exception as e:
                 logger.error(f"Error during agent resume (Thread: {thread_id}): {e}", exc_info=True)
                 return final_state_this_resume if final_state_this_resume else {"error": str(e)}


        def get_current_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
             """Gets the current persisted state of a thread."""
             config = {"configurable": {"thread_id": thread_id}}
             try:
                 state = self.app.get_state(config)
                 return state.values if state else None
             except Exception as e:
                  logger.error(f"Error getting state for thread {thread_id}: {e}")
                  return None

        # resume_after_customer will be added in Milestone 6
    ```

5.  **`scripts/run_agent.py`**
    *   Update the simulation logic to correctly interact with the `PPAAgentRunner`, check for pauses, provide feedback, and resume.

    ```python
    # scripts/run_agent.py
    import sys
    import os
    import logging
    import json
    import time

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from src.ppa_agent.agent import PPAAgentRunner
    from src.ppa_agent.state import AgentState # For type hint
    from src.ppa_agent.config import logger, SQLITE_DB_NAME, PPA_QUOTE_REQUIREMENTS

    # --- Helper Functions ---
    def print_state_summary(state_dict: dict, title: str = "State Summary"):
        """Prints key parts of the state."""
        print(f"\n--- {title} ---")
        if not state_dict or isinstance(state_dict.get("error"), str):
            print(f"Error or empty state: {state_dict}")
            return

        print(f"Thread ID: {state_dict.get('thread_id')}")
        messages = state_dict.get('messages', [])
        print(f"Messages Count: {len(messages)}")
        # if messages: print(f"Last Message Type: {type(messages[-1]).__name__}")
        print(f"Customer Info: {json.dumps(state_dict.get('customer_info'), indent=2, default=str)}")
        print(f"Mercury Session: {state_dict.get('mercury_session')}")
        print(f"Planned Tool: {state_dict.get('planned_tool_inputs')}")
        print(f"Last Output: {state_dict.get('last_tool_outputs')}")
        print(f"Requires Review: {state_dict.get('requires_agency_review')}")
        print(f"Awaiting Customer: {state_dict.get('awaiting_customer_reply')}")
        print(f"Human Feedback: {state_dict.get('human_feedback')}") # Should usually be None here
        print(f"Scratchpad: {state_dict.get('agent_scratchpad')}")
        print("------")

    def check_if_paused_for_review(agent_runner: PPAAgentRunner, thread_id: str) -> bool:
         """Checks the graph state to see if it's paused for agency review."""
         config = {"configurable": {"thread_id": thread_id}}
         graph_state = agent_runner.app.get_state(config)
         return graph_state and graph_state.next == ("agency_review_pause",) # Use the named node

    def simulate_agency_review(agent_runner: PPAAgentRunner, thread_id: str) -> Optional[Dict[str, Any]]:
        """Simulates the human review process. Returns final state after resume."""
        print("\n--- !!! AGENCY REVIEW REQUIRED - SIMULATING !!! ---")
        current_state = agent_runner.get_current_state(thread_id)
        if not current_state:
            print("ERROR: Cannot get state for review.")
            return {"error": "Cannot get state for review."}

        planned_action = current_state.get('planned_tool_inputs')
        print(f"Reviewing plan for Thread: {thread_id}")
        print(f"Planned Action: {planned_action}")
        print(f"Planner Thought:\n{current_state.get('agent_scratchpad')}\n")

        if not hasattr(simulate_agency_review, "rejected_tools"):
             simulate_agency_review.rejected_tools = {} # Track rejections per tool per thread

        tool_name = planned_action.get("tool_name") if planned_action else None
        thread_tool_key = f"{thread_id}_{tool_name}"

        # Simulation Policy:
        # - Always approve ask_customer
        # - Reject rate_quote the first time it's planned
        # - Approve all others
        approved = True
        feedback_comment = "Approved by simulation."
        edited_inputs = None # No edits in this simulation

        if tool_name == "ask_customer_tool":
            print("SIMULATION: Auto-approving 'ask_customer_tool'.")
            approved = True
        elif tool_name == "rate_quote_tool":
            if thread_tool_key not in simulate_agency_review.rejected_tools:
                 print("SIMULATION: REJECTING 'rate_quote_tool' (first attempt).")
                 approved = False
                 feedback_comment = "Simulated rejection: Please verify all data before rating."
                 simulate_agency_review.rejected_tools[thread_tool_key] = True # Mark as rejected once
            else:
                 print("SIMULATION: Approving 'rate_quote_tool' (subsequent attempt).")
                 approved = True
        else:
            print(f"SIMULATION: Auto-approving tool '{tool_name}'.")
            approved = True

        # Prepare state update for resume
        human_feedback_payload = {
            "approved": approved,
            "comment": feedback_comment,
            "edited_inputs": edited_inputs # Could allow edits here
        }
        update_for_resume = {"human_feedback": human_feedback_payload}

        if not approved:
             # If rejecting, clear the plan so planner must replan
             update_for_resume["planned_tool_inputs"] = None
             logger.info("Clearing planned_tool_inputs due to rejection.")

        print(f"Simulated Feedback: {human_feedback_payload}")
        print("-----------------------------------------")

        # Resume the agent with the feedback
        resumed_state = agent_runner.update_state_and_resume(thread_id, update_for_resume)
        return resumed_state


    # --- Main Execution ---
    if __name__ == "__main__":
        logger.info("--- Milestone 5: Agency Review HITL Test ---")

        # Clean slate
        if os.path.exists(SQLITE_DB_NAME):
            logger.warning(f"Deleting existing database: {SQLITE_DB_NAME}")
            os.remove(SQLITE_DB_NAME)

        agent_runner = PPAAgentRunner()
        thread_id = "hitl_review_test_m5_001"

        # Turn 1: Should plan quote_initiate, potentially trigger review
        print(f"\n--- Turn 1 START (Thread: {thread_id}) ---")
        turn1_input = "Quote needed for John Doe, DOB 1990-05-15, 1 Main St, Anytown, CA 90210."
        current_state = agent_runner.run_turn(thread_id, turn1_input) # Run until end or pause
        print_state_summary(current_state, "State After Turn 1 Initial Run")

        # Loop to handle potential review pauses within the turn
        while check_if_paused_for_review(agent_runner, thread_id):
             print("\n--- Turn 1 PAUSED for Agency Review ---")
             current_state = simulate_agency_review(agent_runner, thread_id) # Simulate and resume
             print_state_summary(current_state, "State After Agency Review Resume")
             # Check again in case it paused immediately after resuming

        print("\n--- Turn 1 Fully Complete (No more agency pauses) ---")
        final_t1_state = agent_runner.get_current_state(thread_id)
        print_state_summary(final_t1_state, "Final State Turn 1")

        # Turn 2: Should plan ask_customer, trigger review
        print(f"\n--- Turn 2 START (Thread: {thread_id}) ---")
        # No new input, let planner continue based on state after quote_initiate
        current_state_t2 = agent_runner.run_turn(thread_id)
        print_state_summary(current_state_t2, "State After Turn 2 Initial Run")

        while check_if_paused_for_review(agent_runner, thread_id):
             print("\n--- Turn 2 PAUSED for Agency Review ---")
             current_state_t2 = simulate_agency_review(agent_runner, thread_id)
             print_state_summary(current_state_t2, "State After Agency Review Resume")

        print("\n--- Turn 2 Fully Complete (No more agency pauses) ---")
        final_t2_state = agent_runner.get_current_state(thread_id)
        print_state_summary(final_t2_state, "Final State Turn 2")
        # Check if it paused for customer reply (Milestone 6 check)
        # paused_for_customer = final_t2_state and final_t2_state.get('awaiting_customer_reply')
        # print(f"\nPaused waiting for customer? {paused_for_customer}")

        print("\nMilestone 5 Test Complete.")
    ```

**To Run Milestone 5:**

1.  Ensure the V3 prompt template is in `prompts.py`.
2.  Ensure the updated `agent.py` (planner, graph, runner) is saved.
3.  Ensure the simulation script `run_agent.py` is updated.
4.  Run `python scripts/run_agent.py`.
5.  **Observe:**
    *   The planner should now include `"requires_review": true` in its JSON output for certain tools (like the first API call if you add that policy, or `ask_customer_tool`).
    *   The graph should pause ("Agent Paused for Agency Review").
    *   The simulation script will detect the pause, provide simulated feedback (potentially rejecting once), and call `update_state_and_resume`.
    *   If rejected, the planner node runs again, processes the feedback from the state, and makes a *new* plan.
    *   The process continues until the turn finishes naturally or hits the (not yet implemented) customer wait pause.

This milestone confirms the agency review loop functions correctly.

---

**Milestone 6: Implement "Wait for Customer Reply" HITL & Complete Tools**

**Goal:** Implement the second interrupt to pause after `ask_customer` and wait for external resumption. Implement the remaining Mercury API tools (or realistic mocks).

1.  **`src/ppa_agent/state.py`**
    *   Ensure `awaiting_customer_reply: bool` field is present.

2.  **`src/ppa_agent/tools.py`**
    *   Implement *all* required Mercury API tools (`add_vehicle_tool`, `update_driver_tool`, `rate_quote_tool`, etc.) using the same pattern as `quote_initiate_tool` (either real calls or realistic mocks). Ensure they have accurate docstrings and schemas.
    *   Update the `all_tools` list to include them.

    ```python
    # src/ppa_agent/tools.py
    # ... (add implementations/mocks for all tools from M4/M5) ...
    # Make sure rate_quote_tool description mentions needing review

    # --- Tool Registry (Complete List) ---
    all_tools = [
        quote_initiate_tool,
        ask_customer_tool,
        add_vehicle_tool, # Implemented in M6
        update_driver_tool, # Implemented in M6
        # quote_prefill_tool, # Add implementations/mocks
        # get_driver_tool,
        # update_coverage_tool,
        # assign_vehicle_tool,
        rate_quote_tool, # Implemented in M6
        prepare_final_summary_tool, # Implemented in M6
    ]
    ```

3.  **`src/ppa_agent/prompts.py`**
    *   Update to `PLANNER_PROMPT_TEMPLATE_V4` which includes the `await_customer` field in the expected JSON output. Ensure the instructions clearly state *when* to set this flag (only if the plan is `ask_customer_tool`).

    ```python
    # src/ppa_agent/prompts.py
    # Replace the V3 template content with the V4 template content provided previously.
    # Ensure the format_planner_prompt function uses PLANNER_PROMPT_TEMPLATE_V4.

    # ... (rest of the file) ...
    PLANNER_PROMPT_TEMPLATE_V4 = """
    # [Copy the full V4 prompt content from the previous response here]
    # ... ensure it includes the 'await_customer' instruction and output field ...
    """

    def format_planner_prompt(state: 'AgentState', tools: List[Any]) -> str:
        # ... (logic is the same, just uses the V4 template string) ...
        # ... (ensure human_feedback_str formatting is correct) ...
        prompt = PLANNER_PROMPT_TEMPLATE_V4.format( # <-- Use V4
            # ... all fields ...
        )
        return prompt

    # ... (GENERATE_INFO_REQUEST_PROMPT_TEMPLATE remains the same) ...
    ```

4.  **`src/ppa_agent/agent.py`**
    *   Update `planner_node` to parse the `await_customer` flag from the LLM JSON output and include it in the state update dictionary.
    *   Update `check_execution_result` conditional edge logic to check `state.awaiting_customer_reply` *after* `ask_customer_tool` succeeds, and route to the new "wait_customer" interrupt point.
    *   Modify `build_agent_graph` to add the `wait_customer` node name and include it in the `interrupt_before` list during compilation. Wire the `check_execution_result` edge to this new interrupt point.
    *   Add the `resume_after_customer` method to `PPAAgentRunner`.

    ```python
    # src/ppa_agent/agent.py
    # ... (imports) ...
    from .prompts import format_planner_prompt # Use V4

    # ... (AgentState, tools, execute_tool_node) ...
    logger = logging.getLogger(__name__)

    def planner_node(state: AgentState) -> Dict[str, Any]:
        """
        Planner node (V4). Considers feedback, decides action, review, AND wait need.
        """
        logger.info("--- Entering Planner Node (V4) ---")
        # --- Prepare State Update Dictionary ---
        update: Dict[str, Any] = {
            "planned_tool_inputs": None,
            "last_tool_outputs": state.last_tool_outputs, # Keep last output
            "requires_agency_review": False,
            "awaiting_customer_reply": False, # <-- Planner will set this
            "human_feedback": None,
            "agent_scratchpad": None
        }

        # --- Prepare & Call LLM (using V4 prompt) ---
        prompt = format_planner_prompt(state, all_tools)
        llm = get_llm_client()
        logger.info("Invoking Planner LLM...")
        llm_decision = None
        try:
            # ... (LLM call and JSON parsing as in M5) ...
            ai_msg = llm.invoke(prompt)
            response_content = ai_msg.content
            # ... (Robust JSON parsing) ...
            content_to_parse = response_content.strip()
            if content_to_parse.startswith("```json"): content_to_parse = content_to_parse[7:]
            if content_to_parse.endswith("```"): content_to_parse = content_to_parse[:-3]
            llm_decision = json.loads(content_to_parse.strip())
            logger.info(f"Planner LLM Decided Action: {llm_decision}")

        except Exception as e:
             # ... (Error handling as in M5, ensure llm_decision includes await_customer) ...
             logger.error(f"Planner LLM Error: {e}", exc_info=True)
             llm_decision = {"tool_name": None, "args": None, "requires_review": True, "await_customer": False, "thought": f"Error processing LLM response: {e}"}
             update["requires_agency_review"] = True


        # --- Update State Based on LLM Decision (including wait flag) ---
        update["last_tool_outputs"] = None # Clear last output *before* setting new plan

        tool_name = llm_decision.get("tool_name") if llm_decision else None
        if tool_name:
            update["planned_tool_inputs"] = {
                "tool_name": tool_name,
                "args": llm_decision.get("args", {}) or {}
            }
            logger.info(f"Planner setting planned_tool_inputs: {update['planned_tool_inputs']}")
        # else: planned_tool_inputs remains None

        update["requires_agency_review"] = llm_decision.get("requires_review", False) if llm_decision else False
        # --> Set the await flag based on planner output <--
        update["awaiting_customer_reply"] = llm_decision.get("await_customer", False) if llm_decision else False
        logger.info(f"Planner setting requires_agency_review: {update['requires_agency_review']}")
        logger.info(f"Planner setting awaiting_customer_reply: {update['awaiting_customer_reply']}") # <-- Log wait flag

        if llm_decision and "thought" in llm_decision:
            update["agent_scratchpad"] = llm_decision['thought']

        logger.info("--- Exiting Planner Node ---")
        return update

    # --- Executor Wrapper (No changes needed) ---
    def executor_node_wrapper(state: AgentState) -> Dict[str, Any]:
         # ... (Same as M5) ...
         logger.info("--- Entering Executor Node ---")
         planned_inputs = state.planned_tool_inputs
         if not planned_inputs:
              logger.warning("Executor called without planned_tool_inputs. Skipping.")
              # Return the state change explicitly clearing planned_inputs if necessary
              return {"planned_tool_inputs": None, "last_tool_outputs": {"status": "skipped", "message": "No tool planned."}}

         logger.info(f"Executing Tool: {planned_inputs.get('tool_name')}")
         tool_output_dict = {}
         try:
             # Invoke ToolNode - assuming it reads relevant parts of state or gets args passed correctly
             # Note: ToolNode logic might need adaptation based on how it receives args from the state.
             # It might expect args directly under a specific key if not using `invoke_tool` directly.
             # For simplicity, we'll assume it works via state inspection or graph wiring.
             # Simulate invocation result:
             tool_name_to_run = planned_inputs["tool_name"]
             tool_args = planned_inputs["args"]
             selected_tool = next((t for t in all_tools if t.name == tool_name_to_run), None)
             if selected_tool:
                  tool_output_dict = selected_tool.invoke(tool_args) # Direct invoke for simulation
             else:
                  raise ValueError(f"Tool '{tool_name_to_run}' not found in registry.")

             if not isinstance(tool_output_dict, dict) or 'status' not in tool_output_dict:
                  logger.error(f"Tool executed but returned unexpected format: {tool_output_dict}")
                  tool_output_dict = {"status": "error", "message": "Tool returned invalid format."}
         except Exception as e:
              logger.error(f"Error during tool execution: {e}", exc_info=True)
              tool_output_dict = {"status": "error", "message": f"Execution failed: {e}"}

         logger.info(f"Tool Execution Result: Status={tool_output_dict.get('status')}")
         logger.info("--- Exiting Executor Node ---")
         return {
             "planned_tool_inputs": None, # Clear the executed plan
             "last_tool_outputs": tool_output_dict
             }


    # --- Conditional Edges (Updated for Customer Wait) ---
    # check_agency_review remains the same as M5

    def check_execution_result(state: AgentState) -> str:
        """Routes after tool execution. Checks for ask_customer success + wait flag."""
        last_output = state.last_tool_outputs or {}
        status = last_output.get("status")
        message_type = last_output.get("message_type") # Specific to ask_customer_tool output

        logger.info(f"Checking execution result. Status={status}, MessageType={message_type}, AwaitingFlag={state.awaiting_customer_reply}")

        # ---> Check if ask_customer succeeded AND planner wants to wait <---
        if state.awaiting_customer_reply and message_type == "info_request" and status == "success":
            logger.info("Routing to: wait_customer_interrupt (Ask Customer succeeded & wait requested)")
            # We clear the awaiting flag *when entering* the wait state, not here.
            # Or maybe clear it here before routing to wait? Let's clear in planner next time.
            return "wait_for_customer"
        elif status == "error":
            logger.info("Routing to: planner (Tool execution failed)")
            return "plan_next"
        elif status == "success":
            if last_output.get("summary_complete"):
                 logger.info("Routing to: END (Final Summary Complete)")
                 return END
            logger.info("Routing to: planner (Tool succeeded, continue flow)")
            return "plan_next"
        else: # No status or unknown
            logger.warning(f"Unknown tool execution status: {status}. Routing to planner.")
            return "plan_next"


    # --- Graph Definition (Updated for Both Interrupts) ---
    def build_agent_graph():
        """Builds the LangGraph StateGraph with Both HITL Interrupts."""
        workflow = StateGraph(AgentState)

        # Node names for interrupts
        AGENCY_REVIEW_NODE_NAME = "agency_review_pause"
        WAIT_CUSTOMER_NODE_NAME = "wait_customer_pause"

        # Add nodes
        workflow.add_node("planner", planner_node)
        workflow.add_node("executor", executor_node_wrapper) # Use the wrapper
        # Add placeholder nodes for interrupt targets if needed for explicit edge connection
        # These nodes don't do anything themselves, the pause happens *before* them
        workflow.add_node(AGENCY_REVIEW_NODE_NAME, lambda state: state)
        workflow.add_node(WAIT_CUSTOMER_NODE_NAME, lambda state: state)

        workflow.set_entry_point("planner")

        # Planner -> Check Review -> (Interrupt | Check Action)
        workflow.add_conditional_edges(
            "planner",
            check_agency_review,
            {
                "review_needed": AGENCY_REVIEW_NODE_NAME,
                # If no review needed, immediately check if we execute or wait/end
                "execute_tool": "executor", # Route directly if review not needed & tool planned
                 END: END # Route directly if planner decided to end
            }
        )

        # After AGNECY review INTERRUPT (if approved externally), proceed to executor
        # The external resume directs flow here.
        workflow.add_edge(AGENCY_REVIEW_NODE_NAME, "executor")

        # After Executor -> Check Result -> (Planner | Wait Interrupt | End)
        workflow.add_conditional_edges(
            "executor",
            check_execution_result,
            {
                "plan_next": "planner",
                "wait_for_customer": WAIT_CUSTOMER_NODE_NAME, # Route to the wait interrupt point
                END: END
            }
        )

        # After WAIT CUSTOMER interrupt (resumed externally), loop back to planner
        workflow.add_edge(WAIT_CUSTOMER_NODE_NAME, "planner")


        # Compile with checkpointer and BOTH interrupt points
        memory = SqliteSaver.sqlite(db_name=SQLITE_DB_NAME)
        app = workflow.compile(
            checkpointer=memory,
            interrupt_before=[AGENCY_REVIEW_NODE_NAME, WAIT_CUSTOMER_NODE_NAME] # Pause before these nodes
        )
        logger.info("Agent graph compiled (Milestone 6 - Both HITLs Added).")
        return app

    # --- Main Agent Class ---
    class PPAAgentRunner:
        # ... (__init__ is same - calls build_agent_graph) ...
        def __init__(self):
            self.app = build_agent_graph()

        def run_turn(self, thread_id: str, user_input: Optional[str] = None) -> Optional[Dict[str, Any]]:
            """Runs agent turns, handling pauses for review or customer reply."""
            logger.info(f"--- Running Agent Turn (Thread: {thread_id}) ---")
            if user_input: logger.info(f"Input: '{user_input}'")

            input_dict = {"messages": []}
            if user_input:
                input_dict["messages"].append(HumanMessage(content=user_input))

            config = {"configurable": {"thread_id": thread_id}}
            final_state_this_run = None
            paused_at = None

            try:
                for step_output in self.app.stream(input_dict, config=config, stream_mode="values"):
                    final_state_this_run = step_output
                    step_keys = list(step_output.keys())
                    logger.info(f"Graph Step Completed. State Keys: {len(step_keys)}")

                    current_graph_state = self.app.get_state(config)
                    if current_graph_state and current_graph_state.next:
                         next_nodes = current_graph_state.next
                         # Check against the *actual* node names used in compile()
                         if AGENCY_REVIEW_NODE_NAME in next_nodes:
                             logger.warning(f"--- Agent Paused for Agency Review (Thread: {thread_id}) ---")
                             paused_at = "agency_review"
                             break
                         elif WAIT_CUSTOMER_NODE_NAME in next_nodes:
                             logger.warning(f"--- Agent Paused Waiting for Customer (Thread: {thread_id}) ---")
                             paused_at = "wait_customer"
                             break

                if not paused_at:
                    logger.info(f"--- Agent Turn Finished Naturally (Thread: {thread_id}) ---")

                final_state_persisted = self.app.get_state(config)
                # Include pause reason if applicable
                final_values = final_state_persisted.values if final_state_persisted else {}
                if paused_at: final_values["_paused_at"] = paused_at # Add marker
                return final_values if final_state_persisted else {"error": f"State not found, paused at: {paused_at}"}

            except Exception as e:
                 logger.error(f"Error during agent run (Thread: {thread_id}): {e}", exc_info=True)
                 return final_state_this_run if final_state_this_run else {"error": str(e)}


        def update_state_and_resume_after_review(self, thread_id: str, human_feedback: HumanFeedback) -> Optional[Dict[str, Any]]:
             """Handles Agency Review feedback and resumes."""
             config = {"configurable": {"thread_id": thread_id}}
             logger.info(f"--- Resuming Agent (Thread: {thread_id}) after Agency Review ---")
             logger.debug(f"Feedback Received: {human_feedback}")

             # Prepare state update based on feedback
             update_for_resume = {"human_feedback": human_feedback}
             if not human_feedback.get("approved", True): # Assume approved if key missing? Or require it? Let's require.
                  # If rejecting, clear the plan so planner must replan
                  update_for_resume["planned_tool_inputs"] = None
                  update_for_resume["requires_agency_review"] = False # Clear review flag after rejection
                  logger.info("Clearing planned_tool_inputs due to rejection.")
             # Note: Planner node is responsible for clearing human_feedback from state *after* processing it

             try:
                  logger.debug(f"Applying state update: {update_for_resume}")
                  self.app.update_state(config, update_for_resume)
                  logger.info("State updated successfully.")
             except Exception as e:
                   logger.error(f"Failed to update state for thread {thread_id}: {e}")
                   return {"error": f"Failed to update state: {e}"}

             # Resume execution by invoking stream with None input
             final_state_this_resume = None
             paused_at = None
             try:
                for step_output in self.app.stream(None, config=config, stream_mode="values"):
                    final_state_this_resume = step_output
                    step_keys = list(step_output.keys())
                    logger.info(f"Step after resume completed. State Keys: {len(step_keys)}")

                    current_graph_state = self.app.get_state(config)
                    if current_graph_state and current_graph_state.next:
                         next_nodes = current_graph_state.next
                         if AGENCY_REVIEW_NODE_NAME in next_nodes:
                             logger.warning(f"--- Agent Paused AGAIN for Agency Review (Thread: {thread_id}) ---")
                             paused_at = "agency_review"
                             break
                         elif WAIT_CUSTOMER_NODE_NAME in next_nodes:
                              logger.warning(f"--- Agent Paused Waiting for Customer (Thread: {thread_id}) ---")
                              paused_at = "wait_customer"
                              break

                if not paused_at:
                    logger.info(f"--- Agent Resume Finished Naturally (Thread: {thread_id}) ---")

                final_state_persisted = self.app.get_state(config)
                final_values = final_state_persisted.values if final_state_persisted else {}
                if paused_at: final_values["_paused_at"] = paused_at
                return final_values if final_state_persisted else {"error": "Final state not found post-resume"}

             except Exception as e:
                 logger.error(f"Error during agent resume (Thread: {thread_id}): {e}", exc_info=True)
                 return final_state_this_resume if final_state_this_resume else {"error": str(e)}


        def resume_after_customer(self, thread_id: str, customer_reply: str) -> Optional[Dict[str, Any]]:
             """ Resumes from 'wait_customer' by running a new turn with the reply."""
             logger.info(f"--- Resuming Agent (Thread: {thread_id}) after Customer Reply ---")
             # The checkpointer loads the state paused at 'wait_customer'.
             # The new HumanMessage is added. The interrupt node completes,
             # and the graph routes back to the planner, which sees the new message.
             return self.run_turn(thread_id, customer_reply)

        # get_current_state remains the same
        def get_current_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
             # ... (Same as M5) ...
             config = {"configurable": {"thread_id": thread_id}}
             try:
                 state = self.app.get_state(config)
                 # logger.debug(f"Raw state from get_state: {state}")
                 return state.values if state else None
             except Exception as e:
                  logger.error(f"Error getting state for thread {thread_id}: {e}")
                  return None

    ```

5.  **`scripts/run_agent.py`**
    *   Update the simulation to handle the `wait_customer` pause and resume using the new runner method.

    ```python
    # scripts/run_agent.py
    # ... (imports, print_state_summary) ...

    # Use the actual node names defined in the agent graph
    AGENCY_REVIEW_NODE_NAME = "agency_review_pause"
    WAIT_CUSTOMER_NODE_NAME = "wait_customer_pause"

    def check_if_paused(agent_runner: PPAAgentRunner, thread_id: str) -> Optional[str]:
         """Checks if paused and returns the pause reason ('agency_review' or 'wait_customer') or None."""
         config = {"configurable": {"thread_id": thread_id}}
         graph_state = agent_runner.app.get_state(config)
         if graph_state and graph_state.next:
             next_nodes = graph_state.next
             if AGENCY_REVIEW_NODE_NAME in next_nodes:
                 return "agency_review"
             elif WAIT_CUSTOMER_NODE_NAME in next_nodes:
                 return "wait_customer"
         return None

    def simulate_agency_review(agent_runner: PPAAgentRunner, thread_id: str) -> Optional[Dict[str, Any]]:
        """Simulates agency review. Resumes agent and returns final state of that resume run."""
        print("\n--- !!! AGENCY REVIEW REQUIRED - SIMULATING !!! ---")
        current_state = agent_runner.get_current_state(thread_id)
        if not current_state: return {"error": "Cannot get state for review."}

        planned_action = current_state.get('planned_tool_inputs')
        tool_name = planned_action.get("tool_name") if planned_action else None
        print(f"Reviewing plan: {planned_action}")
        # print(f"Planner Thought:\n{current_state.get('agent_scratchpad')}\n")

        # Reset rejection tracker for each simulation instance if needed
        # if not hasattr(simulate_agency_review, "rejected_tools"): simulate_agency_review.rejected_tools = {}
        # thread_tool_key = f"{thread_id}_{tool_name}" # Key potentially needed if tracking rejections

        # Simulation Policy (Simplified)
        approved = True
        feedback_comment = "Approved by simulation."
        if tool_name == "rate_quote_tool": # Example: Always approve rating for now
             print("SIMULATION: Auto-approving 'rate_quote_tool'.")
             approved = True
        elif tool_name == "ask_customer_tool":
            print("SIMULATION: Auto-approving 'ask_customer_tool'.")
            approved = True
        else: # Auto-approve other API calls
             print(f"SIMULATION: Auto-approving tool '{tool_name}'.")
             approved = True

        # Prepare feedback payload
        human_feedback_payload = {"approved": approved, "comment": feedback_comment}
        print(f"Simulated Feedback: {human_feedback_payload}")
        print("-----------------------------------------")

        # Resume the agent with the feedback
        return agent_runner.update_state_and_resume_after_review(thread_id, human_feedback_payload)

    # --- Main Execution ---
    if __name__ == "__main__":
        logger.info("--- Milestone 6: Full HITL Simulation Test ---")

        # Clean slate
        db_path = SQLITE_DB_NAME
        if os.path.exists(db_path):
            logger.warning(f"Deleting existing database: {db_path}")
            os.remove(db_path)
        else:
            logger.info(f"Database {db_path} not found, starting fresh.")


        agent_runner = PPAAgentRunner()
        thread_id = "full_hitl_test_m6_002"
        current_state = None

        # --- Turn 1: Initial -> Plan ask_customer -> Agency Review -> Execute -> Wait ---
        print(f"\n--- Turn 1 START (Thread: {thread_id}) ---")
        turn1_input = "Hi, I need car insurance. My name is Bob Smith."
        current_state = agent_runner.run_turn(thread_id, turn1_input)

        # Handle potential agency review within the turn
        pause_reason = check_if_paused(agent_runner, thread_id)
        if pause_reason == "agency_review":
            current_state = simulate_agency_review(agent_runner, thread_id)

        print_state_summary(current_state, "Final State After Turn 1 Run/Resume")

        # Check if paused for customer
        pause_reason = check_if_paused(agent_runner, thread_id)
        if pause_reason == "wait_customer":
             print("\n*** Agent Paused Waiting for Customer Reply ***")
             last_output = current_state.get('last_tool_outputs', {}) if current_state else {}
             message_to_send = last_output.get('message_content', 'Error: Could not find message content.')
             print(f"Message generated by agent: '{message_to_send}'")
             # External system would send this message now.
        else:
             print("WARNING: Agent did not pause for customer reply as expected in Turn 1.")


        # --- Turn 2: Simulate Customer Reply ---
        print(f"\n--- Turn 2 START (Thread: {thread_id}) ---")
        customer_reply = "OK, Age 40, Car 2022 Ford Bronco VIN XYZ987, Address 100 Main St, Testville, FL 33333"
        current_state = agent_runner.resume_after_customer(thread_id, customer_reply)

        # Handle potential agency review again
        pause_reason = check_if_paused(agent_runner, thread_id)
        if pause_reason == "agency_review":
             current_state = simulate_agency_review(agent_runner, thread_id) # Simulate approval

        print_state_summary(current_state, "Final State After Turn 2 Run/Resume")

        # Agent should continue planning... e.g., maybe add vehicle now

        # --- Turn 3: No input, let agent continue planning ---
        print(f"\n--- Turn 3 START (Thread: {thread_id}) ---")
        current_state = agent_runner.run_turn(thread_id) # Run with None input

        pause_reason = check_if_paused(agent_runner, thread_id)
        if pause_reason == "agency_review":
             current_state = simulate_agency_review(agent_runner, thread_id)

        print_state_summary(current_state, "Final State After Turn 3 Run/Resume")

        # ... Continue simulating more turns until rate_quote or completion ...
        # Example: Turn 4 might plan rate_quote, get rejected once, then approved

        print("\nMilestone 6 Test Complete.")

    ```

**To Run Milestone 6:**

1.  Implement or add realistic mocks for all tools in `tools.py`.
2.  Update `prompts.py` with `PLANNER_PROMPT_TEMPLATE_V4`.
3.  Update `agent.py` with the V4 planner, new conditional logic, and graph structure.
4.  Update `run_agent.py` with the new simulation logic.
5.  Run `python scripts/run_agent.py`.
6.  **Observe:**
    *   The agent should now cycle through planning -> review (optional) -> execution.
    *   When `ask_customer_tool` is planned, approved, and executed, the run should pause at the "Wait for Customer" step.
    *   The script then simulates a reply using `resume_after_customer`.
    *   The agent resumes, the planner sees the new message, updates state, and continues planning the next API call (e.g., `add_vehicle_tool`).
    *   This API call might trigger another agency review before execution.

This milestone completes the core agentic loop including both HITL points and the ability to handle the full conversation cycle.

---

**Next Steps (Towards Production Readiness)**

You now have a functional agentic core. To make it production-ready:

1.  **Complete All Tool Implementations:** Replace *all* mocks in `src/ppa_agent/tools.py` with robust calls to the actual Mercury Insurance APIs. This includes:
    *   Handling authentication securely (reading tokens/credentials from config/env).
    *   Mapping `AgentState` fields correctly to API request parameters.
    *   Parsing API responses accurately, extracting necessary data (e.g., driver IDs, vehicle IDs, coverage options, rate details) and storing it back into `AgentState` (likely via `mercury_session` or `customer_info`).
    *   Implementing comprehensive error handling for API-specific error codes and network issues, returning informative `{"status": "error", "message": ...}` dictionaries.

2.  **Refine Planner Prompt (V5+):** This is the most critical iterative step.
    *   **Error Handling:** Explicitly instruct the LLM how to react to different `error` statuses and messages in `last_tool_outputs`. Should it retry? Ask the customer for clarification? Escalate via `request_human_review_tool`?
    *   **Tool Selection Logic:** Improve guidance on choosing the *correct* next API based on the sequence (Initiate -> Prefill? -> Add/Get Driver -> Add Vehicle -> Update Coverage -> Assign Vehicle -> Rate). How does it know which driver/vehicle ID to use? (Needs careful state management and prompt instructions).
    *   **Information Extraction:** While tools execute actions, the planner might still need to extract details from the latest `HumanMessage` to update `customer_info` *before* planning the next tool. This could be a separate "update_customer_info" tool or part of the planner's reasoning.
    *   **Edge Cases:** Test with ambiguous customer input, corrections, requests to change previously provided info, multiple vehicles/drivers.
    *   **Goal Completion:** Define precisely how the planner recognizes the quote is complete (e.g., after `rate_quote_tool` succeeds) and what the final action should be (e.g., use `prepare_final_summary_tool`).
    *   **Efficiency:** Can the planner chain simple calls or does it need to plan every single step? (Start with step-by-step).

3.  **Implement `request_human_review_tool`:** Add a tool that the *planner* can explicitly choose to call when it's stuck or needs human intervention beyond the standard pre-execution review. This tool would simply set `requires_agency_review = True` and perhaps add a specific note to the scratchpad.

4.  **External System & HITL Interface:**
    *   **Email Handling:** Build the service that polls/receives emails, parses them (using `email` library, regex, etc.) to extract sender, subject, body, and potentially thread identifiers (like `In-Reply-To` headers or subject parsing).
    *   **Thread Resolution:** Implement logic to map incoming emails to existing `thread_id`s stored in your `SqliteSaver` database or create new ones.
    *   **Triggering/Resuming:** Call the appropriate `PPAAgentRunner` methods (`run_turn`, `update_state_and_resume_after_review`, `resume_after_customer`).
    *   **Sending Messages:** Retrieve the message content from the state after the agent pauses at `wait_customer` and send the email.
    *   **HITL UI:** Develop a web interface (e.g., using Flask/Django/FastAPI) for human agents. This UI should:
        *   Query for paused agent runs (checking the state in the SQLite DB).
        *   Display relevant `AgentState` information (history, planned action, customer info).
        *   Allow agents to Approve, Reject with comments, or potentially Edit planned `tool_inputs`.
        *   Call back to the agent server/runner to update state and resume the graph.

5.  **Robust Error Handling & Logging:**
    *   Add more specific `try...except` blocks in tools and nodes.
    *   Implement retry mechanisms for transient API errors (e.g., using `tenacity`).
    *   Use structured logging (e.g., including `thread_id` in all logs) for easier debugging.
    *   Consider using LangSmith for detailed tracing of LLM calls, tool executions, and state changes.

6.  **Security:**
    *   Secure API keys/credentials (use environment variables, secrets management).
    *   Be mindful of PII in logs and state persistence (masking, encryption at rest).
    *   Input validation for data coming from external sources.

7.  **Testing:**
    *   **Unit Tests:** For individual tools (mocking API calls), state transformations, prompt formatting.
    *   **Integration Tests:** Test full conversation flows using mocked APIs first, then potentially against a staging API environment. Cover success paths, info requests, API errors, HITL interactions. Use `pytest`.

8.  **Deployment:**
    *   Containerize the agent runner and any external system components (Dockerfile).
    *   Choose deployment strategy (Cloud Run, Kubernetes, VMs).
    *   Set up monitoring and alerting.

This detailed plan provides the code for Milestones 5 & 6 and outlines the significant effort required to move from this functional core to a production-ready system. Good luck!

## Refined Agency Review (HITL) Workflow (Milestone 6)

This milestone focused on improving the Human-in-the-Loop (HITL) process for agency review:

1.  **Planner Node (`planner_node`) Logic Update:**
    *   The planner now explicitly handles the `human_feedback` field in the `AgentState`.
    *   **Approved Feedback:** If feedback is `{"approved": true, ...}`, the planner:
        *   Retains the previously `planned_tool_inputs`.
        *   Sets `requires_agency_review` to `False`.
        *   Updates `last_tool_outputs` to indicate human approval.
        *   **Skips calling the LLM** for replanning.
        *   Routes directly to the `executor_node_wrapper` via the conditional edge logic.
    *   **Rejected Feedback:** If feedback is `{"approved": false, ...}`, the planner:
        *   Clears the `planned_tool_inputs`.
        *   Updates `last_tool_outputs` to indicate human rejection.
        *   Includes the rejection comment in the scratchpad/prompt history.
        *   **Calls the LLM** to generate a new plan based on the rejection feedback.

2.  **Executor Node (`executor_node_wrapper`) Update:**
    *   The executor is now responsible for looking up the correct tool function based on the `tool_name` provided in `planned_tool_inputs`.
    *   It uses the `TOOL_MAP` imported from `src.ppa_agentic_v2.tools`.
    *   It invokes the tool asynchronously using `selected_tool.ainvoke(input_args)`.
    *   Error handling for missing tools or execution failures has been added.

3.  **Prompt Formatting (`format_planner_prompt`) Update:**
    *   The function signature was updated to accept `current_tool_outputs` and `human_feedback_str` directly, making it more modular.

**Outcome:** This refined flow allows an approved action to proceed directly to execution without requiring a redundant LLM call or a second agency review pause. Rejections trigger a replanning step, incorporating the feedback provided by the human reviewer.