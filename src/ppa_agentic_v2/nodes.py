# src/ppa_agentic_v2/nodes.py

import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.messages import BaseMessage
from pydantic import ValidationError
from langchain_core.exceptions import OutputParserException
from langgraph.constants import Interrupt
from .state import AgentState
from .tools import all_tools, TOOL_MAP
from .config import logger
from .prompts import format_planner_prompt
from .llm_setup import llm_with_tools

# --- Planner Node Helper Functions ---

def _handle_human_feedback(state: AgentState, update: Dict[str, Any]) -> Tuple[bool, Optional[List[BaseMessage]], Optional[Dict[str, Any]]]:
    """Processes human feedback if present, updates the state update dict, and determines if LLM call is needed."""
    logger.debug("Checking for human feedback.")
    call_llm = True
    prompt_input_messages = None
    current_tool_outputs = None

    if state.human_feedback:
        logger.info("Planner processing human feedback from previous step.")
        feedback = state.human_feedback
        approved = feedback.get("approved", False)
        comment = feedback.get("comment", "")
        planned_action = state.planned_tool_inputs # Action planned *before* review
        tool_name_reviewed = planned_action.get("tool_name", "unknown") if planned_action else "unknown"

        if approved:
            logger.info(f"Action '{tool_name_reviewed}' approved by human. Routing to executor.")
            # Keep the previously planned action but mark review as done
            update["planned_tool_inputs"] = planned_action
            update["requires_agency_review"] = False # Approved, no need for review *again*
            update["last_tool_outputs"] = {"status": "approved_by_human", "tool_name": tool_name_reviewed, "message": comment}
            update["human_feedback"] = None # Consume feedback
            call_llm = False # Skip LLM call, proceed directly to execution

        else: # Rejected
            logger.info(f"Action '{tool_name_reviewed}' rejected by human. Replanning needed.")
            current_tool_outputs = { # This becomes input for replanning prompt
                "status": "rejected_by_human",
                "message": comment or "Action rejected by human review.",
                "tool_name": tool_name_reviewed
            }
            update["last_tool_outputs"] = current_tool_outputs # Record rejection as last output
            update["planned_tool_inputs"] = None # Clear the rejected plan
            update["requires_agency_review"] = False # No action planned yet
            update["human_feedback"] = None # Consume feedback
            # Let call_llm remain True to proceed to LLM replanning below
            prompt_input_messages = format_planner_prompt(state, all_tools, current_tool_outputs=current_tool_outputs, human_feedback_str=None)
    else:
        logger.debug("No human feedback found.")

    return call_llm, prompt_input_messages, current_tool_outputs

async def _invoke_llm_and_parse(state: AgentState, prompt_input_messages: Optional[List[BaseMessage]], current_tool_outputs: Optional[Dict[str, Any]], update: Dict[str, Any]):
    """Invokes the LLM, parses the response, and updates the state update dict."""
    if not llm_with_tools:
        logger.error("LLM with tools not initialized. Cannot invoke planner LLM.")
        update["agent_scratchpad"] = "Error: LLM not available."
        update["requires_agency_review"] = True # Force review if LLM failed
        update["planned_tool_inputs"] = None
        return

    # Generate prompt if not provided (e.g., standard run, not after rejection)
    if prompt_input_messages is None:
        logger.info("Planner running standard logic (no feedback processed or action approved).")
        # Use state.last_tool_outputs from the *previous* node (e.g., executor)
        current_tool_outputs = state.last_tool_outputs
        prompt_input_messages = format_planner_prompt(state, all_tools, current_tool_outputs=current_tool_outputs, human_feedback_str=None)

    # --- Invoke LLM --- #
    try:
        logger.debug(f"Planner Prompt Messages: {prompt_input_messages}")
        response = await llm_with_tools.ainvoke(prompt_input_messages)
        logger.debug(f"LLM Response: {response.content}")

        # --- Parse LLM Output --- #
        parsed_output = None
        if isinstance(response.content, str):
            raw_llm_output = response.content
            # Relaxed JSON parsing: Look for JSON block or try parsing directly
            json_block_match = re.search(r"```json\n(.*?)\n```", raw_llm_output, re.DOTALL)
            if json_block_match:
                json_str = json_block_match.group(1).strip()
            else:
                # Try parsing the whole string if no block found
                json_str = raw_llm_output.strip()

            try:
                parsed_output = json.loads(json_str)
                logger.info(f"Successfully parsed LLM JSON: {parsed_output}")
            except json.JSONDecodeError as json_e:
                logger.error(f"Failed to parse JSON from LLM output: {json_e}")
                logger.debug(f"LLM Raw output that failed parsing: {raw_llm_output}")
                raise OutputParserException(f"LLM output could not be parsed as JSON. Raw: {raw_llm_output}") from json_e
        else:
            logger.error(f"Unexpected LLM response content type: {type(response.content)}")
            raise OutputParserException(f"Unexpected LLM response content type: {type(response.content)}")

        if not isinstance(parsed_output, dict):
            logger.error(f"Parsed output is not a dictionary: {parsed_output}")
            raise OutputParserException(f"Parsed output is not a dictionary: {parsed_output}")

        # --- Update State Based on LLM Output --- #
        update["agent_scratchpad"] = parsed_output.get("thought", "") # Extract thought process
        action_details = parsed_output.get("action")

        # Reset plan and review status initially
        update["planned_tool_inputs"] = None

        if isinstance(action_details, str) and action_details == "WAIT_FOR_CUSTOMER":
            logger.info("Planner decided to wait for customer response.")
            # No tool is planned, the graph will route to CUSTOMER_WAIT_NODE based on check_planner_output
            # We might need a flag in the state if check_planner_output needs it, but the interrupt should handle the pause.
            pass # planned_tool_inputs remains None

        elif isinstance(action_details, dict):
            tool_name = action_details.get("tool_name")
            tool_inputs = action_details.get("tool_inputs", {})
            if tool_name:
                update["planned_tool_inputs"] = {
                    "tool_name": tool_name,
                    "args": tool_inputs # Use 'args' key consistent with executor node
                }
                logger.info(f"Planner planned tool: {tool_name}")
                # Note: Review requirement is now determined by check_planner_output
            else:
                logger.warning("LLM action dictionary is missing 'tool_name'. Treating as no action.")
                # planned_tool_inputs remains None

        else:
            logger.error(f"LLM 'action' field has unexpected format: {action_details}. Treating as no action.")
            # planned_tool_inputs remains None
            # Consider forcing review here if this happens often
            # update["requires_agency_review"] = True

    except (OutputParserException, ValidationError, json.JSONDecodeError) as e:
        logger.error(f"Error processing planner LLM output: {e}")
        update["agent_scratchpad"] = f"Error: Could not process LLM output. Error: {e}. Raw output: {response.content if 'response' in locals() else 'N/A'}"
        update["requires_agency_review"] = True # Force review if parsing fails
        update["planned_tool_inputs"] = None
    except Exception as e:
        logger.error(f"Unexpected error during planner LLM invocation/parsing: {e}", exc_info=True)
        update["agent_scratchpad"] = f"Error: An unexpected error occurred in the planner LLM step. Error: {e}"
        update["requires_agency_review"] = True # Force review on unexpected errors
        update["planned_tool_inputs"] = None


# --- Graph Nodes ---

async def planner_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Entering Planner Node ---")
    update: Dict[str, Any] = {
        "agent_scratchpad": state.agent_scratchpad, # Carry over previous thought unless updated
        "requires_agency_review": False, # Default unless feedback/LLM sets it
        "planned_tool_inputs": None, # Default unless feedback/LLM sets it or approval keeps it
        "human_feedback": None, # Always clear feedback after processing
        "last_tool_outputs": None # Default unless feedback/LLM sets it
    }

    # 1. Handle Human Feedback (if any)
    call_llm, prompt_input_messages, current_tool_outputs = _handle_human_feedback(state, update)

    # 2. Invoke LLM and Parse Output (if not bypassed by feedback handling)
    if call_llm:
        await _invoke_llm_and_parse(state, prompt_input_messages, current_tool_outputs, update)

    # Filter out None values before logging and returning
    final_update = {k: v for k, v in update.items() if v is not None}
    logger.info(f"--- Exiting Planner Node. State updates: {final_update} ---")
    return final_update

async def execute_tool_no_review(state: AgentState) -> Dict[str, Any]:
    """Executes the tool specified in state.planned_tool_inputs.
    This node is ONLY called when the planner decides to execute a tool
    AND does NOT require agency review.
    """
    logger.info("--- Entering Executor Node (Review NOT Required) ---")
    planned_tool_inputs = state.planned_tool_inputs

    if not planned_tool_inputs or not isinstance(planned_tool_inputs, dict):
        logger.warning("Executor called without valid planned_tool_inputs.")
        return {"last_tool_outputs": {"status": "error", "message": "No tool planned for execution."}}

    tool_name = planned_tool_inputs.get("tool_name")
    tool_args = planned_tool_inputs.get("args", {}) # Args might be empty

    if not tool_name:
        logger.error("Planned tool inputs missing 'tool_name'.")
        return {"last_tool_outputs": {"status": "error", "message": "Tool name missing in plan."}}

    logger.info(f"Executing tool (no review required): {tool_name} with args: {tool_args}")

    tool_output = None
    if tool_name in TOOL_MAP:
        selected_tool = TOOL_MAP[tool_name]
        try:
            # Ensure args are passed correctly, tool.ainvoke expects a dict
            # If tool_args is None or not a dict, provide an empty dict
            input_args = tool_args if isinstance(tool_args, dict) else {}
            tool_result = await selected_tool.ainvoke(input_args)

            # Assuming tool_result is serializable (string, dict, list, etc.)
            tool_output = {"status": "success", "tool_name": tool_name, "result": tool_result}
            logger.info(f"Tool {tool_name} executed successfully. Result: {tool_result}")

        except ValidationError as ve:
            logger.error(f"Validation error executing tool {tool_name} with args {tool_args}: {ve}", exc_info=True)
            tool_output = {"status": "error", "tool_name": tool_name, "message": f"Input validation failed for {tool_name}: {str(ve)}"}
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            tool_output = {"status": "error", "tool_name": tool_name, "message": f"Error executing tool {tool_name}: {str(e)}"}
    else:
        logger.error(f"Tool '{tool_name}' not found in TOOL_MAP.")
        tool_output = {"status": "error", "message": f"Tool '{tool_name}' not recognized."}

    # Clear the plan after execution attempt (success or failure)
    return {"last_tool_outputs": tool_output, "planned_tool_inputs": None}


async def agency_review_pause_node(state: AgentState) -> Interrupt:
    """Placeholder node for the agency review interrupt point.

    The graph pauses *before* AND *after* this node when run in dev mode.
    To simulate review:
    1. Click 'Continue' on the first pause (before).
    2. Manually edit the 'human_feedback' field in the state JSON in the UI
       (e.g., {"approved": false, "comment": "Needs VIN"}).
    3. Click 'Continue' on the second pause (after).
    """
    logger.info(f"--- Pausing for Agency Review (Interrupt). Planned Action: {state.planned_tool_inputs} ---")
    # No state modification needed here by the node itself.
    # Feedback is added externally via API call.
    return Interrupt(value=None)


async def customer_wait_node(state: AgentState) -> Interrupt:
    """Node to pause the graph while waiting for customer input.

    The graph configuration uses this node as an interrupt point.
    When the graph is resumed (presumably with updated 'messages' in the state),
    it will proceed back to the planner.
    """
    logger.info("--- Pausing for Customer Input (Interrupt) ---")
    # No state update needed here, just pausing.
    # The actual message update happens externally via API call.
    return Interrupt(value=None)
