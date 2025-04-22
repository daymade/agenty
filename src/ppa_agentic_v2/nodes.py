# src/ppa_agentic_v2/nodes.py

import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.messages import BaseMessage
from langchain_core.language_models.base import BaseLanguageModel
from pydantic import ValidationError
from langchain_core.exceptions import OutputParserException
from langgraph.constants import Interrupt
from .state import AgentState
from .tools import all_tools, TOOL_MAP
from .prompts import format_planner_prompt
from .llm_setup import llm_with_tools

logger = logging.getLogger(__name__)

# --- Planner Node Helper Functions ---

def _handle_human_feedback(state: AgentState, update: Dict[str, Any], feedback: Dict[str, Any]) -> Tuple[bool, Optional[List[BaseMessage]]]:
    """Processes human feedback, updates state, and returns messages for LLM if rejected."""
    logger.info("--- Processing Human Feedback --- ")
    messages_for_llm = None
    try:
        approved = feedback.get("approved", False)
        comment = feedback.get("comment", "")
        # Read the action that WAS pending review
        action_that_was_reviewed = state.action_pending_review
        logger.debug(f"[_handle_human_feedback] Read state.action_pending_review: {action_that_was_reviewed}")
        tool_name_reviewed = action_that_was_reviewed.get("tool_name", "unknown") if isinstance(action_that_was_reviewed, dict) else "unknown"

        if approved:
            logger.info(f"Action '{tool_name_reviewed}' approved by human. Routing to executor.")
            # Move the approved action to planned_tool_inputs for the executor
            update["planned_tool_inputs"] = action_that_was_reviewed
            feedback_output = {"status": "approved_by_human", "tool_name": tool_name_reviewed, "message": comment}
            update["last_tool_outputs"] = feedback_output # For immediate routing
            # Log to history
            update["event_history"] = [{"type": "human_feedback", **feedback_output}]
            # Clear pending review state
            update["action_pending_review"] = None
            update["requires_agency_review"] = False
            update["human_feedback"] = None # Consume feedback
            return True, None # Indicates feedback processed, no LLM messages needed
        else:
            logger.info(f"Action '{tool_name_reviewed}' rejected by human. Routing back to planner.")
            # Prepare messages for the planner to explain the rejection
            messages_for_llm = [
                BaseMessage(content=json.dumps(action_that_was_reviewed), tool_calls=[]), # Show the rejected action
                BaseMessage(content=f"Human Feedback: Rejected. Comment: {comment}")
            ]
            feedback_output = {"status": "rejected_by_human", "tool_name": tool_name_reviewed, "message": comment}
            update["last_tool_outputs"] = feedback_output # For immediate routing
            # Log to history
            update["event_history"] = [{"type": "human_feedback", **feedback_output}]
            # Clear pending review state, requires_review flag, and feedback
            update["action_pending_review"] = None
            update["requires_agency_review"] = False
            update["human_feedback"] = None
            # Ensure no tool is planned for execution
            update["planned_tool_inputs"] = None
            return True, messages_for_llm # Indicates feedback processed, provide messages for LLM

    except Exception as e:
        logger.error(f"Error processing human feedback: {e}", exc_info=True)
        error_output = {"status": "error", "message": "Error processing feedback."}
        update["last_tool_outputs"] = error_output # For immediate routing
        # Log to history
        update["event_history"] = [{"type": "feedback_error", **error_output}]
        update["action_pending_review"] = None
        update["requires_agency_review"] = False
        update["human_feedback"] = None
        return True, [BaseMessage(content=f"System Error processing feedback: {e}")] # Let LLM know

async def _invoke_llm_and_parse(state: AgentState, prompt_input: Any, llm_with_tools: BaseLanguageModel, update: Dict[str, Any]):
    """Invokes the LLM, parses the JSON output, and updates the state dictionary."""
    try:
        logger.debug(f"Invoking planner LLM with prompt input type: {type(prompt_input)}")
        # Removed overly strict type checking - pass prompt_input directly to ainvoke
        response = await llm_with_tools.ainvoke(prompt_input)
        logger.debug(f"LLM Raw Response: {response}")

        # Assuming response.content is the JSON string
        json_string = response.content.strip()
        if json_string.startswith("```json"): # Check for markdown fences
            json_string = json_string[7:-3].strip() # Strip ```json\n and ```
        elif json_string.startswith("```"):
            json_string = json_string[3:-3].strip() # Strip ```\n and ```

        # Add check after potential stripping
        if not json_string.startswith('{'):
            raise ValueError(f"LLM response content, after potential stripping, is not a valid JSON object string: {json_string}")

        print(f"--- RAW LLM OUTPUT ---\n{json_string}\n----------------------")

        parsed_output = json.loads(json_string)
        print(f"--- PARSED JSON ---\n{parsed_output}\n-------------------")

        logger.info(f"LLM Parsed Output: {parsed_output}")

        thought = parsed_output.get("thought", "")
        planned_action = parsed_output.get("action") # Can be dict or null
        requires_review = parsed_output.get("requires_agency_review", False)

        # Update state based on parsed output
        update["agent_scratchpad"] = thought
        update["requires_agency_review"] = requires_review

        if requires_review and isinstance(planned_action, dict):
            # Action planned, requires review -> Store in action_pending_review
            logger.info(f"Planner requires review for action: {planned_action.get('tool_name')}")
            update["action_pending_review"] = planned_action
            update["planned_tool_inputs"] = None
        elif isinstance(planned_action, dict):
            # Action planned, no review needed -> Store in planned_tool_inputs
            logger.info(f"Planner proceeding with action (no review): {planned_action.get('tool_name')}")
            update["planned_tool_inputs"] = planned_action
            update["action_pending_review"] = None
        else:
            # No action planned (likely waiting for customer) -> Clear both
            logger.info("Planner decided to wait (action is null).")
            update["planned_tool_inputs"] = None
            update["action_pending_review"] = None
            # Optionally signal explicit wait? Depends on graph logic

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON response: {e}\nResponse: {response.content if 'response' in locals() else 'N/A'}", exc_info=True)
        update["agent_scratchpad"] = f"Error: Failed to parse LLM JSON response. {e}"
        update["planned_tool_inputs"] = None
        update["action_pending_review"] = None
        update["requires_agency_review"] = False
    except Exception as e:
        logger.error(f"Unexpected error during planner LLM invocation/parsing: {e}", exc_info=True)
        update["agent_scratchpad"] = f"Error: Unexpected error during planning. {e}"
        update["planned_tool_inputs"] = None
        update["action_pending_review"] = None
        update["requires_agency_review"] = False


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
    feedback = state.human_feedback
    if feedback:
        feedback_processed, messages_for_llm = _handle_human_feedback(state, update, feedback)
        if feedback_processed and not messages_for_llm:
            # Feedback was processed (e.g., approved), no need to call LLM again now.
            logger.info("Human feedback processed (approved), skipping LLM call in this step.")
            # Update scratchpad to reflect feedback was handled
            last_status = state.last_tool_outputs.get('status', 'unknown') if state.last_tool_outputs else 'N/A'
            update["agent_scratchpad"] = f"Processed human feedback: {last_status}. Comment: {feedback.get('comment','')}"
            print(f">>> PLANNER NODE UPDATE (feedback path) for thread {state.thread_id if hasattr(state, 'thread_id') and state.thread_id else 'UNKNOWN'}: {update}")
            return update # Return immediately, graph proceeds based on feedback handler's updates
        elif feedback_processed and messages_for_llm:
            # Feedback was processed (e.g., rejected), add rejection details for LLM
            logger.info("Human feedback processed (rejected), adding details to LLM input.")
            base_messages = messages_for_llm
            # Clear feedback from state as it's now in message history
            state.human_feedback = None # Clear it here before format_planner_prompt

    # Format the prompt for the LLM using the entire state
    tools = all_tools
    prompt_input = format_planner_prompt(state, tools)
    logger.debug(f"Formatted prompt input type: {type(prompt_input)}")

    # Invoke LLM and parse output to update the state dictionary
    await _invoke_llm_and_parse(state, prompt_input, llm_with_tools, update)

    thread_id_for_log = state.thread_id if hasattr(state, 'thread_id') and state.thread_id else 'UNKNOWN'
    print(f">>> PLANNER NODE UPDATE (LLM path) for thread {thread_id_for_log}: {update}")
    return update

async def execute_tool_no_review(state: AgentState) -> Dict[str, Any]:
    """Executes the tool specified in state.planned_tool_inputs.
    This node is ONLY called when the planner decides to execute a tool
    AND does NOT require agency review, OR when an action was approved via HITL.
    """
    logger.info("--- Entering Executor Node (Review NOT Required) ---")
    # Tool to execute is now reliably in planned_tool_inputs
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

    # Log the execution result to history
    event_to_log = {"type": "tool_execution", **tool_output}

    # Clear the plan after execution attempt (success or failure)
    # Also clear any lingering pending review action if somehow it wasn't cleared before
    print(f">>> TOOL NODE OUTPUT for tool '{tool_name}': {tool_output}")
    return {
        "last_tool_outputs": tool_output, # For immediate routing
        "event_history": [event_to_log], # Append to history
        "planned_tool_inputs": None,
        "action_pending_review": None # Ensure cleared here too
    }

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
