# src/ppa_agentic_v2/graph.py

import logging
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from .state import AgentState
# Import config values including node names
from .config import (
    logger,
    SQLITE_DB_NAME,
    PLANNER_NODE_NAME,
    EXECUTOR_NO_REVIEW_NODE_NAME,
    AGENCY_REVIEW_NODE_NAME,
    CUSTOMER_WAIT_NODE_NAME,
)
from .nodes import (
    planner_node,
    execute_tool_no_review,
    agency_review_pause_node,
    customer_wait_node
)

# Define which tools require agency review before execution
TOOLS_REQUIRING_REVIEW = {
    # Add tool names that need human oversight
    # Example: "initiate_quote_tool", "add_vehicle_tool"
    "initiate_quote_tool", # Example - adjust as needed
    "ask_customer_tool"  # Always review before sending messages to customer
}

# --- Graph Conditional Edges --- #

def check_planner_output(state: AgentState) -> str:
    """Determines the next step based on the planner's output and human feedback status."""
    logger.info("--- Checking Planner Output for Routing --- ")
    last_output = state.last_tool_outputs

    # 1. Check for Human Feedback Status first
    if isinstance(last_output, dict):
        feedback_status = last_output.get("status")
        if feedback_status == "approved_by_human":
            approved_tool = last_output.get('tool_name', 'unknown')
            logger.info(f"Detected human approval for tool: {approved_tool}. Routing to executor.")
            # Sanity check: Ensure the approved action is actually in planned_tool_inputs
            if state.planned_tool_inputs:
                logger.debug(f"Approved action details found in planned_tool_inputs: {state.planned_tool_inputs}")
            else:
                # This case *shouldn't* happen with the new logic, but log if it does.
                logger.warning(f"Human approval status detected, but planned_tool_inputs is empty! Routing to executor based on status.")
            return EXECUTOR_NO_REVIEW_NODE_NAME
        elif feedback_status == "rejected_by_human":
            rejected_tool = last_output.get('tool_name', 'unknown')
            logger.info(f"Detected human rejection for tool: {rejected_tool}. Routing back to planner for replanning.")
            return PLANNER_NODE_NAME
        elif feedback_status == "error" and last_output.get("message") == "Error processing feedback.":
             # If feedback processing itself failed, go back to planner
             logger.error("Error processing human feedback detected. Routing back to planner.")
             return PLANNER_NODE_NAME

    # 2. If no feedback processed in this step, check if review is required for a pending action
    if state.requires_agency_review:
        pending_action = state.action_pending_review
        tool_name = pending_action.get('tool_name', 'unknown') if isinstance(pending_action, dict) else 'unknown'
        logger.info(f"Planner requires agency review for planned action: {tool_name}. Routing to review pause.")
        return AGENCY_REVIEW_NODE_NAME

    # 3. If no review required, check if an action is planned for direct execution
    if state.planned_tool_inputs:
        tool_name = state.planned_tool_inputs.get('tool_name', 'unknown')
        logger.info(f"Planner decided to execute tool '{tool_name}' without review. Routing to executor.")
        return EXECUTOR_NO_REVIEW_NODE_NAME

    # 4. If no action planned and no review needed, assume waiting for customer
    # (This relies on the planner outputting action: null when waiting)
    logger.info("No action planned or pending review. Assuming wait for customer input. Routing to wait pause.")
    return CUSTOMER_WAIT_NODE_NAME

# --- Build Agent Graph ---
def build_agent_graph() -> CompiledStateGraph:
    """Builds the LangGraph StateGraph for the PPA Agent V2."""
    logger.info("Building agent graph structure...")
    graph = StateGraph(AgentState)

    # 1. Add Nodes
    graph.add_node(PLANNER_NODE_NAME, planner_node) 
    graph.add_node(EXECUTOR_NO_REVIEW_NODE_NAME, execute_tool_no_review) 
    graph.add_node(AGENCY_REVIEW_NODE_NAME, agency_review_pause_node) 
    graph.add_node(CUSTOMER_WAIT_NODE_NAME, customer_wait_node) 

    # 2. Define Edges
    graph.add_edge(START, PLANNER_NODE_NAME)

    # Conditional edge from Planner based on review requirement
    graph.add_conditional_edges(
        PLANNER_NODE_NAME,
        check_planner_output,
        {
            AGENCY_REVIEW_NODE_NAME: AGENCY_REVIEW_NODE_NAME,
            EXECUTOR_NO_REVIEW_NODE_NAME: EXECUTOR_NO_REVIEW_NODE_NAME, 
            CUSTOMER_WAIT_NODE_NAME: CUSTOMER_WAIT_NODE_NAME, 
            PLANNER_NODE_NAME: PLANNER_NODE_NAME,
        }
    )

    # Edge from Executor (No Review) back to Planner for next step
    graph.add_edge(EXECUTOR_NO_REVIEW_NODE_NAME, PLANNER_NODE_NAME)

    # Edge *after* Agency Review Pause (human provides feedback) back to Planner
    graph.add_edge(AGENCY_REVIEW_NODE_NAME, PLANNER_NODE_NAME)

    # Edge *after* Customer Wait Pause (new customer input) back to Planner
    # The graph pauses *after* the customer_wait_node. When resumed,
    # it proceeds to the planner, which will then process the new input (in messages).
    # graph.add_edge(CUSTOMER_WAIT_NODE_NAME, PLANNER_NODE_NAME) # <-- REMOVED: Interrupt handles pause; resume implicitly restarts flow

    logger.info("Agent graph structure defined (compilation deferred).")

    # Compile the graph, specifying interrupt points
    compiled_graph = graph.compile(
        interrupt_before=[ # <-- Keep interrupt configuration
            AGENCY_REVIEW_NODE_NAME,
            CUSTOMER_WAIT_NODE_NAME,
        ],
    )
    logger.info("Agent graph compiled with interrupt points.") 
    return compiled_graph
