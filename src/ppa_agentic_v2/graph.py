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
    CUSTOMER_WAIT_NODE_NAME
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
    "initiate_quote_tool" # Example - adjust as needed
}

# --- Conditional Edge Logic from Planner --- #
def check_planner_output(state: AgentState) -> str:
    """
    Routes control based on the planner's decision stored in planned_tool_inputs.
    - If no tool is planned, assumes LLM wants to wait for customer.
    - If ask_customer_tool is planned, routes to wait.
    - If another tool is planned, checks if it requires review.
    """
    logger.info("--- Checking Planner Output for Routing ---")
    planned_tool = state.planned_tool_inputs

    if planned_tool is None:
        # This occurs if the LLM explicitly outputted 'WAIT_FOR_CUSTOMER' action.
        logger.info("No tool planned. Assuming wait for customer input. Routing to customer wait node.")
        return CUSTOMER_WAIT_NODE_NAME

    # Tool *is* planned
    if isinstance(planned_tool, dict):
        tool_name = planned_tool.get("tool_name")
        if not tool_name:
             logger.error("Planned tool inputs exist but missing 'tool_name'. Routing to END to prevent loop.")
             return END # Avoid infinite loops if plan is malformed

        logger.info(f"Planner decided to use tool: {tool_name}")

        # Check against the fully qualified name the LLM seems to be using
        if tool_name == "functions.ask_customer_tool":
            logger.info("Tool is ask_customer_tool. Routing to customer wait node.")
            return CUSTOMER_WAIT_NODE_NAME

        if tool_name in TOOLS_REQUIRING_REVIEW:
            logger.info(f"Tool '{tool_name}' requires review. Routing to agency review node.")
            return AGENCY_REVIEW_NODE_NAME
        else:
            logger.info(f"Tool '{tool_name}' does not require review. Routing to execution node.")
            return EXECUTOR_NO_REVIEW_NODE_NAME
    else:
        # Should not happen if planner node works correctly, but handle defensively
        logger.error(f"Planned tool inputs is not a dictionary or None: {planned_tool}. Routing to END.")
        return END


# --- Build Agent Graph ---
def build_agent_graph() -> CompiledStateGraph:
    """Builds and compiles the LangGraph StateGraph for the PPA Agentic V2."""
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
            END: END,
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
