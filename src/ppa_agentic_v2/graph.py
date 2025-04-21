# src/ppa_agentic_v2/graph.py

import logging
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import AgentState
from .config import logger, SQLITE_DB_NAME
from .nodes import (
    planner_node,
    execute_tool_no_review,
    agency_review_pause_node
)

# --- Constants (Node Names) ---
PLANNER_NODE_NAME = "planner"
EXECUTOR_NO_REVIEW_NODE_NAME = "execute_tool_no_review"
AGENCY_REVIEW_NODE_NAME = "agency_review_pause"

# --- Conditional Edge Logic ---
def check_agency_review(state: AgentState) -> str:
    """
    Conditional edge logic after the planner node.
    Routes to review, execution, or end based on planner's decision.
    """
    logger.info("--- Checking Agency Review Requirement --- ")
    requires_review = state.requires_agency_review
    planned_action = state.planned_tool_inputs

    if requires_review and planned_action:
        logger.info("Routing to Agency Review Pause.")
        return AGENCY_REVIEW_NODE_NAME # Route to the pause node
    elif planned_action:
        # Action planned, but no review needed
        logger.info(f"Routing directly to Executor for tool: {planned_action.get('tool_name', 'unknown')}.")
        return EXECUTOR_NO_REVIEW_NODE_NAME # Route to direct execution
    else:
        # No action planned (e.g., final answer or error)
        logger.info("No tool planned or final answer. Routing to End.")
        return END

# --- Build Agent Graph ---
def build_agent_graph() -> StateGraph:
    """Builds the LangGraph StateGraph for the PPA Agent (V3 with Agency Review)."""
    memory = SqliteSaver.from_conn_string(SQLITE_DB_NAME)
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node(PLANNER_NODE_NAME, planner_node)
    workflow.add_node(AGENCY_REVIEW_NODE_NAME, agency_review_pause_node)
    workflow.add_node(EXECUTOR_NO_REVIEW_NODE_NAME, execute_tool_no_review) # Add the no-review executor

    # 2. Define Edges
    workflow.add_edge(START, PLANNER_NODE_NAME)

    # Conditional edge from Planner based on review requirement
    workflow.add_conditional_edges(
        PLANNER_NODE_NAME,
        check_agency_review,
        {
            AGENCY_REVIEW_NODE_NAME: AGENCY_REVIEW_NODE_NAME,
            EXECUTOR_NO_REVIEW_NODE_NAME: EXECUTOR_NO_REVIEW_NODE_NAME, # Route to no-review executor
            END: END
        }
    )

    # Edge from Executor (No Review) back to Planner for next step
    workflow.add_edge(EXECUTOR_NO_REVIEW_NODE_NAME, PLANNER_NODE_NAME)

    # Edge *after* Agency Review Pause (human provides feedback) back to Planner
    # The graph pauses *after* the agency_review_pause_node. When resumed,
    # it proceeds to the planner, which will then process the human_feedback.
    workflow.add_edge(AGENCY_REVIEW_NODE_NAME, PLANNER_NODE_NAME)

    # 3. Compile the graph
    # Use memory for persistence. Add interrupt points.
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=[AGENCY_REVIEW_NODE_NAME],
        interrupt_after=[AGENCY_REVIEW_NODE_NAME]
    )
    logger.info("Agent graph compiled successfully with checkpointer and interrupts.")
    return app
