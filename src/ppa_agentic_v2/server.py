# src/ppa_agentic_v2/server.py

import logging

from fastapi import FastAPI

from langserve import add_routes

from sse_starlette.sse import EventSourceResponse # Optional: For streaming UI updates if needed later

from pydantic import BaseModel, Field
from pydantic.generics import GenericModel
from typing import List, Dict, Any, Optional, TypeVar
import uuid # Added for thread ID management

# Ensure project root is in path if running server directly
# import sys
# import os
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if project_root not in sys.path:
#      sys.path.insert(0, project_root)

# Import necessary components from your agent code

from src.ppa_agentic_v2.agent import graph as app_agent
from src.ppa_agentic_v2.state import AgentState # Your Pydantic state model
from src.ppa_agentic_v2.config import logger, AGENCY_REVIEW_NODE_NAME, CUSTOMER_WAIT_NODE_NAME # Use configured logger and node names

# Import message types if needed for input schema
from langchain_core.messages import HumanMessage, BaseMessage

logger.info("Initializing FastAPI server for PPA Agent...")

# --- FastAPI App Setup ---
app_fastapi = FastAPI(
    title="PPA Insurance Agent Server",
    version="1.0",
    description="API server for the agentic PPA insurance quoting workflow (Milestone 2 - Mocked).",
)

# --- Input Schema for the API ---
# Define Pydantic models for expected input and output
class AgentInvokeInput(BaseModel):
    messages: List[Any] = Field(..., description="The list of messages in the conversation.")
    # Add other fields if needed, e.g., thread_id for stateful operations later

# --- Pydantic Models for Custom Endpoints --- #

class ReviewStatusResponse(BaseModel):
    thread_id: str
    awaiting_review: bool
    planned_action: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ReviewInput(BaseModel):
    approved: bool
    comment: Optional[str] = None
    # Potentially add corrected_args: Optional[Dict[str, Any]] = None if allowing edits

# --- LangServe Routes ---
logger.info("Adding LangServe routes...")
add_routes(
    app_fastapi,
    app_agent,
    path="/ppa_agent_v2",
    input_type=AgentInvokeInput, # Specify input type if needed
    output_type=AgentState, # Specify output type if needed
    enable_feedback_endpoint=True, # Optional: Enable feedback endpoint
    enable_public_trace_link_endpoint=True, # Optional: Enable public trace link endpoint
    config_keys=['configurable'], # Specify configurable fields
)
logger.info("LangServe routes added.")


# --- Add a root path for basic check ---
@app_fastapi.get("/")
async def read_root():
    return {"message": "PPA Agent Server is running. Navigate to /docs or /ppa_agent/playground"}

# --- Custom HITL Endpoints --- #

@app_fastapi.get("/threads/{thread_id}/status", response_model=ReviewStatusResponse)
async def get_thread_status(thread_id: str):
    """Checks if a given thread is currently paused awaiting agency review."""
    logger.info(f"Checking status for thread_id: {thread_id}")
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = await app_agent.aget_state(config)
        if not state:
            return ReviewStatusResponse(thread_id=thread_id, awaiting_review=False, error="Thread not found.")
        
        is_awaiting_review = state.next == (AGENCY_REVIEW_NODE_NAME,)
        planned_action = state.values.get("planned_tool_inputs") if is_awaiting_review else None
        
        logger.info(f"Thread {thread_id} status: Awaiting Review = {is_awaiting_review}")
        return ReviewStatusResponse(
            thread_id=thread_id,
            awaiting_review=is_awaiting_review,
            planned_action=planned_action
        )
    except Exception as e:
        logger.error(f"Error getting state for thread {thread_id}: {e}", exc_info=True)
        return ReviewStatusResponse(thread_id=thread_id, awaiting_review=False, error=str(e))

@app_fastapi.post("/threads/{thread_id}/review")
async def submit_review(thread_id: str, review: ReviewInput):
    """Submits human feedback for a paused thread and resumes the agent."""
    logger.info(f"Submitting review for thread_id: {thread_id} - Approved: {review.approved}")
    config = {"configurable": {"thread_id": thread_id}}
    try:
        # Get current state to verify it's awaiting review
        current_state = await app_agent.aget_state(config)
        if not current_state:
            raise HTTPException(status_code=404, detail="Thread not found.")

        # Simplified check: Assume if state exists, it *might* be waiting.
        # A more robust check would involve inspecting current_state.values['agent_scratchpad'] or a specific flag.
        # logger.debug(f"Current state for review check: {current_state.values}")

        feedback_content = f"Review Comment: {review.comment}" if review.comment else "No comment."
        human_feedback = f"HumanReviewDecision: {'Approved' if review.approved else 'Rejected'}\n{feedback_content}"

        # Update the state with the feedback. The graph should handle resuming.
        # The update_state method expects the *values* dictionary to update.
        await app_agent.update_state(config, values={'human_feedback': human_feedback})

        logger.info(f"Review submitted for thread {thread_id}. Agent should resume.")
        return {"message": "Review submitted successfully. Agent processing resumed."} # Updated return value

    except HTTPException as http_exc:
        logger.error(f"HTTP Exception submitting review for thread {thread_id}: {http_exc}", exc_info=True)
        return {"message": f"Error submitting review: {http_exc.detail}", "status": "error"}

# --- CORS Middleware --- #
# Allows requests from the playground, which is often served on a different port