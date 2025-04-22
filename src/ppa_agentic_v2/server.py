# src/ppa_agentic_v2/server.py

import logging

from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from langserve import add_routes
from sse_starlette.sse import EventSourceResponse # Optional: For streaming UI updates if needed later

from pydantic import BaseModel, Field
from pydantic.generics import GenericModel
from typing import List, Optional, Dict, Any, TypeVar
import uuid # Added for thread ID management

# Ensure project root is in path if running server directly
# import sys
# import os
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if project_root not in sys.path:
#      sys.path.insert(0, project_root)

# Import necessary components from your agent code

from src.ppa_agentic_v2.agent import build_agent_graph, all_tools # Import graph builder and tools
from src.ppa_agentic_v2.state import AgentState # Your Pydantic state model
from src.ppa_agentic_v2.config import logger, AGENCY_REVIEW_NODE_NAME, CUSTOMER_WAIT_NODE_NAME, SQLITE_DB_NAME # Use configured logger and node names

# Import message types if needed for input schema
from langchain_core.messages import HumanMessage, BaseMessage

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver # Import AsyncSqliteSaver

logger.info("Initializing FastAPI server for PPA Agent...")

# --- Lifespan Management for Agent and Checkpointer --- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize checkpointer and compile graph
    logger.info("FastAPI Lifespan: Startup sequence starting...")
    async_memory_manager = AsyncSqliteSaver.from_conn_string(SQLITE_DB_NAME)
    async with async_memory_manager as saver:
        logger.info(f"AsyncSqliteSaver instance obtained: {saver}")

        # Build and compile the graph (compilation likely happens inside build_agent_graph)
        compiled_app = build_agent_graph()
        logger.info("Agent graph built (already compiled).")

        # Attach the checkpointer to the compiled graph
        compiled_app.checkpointer = saver
        logger.info(f"Attached checkpointer to compiled graph: {compiled_app.checkpointer}")

        # Store the compiled app in app state for routes to access
        app.state.compiled_app = compiled_app
        logger.info("Compiled agent graph stored in app state.")

        # Add LangServe routes HERE, passing the compiled app
        add_routes(
            app,
            app.state.compiled_app, # Pass the directly compiled app
            path="/ppa_agent_v2",
            input_type=AgentInvokeInput,
            output_type=AgentState,
            enable_feedback_endpoint=True,
            enable_public_trace_link_endpoint=True,
            config_keys=['configurable'],
        )
        logger.info("LangServe routes added within lifespan.")

        yield
    # Shutdown: Cleanup (handled by async with)
    logger.info("FastAPI Lifespan: Shutdown sequence completed.")

# --- FastAPI App Setup --- #
app_fastapi = FastAPI(
    title="PPA Agentic V2 Server",
    version="1.0",
    description="FastAPI server exposing PPA Agentic V2 via LangServe and custom HITL endpoints.",
    lifespan=lifespan # <-- Register lifespan handler
)

# --- Input Schema for the API --- #
# (Keep AgentInvokeInput as is, LangServe uses it for the runnable's input)
class AgentInvokeInput(BaseModel):
    # The input to the compiled graph's invoke/stream methods.
    # This should match the structure expected by the first node in your graph
    # after START. Often, this is the AgentState or a subset of it.
    # Let's assume it expects the full AgentState for now, or parts of it.
    # If your graph starts by expecting just `messages`, adjust accordingly.
    messages: List[Any] = Field(..., description="The list of messages in the conversation.")
    # You might need other initial state fields depending on how your graph starts
    # For example: goal: Optional[str] = None

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

# --- Custom Endpoints for HITL --- #

@app_fastapi.get("/threads/{thread_id}/status")
async def get_thread_status(request: Request, thread_id: str):
    logger.info(f"GET /threads/{thread_id}/status")
    compiled_app = request.app.state.compiled_app # Get compiled app from state
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = await compiled_app.aget_state(config) # Use compiled app
        logger.info(f"Current state for thread {thread_id}: {state.values}")
        if not state:
            return ReviewStatusResponse(thread_id=thread_id, awaiting_review=False, error="Thread not found.")

        # Check for pending review using the dedicated state field
        action_details_pending = state.values.get("action_pending_review")
        is_awaiting_review = action_details_pending is not None

        # Extract action details from the correct field if awaiting review
        planned_action = action_details_pending if is_awaiting_review else None

        logger.info(f"Thread {thread_id} status: Awaiting Review = {is_awaiting_review}, Action = {planned_action}")
        return ReviewStatusResponse(
            thread_id=thread_id,
            awaiting_review=is_awaiting_review,
            planned_action=planned_action
        )
    except Exception as e:
        logger.error(f"Error getting state for thread {thread_id}: {e}", exc_info=True)
        return ReviewStatusResponse(thread_id=thread_id, awaiting_review=False, error=str(e))

@app_fastapi.post("/threads/{thread_id}/review")
async def submit_review(request: Request, thread_id: str, review: ReviewInput):
    logger.info(f"POST /threads/{thread_id}/review - Data: {review.dict()}")
    compiled_app = request.app.state.compiled_app # Get compiled app from state
    config = {"configurable": {"thread_id": thread_id}}
    try:
        # 1. Get the current state to check if review is expected
        current_state = await compiled_app.aget_state(config) # Use compiled app
        if not current_state:
            logger.error(f"Thread {thread_id} not found for review.")
            raise HTTPException(status_code=404, detail="Thread not found.")

        # Simplified check: Assume if state exists, it *might* be waiting.
        # A more robust check would involve inspecting current_state.values['agent_scratchpad'] or a specific flag.
        # logger.debug(f"Current state for review check: {current_state.values}")

        # Construct the human_feedback dictionary to match AgentState definition
        human_feedback = {
            "approved": review.approved,
            "comment": review.comment if review.comment is not None else "",
            # Potentially add "edited_inputs": review.edited_inputs if review.edited_inputs else None
        }
        logger.info(f"Prepared human_feedback dict: {human_feedback}")

        # Update the state with the feedback dictionary directly.
        await compiled_app.aupdate_state(config, {'human_feedback': human_feedback}) # <-- Use compiled app

        logger.info(f"Review submitted for thread {thread_id}. Triggering agent resumption...")
        # Trigger the graph to resume processing using the updated state
        # Pass None as input, hoping it picks up the state from the checkpointer
        # Note: This will run the graph synchronously within this request until the next interrupt/end.
        await compiled_app.ainvoke(None, config) # <-- Use compiled app
        logger.info(f"Agent processing resumed and completed until next interrupt/end for thread {thread_id}.")

        return {"message": "Review submitted successfully. Agent processing resumed."}

    except HTTPException as http_exc:
        logger.error(f"HTTP error submitting review for thread {thread_id}: {http_exc.detail}", exc_info=False)
        # Propagate HTTP exceptions correctly
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error submitting review for thread {thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during review submission: {e}")

# --- CORS Middleware --- #
# Allows requests from the playground, which is often served on a different port
from fastapi.middleware.cors import CORSMiddleware

app_fastapi.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add a basic root endpoint for testing
@app_fastapi.get("/")
async def read_root():
    return {"message": "PPA Agentic V2 Server is running"}