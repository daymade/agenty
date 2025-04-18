# src/ppa_agentic_v2/server.py

import logging

from fastapi import FastAPI

from langserve import add_routes

from sse_starlette.sse import EventSourceResponse # Optional: For streaming UI updates if needed later

from pydantic import BaseModel, Field
from pydantic.generics import GenericModel
from typing import List, Dict, Any, Optional, TypeVar

# Ensure project root is in path if running server directly
# import sys
# import os
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if project_root not in sys.path:
#      sys.path.insert(0, project_root)

# Import necessary components from your agent code

from src.ppa_agentic_v2.agent import build_agent_graph # Function that returns the compiled app

from src.ppa_agentic_v2.state import AgentState # Your Pydantic state model

from src.ppa_agentic_v2.config import logger # Use configured logger

# Import message types if needed for input schema
from langchain_core.messages import HumanMessage, BaseMessage

logger.info("Initializing FastAPI server for PPA Agent...")

# --- FastAPI App Setup ---
app_fastapi = FastAPI(
    title="PPA Insurance Agent Server",
    version="1.0",
    description="API server for the agentic PPA insurance quoting workflow (Milestone 2 - Mocked).",
)

# --- Agent Graph Initialization ---
logger.info("Building agent graph...")
app_agent = build_agent_graph()
logger.info("Agent graph built.")

# --- Input Schema for the API ---
# Define Pydantic models for expected input and output
class AgentInvokeInput(BaseModel):
    messages: List[Any] = Field(..., description="The list of messages in the conversation.")
    # Add other fields if needed, e.g., thread_id for stateful operations later

# --- LangServe Routes ---
logger.info("Adding LangServe routes...")
add_routes(
    app_fastapi,
    app_agent,
    path="/ppa_agent",
    input_type=AgentInvokeInput, # Specify input type if needed
    output_type=AgentState, # Specify output type if needed
    enable_feedback_endpoint=True, # Optional: Enable feedback endpoint
    enable_public_trace_link_endpoint=True, # Optional: Enable public trace link endpoint
    # playground_type="chat", # Use 'chat' playground type
    # config_keys=["configurable"], # Example: Define configurable fields
)
logger.info("LangServe routes added.")


# --- Add a root path for basic check ---
@app_fastapi.get("/")
async def read_root():
    return {"message": "PPA Agent Server is running. Navigate to /docs or /ppa_agent/playground"}

# --- CORS Middleware ---
# Allows requests from the playground, which is often served on a different port