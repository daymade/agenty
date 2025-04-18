# src/ppa_agentic_v2/tools.py
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging
import time
import random

logger = logging.getLogger(__name__)

# --- Mock Tools for Milestone 2 ---

class MockToolInput(BaseModel):
    detail: str = Field(description="Some detail needed by the mock tool")

@tool("mock_quote_initiate", args_schema=MockToolInput)
def mock_quote_initiate_tool(detail: str) -> Dict[str, Any]:
    """MOCK: Simulates initiating a quote. Returns a mock quote ID."""
    logger.info(f"Executing mock_quote_initiate_tool with detail: {detail}")
    time.sleep(0.5) # Simulate work
    if random.random() < 0.1: # Simulate occasional failure
         logger.error("MOCK Failure: quote_initiate failed randomly.")
         return {"status": "error", "message": "Random API failure during quote initiation."}
    mock_id = f"MOCK_Q_{random.randint(1000, 9999)}"
    logger.info(f"MOCK Success: quote_initiate succeeded. Quote ID: {mock_id}")
    return {"status": "success", "quote_id": mock_id, "session_detail": f"Session for {detail}"}

class MockAskCustomerInput(BaseModel):
    missing_fields: List[str] = Field(description="List of fields to ask the customer about.")

@tool("mock_ask_customer", args_schema=MockAskCustomerInput)
def mock_ask_customer_tool(missing_fields: List[str]) -> Dict[str, Any]:
    """MOCK: Simulates generating a message asking the customer for info."""
    logger.info(f"Executing mock_ask_customer_tool for fields: {missing_fields}")
    if not missing_fields:
        logger.error("MOCK Failure: ask_customer called with no missing fields.")
        return {"status": "error", "message": "No fields specified to ask the customer."}
    message = f"MOCK MESSAGE: Please provide: {', '.join(missing_fields)}"
    logger.info(f"MOCK Success: Generated message: '{message}'")
    return {"status": "success", "message_content": message, "message_type": "info_request"}

@tool("mock_api_step_2")
def mock_api_step_2_tool() -> Dict[str, Any]:
    """MOCK: Simulates a generic second API step."""
    logger.info("Executing mock_api_step_2_tool")
    time.sleep(0.3)
    logger.info("MOCK Success: API Step 2 completed.")
    return {"status": "success", "step_2_data": "Some data from step 2"}

# --- Tool Registry ---
# Collect all defined tools
all_tools = [
    mock_quote_initiate_tool,
    mock_ask_customer_tool,
    mock_api_step_2_tool,
]