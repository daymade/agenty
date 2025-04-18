from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging
import time
import random
import uuid
import requests # <-- Add dependency if needed: poetry add requests
import asyncio

# from .llm import get_llm_client # If tool needs LLM for generation
from .prompts import GENERATE_INFO_REQUEST_PROMPT_TEMPLATE # Reuse prompt

logger = logging.getLogger(__name__)

# --- Real Tools (Implementations based on original node logic) ---

class QuoteInitiateInput(BaseModel):
    # Define based on ACTUAL Mercury API requirements
    driver_name: str = Field(..., description="Primary driver's full name")
    driver_dob: str = Field(..., description="Primary driver's date of birth (YYYY-MM-DD)")
    address_line1: str = Field(..., description="Street address line 1")
    city: str = Field(...)
    state_code: str = Field(..., description="2-letter state code")
    zip_code: str = Field(..., description="5-digit zip code")
    # Add other required fields for Quote - Initiate API

@tool("quote_initiate_tool", args_schema=QuoteInitiateInput)
def quote_initiate_tool(**kwargs) -> Dict[str, Any]:
    """
    Initiates a new PPA quote session with the Mercury Insurance API.
    Requires essential customer and address details (name, dob, address).
    Returns the new quote ID and session context on success, or an error message on failure.
    Use this as the first step when starting a new quote after gathering initial info.
    """
    api_endpoint = "https://your-mercury-api-domain.com/api/v1/quote/initiate" # Replace with actual URL
    headers = {"Authorization": "Bearer YOUR_API_TOKEN", "Content-Type": "application/json"} # Replace with actual auth

    try:
        logger.info(f"Calling Mercury Quote Initiate API with: {kwargs}")
        # response = requests.post(api_endpoint, headers=headers, json=kwargs, timeout=30)
        # response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        # data = response.json()

        # --- MOCK API CALL for development ---
        if random.random() < 0.1: raise ConnectionError("Mock network error")
        time.sleep(1) # Simulate latency
        mock_quote_id = f"Q-{uuid.uuid4().hex[:8]}"
        mock_session = {"sessionId": f"S-{uuid.uuid4().hex[:12]}", "context": "abc"}
        data = {"quoteId": mock_quote_id, "session": mock_session}
        logger.info(f"Mock API Success: Quote Initiate. Response: {data}")
        # --- END MOCK API CALL ---

        # Validate response structure if needed
        quote_id = data.get("quoteId")
        session = data.get("session")
        if not quote_id or not session:
             raise ValueError("API response missing 'quoteId' or 'session'")

        return {
            "status": "success",
            "quote_id": quote_id,
            "session": session
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"API Network/HTTP Error calling Quote Initiate: {e}", exc_info=True)
        # Try to extract error details from response if possible
        error_detail = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json().get('message', e.response.text)
            except json.JSONDecodeError:
                error_detail = e.response.text
        return {"status": "error", "message": f"API Error: {error_detail}"}
    except (ValueError, KeyError) as e:
         logger.error(f"API Response Error (Quote Initiate): {e}", exc_info=True)
         return {"status": "error", "message": f"Invalid API response format: {e}"}
    except Exception as e:
        logger.error(f"Unexpected Error in quote_initiate_tool: {e}", exc_info=True)
        return {"status": "error", "message": f"Unexpected error: {e}"}

class AskCustomerInput(BaseModel):
    missing_fields: List[str] = Field(description="A list of specific information field names the customer needs to provide.")

@tool("ask_customer_tool", args_schema=AskCustomerInput)
def ask_customer_tool(missing_fields: List[str]) -> Dict[str, Any]:
    """
    Use this tool ONLY when specific information is missing and needs to be requested FROM THE CUSTOMER.
    Generates the text for a message asking the customer for the specified missing information fields.
    The message should be reviewed by a human agent before sending.
    Input must be a list of strings naming the missing fields (e.g., ['driver_age', 'vehicle_vin']).
    Output indicates success and contains the generated message content.
    """
    if not missing_fields:
        return {"status": "error", "message": "No missing fields specified for asking the customer."}

    # Reuse the prompt structure from the original node
    # Adapt the prompt if needed for better generation
    missing_info_str = ", ".join(missing_fields)
    prompt = GENERATE_INFO_REQUEST_PROMPT_TEMPLATE.format(missing_info_list=missing_info_str)
    logger.info(f"Generating customer request for fields: {missing_fields} using prompt.")
    logger.debug(f"Ask Customer Prompt:\n{prompt}")

    try:
        # Use a simple, dedicated LLM call for this generation, maybe not the main planner LLM
        # Or reuse the main one if configured appropriately.
        # llm_generator = get_llm_client(model="gemini-1.5-flash-latest") # Example: Use flash for generation
        # message_content = llm_generator.invoke(prompt).content

        # --- MOCK LLM Generation ---
        time.sleep(0.5)
        message_content = f"To help me provide an accurate quote, could you please share the following: {missing_info_str}?"
        logger.info(f"Mock LLM Generated message: '{message_content}'")
        # --- END MOCK ---

        if not message_content:
             raise ValueError("LLM returned empty content for customer message.")

        return {
            "status": "success",
            "message_content": message_content.strip(),
            "message_type": "info_request" # Consistent type
        }
    except Exception as e:
        logger.error(f"Error generating customer info request message: {e}", exc_info=True)
        # Fallback message
        message_content = f"I need a bit more information to proceed. Could you please provide details on: {missing_info_str}?"
        return {
            "status": "success", # Still generated *a* message
            "message_content": message_content,
            "message_type": "info_request",
            "error_in_generation": str(e)
        }

# --- Add other REAL tool implementations similarly ---
# e.g., add_vehicle_tool, update_driver_tool, rate_quote_tool...

# --- Keep some Mocks for now ---
@tool("mock_api_step_2")
def mock_api_step_2_tool() -> Dict[str, Any]:
    """MOCK: Simulates a generic second API step."""
    logger.info("Executing mock_api_step_2_tool")
    time.sleep(0.3)
    logger.info("MOCK Success: API Step 2 completed.")
    return {"status": "success", "step_2_data": "Some data from step 2"}

# --- Tool Registry ---
# Update with real tools, keep mocks for unimplemented ones
all_tools = [
    quote_initiate_tool,
    ask_customer_tool,
    mock_api_step_2_tool,
    # Add other real/mock tools as implemented
]

class ReadDocumentInput(BaseModel):
    file_path: str = Field(..., description="The path to the document file to read.")

@tool(args_schema=ReadDocumentInput)
async def read_document(file_path: str) -> str:
    """Reads the content of a specified document file."""
    logger.info(f"Executing read_document tool for path: {file_path}")
    try:
        # Use asyncio.to_thread for the blocking file operation
        content = await asyncio.to_thread(Path(file_path).read_text)
        logger.info(f"Successfully read content from {file_path}")
        # Limit content length for safety?
        # MAX_LEN = 5000 
        # if len(content) > MAX_LEN:
        #     content = content[:MAX_LEN] + "... [truncated]"
        return content
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return f"Error: File not found at path '{file_path}'."
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
        return f"Error: An unexpected error occurred while reading the file '{file_path}'."

all_tools.append(read_document)

# Map tool names to their functions for easy lookup
TOOL_MAP = {t.name: t for t in all_tools}