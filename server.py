import logging
import logging.config
import os
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union, List
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging from YAML file
try:
    with open('log_config.yaml', 'rt') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
    print("Logging configured successfully from log_config.yaml")
except Exception as e:
    print(f"Error loading logging configuration: {e}. Using basicConfig.")
    # Fallback to basicConfig if file loading fails
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Get the root logger (or a specific one if preferred)
logger = logging.getLogger()
logger.info(">>> Logging configured (check console/file based on config).")

# --- Import Real Agent ---
# Ensure src is in the Python path or adjust import accordingly
try:
    logger.info(">>> Attempting to import PPAAgent and AgentState...")
    from src.ppa_agent.agent import PPAAgent
    from src.ppa_agent.state import AgentState
    logger.info(">>> Imports successful.")
except ImportError as e:
    logger.error(f"Failed to import PPAAgent or AgentState: {e}. Check PYTHONPATH or imports.", exc_info=True)
    class MockPPAAgent:
        def process_email(self, email_content: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
            return {"thread_id": thread_id or str(uuid.uuid4()), "status": "mock_error_import", "messages": [], "requires_review": False}
        def resume_after_review(self, thread_id: str, decision: str, feedback: Optional[str] = None) -> Dict[str, Any]:
            return {"thread_id": thread_id, "status": "mock_error_import", "messages": [], "requires_review": False}

    AgentState = Dict[str, Any]
    PPAAgent = MockPPAAgent


# --- Instantiate Agent ---
try:
    # Ensure required environment variables are set (e.g., GOOGLE_API_KEY)
    logger.info(">>> Attempting to initialize PPAAgent...")
    agent = PPAAgent()
    logger.info(">>> PPAAgent initialization successful.")
except Exception as e:
    logger.error(f"Failed to initialize PPAAgent: {e}", exc_info=True)
    logger.warning("Falling back to MockPPAAgent due to initialization error.")
    agent = MockPPAAgent()


app = FastAPI(title="PPA Agent HITL V2 Server", version="1.0.0")

# --- Pydantic Models ---
class StartRequest(BaseModel):
    email_content: str = Field(..., example="Hi, I need a PPA quote for my car.")

class BaseResponse(BaseModel):
    thread_id: str = Field(..., example="a1b2c3d4e5f6")
    agent_response: Union[Dict[str, Any], str, None] = Field(..., example={"message": "Okay, I need your VIN."})
    status: str = Field(..., example="awaiting_information")
    review_required: bool = Field(..., example=False)
    step_requiring_review: Optional[str] = Field(None, example="generate_quote")

class StartResponse(BaseResponse):
    pass

class ProcessRequest(BaseModel):
    email_content: str = Field(..., example="My VIN is 123XYZ...")

class ProcessResponse(BaseResponse):
    pass

class ReviewRequest(BaseModel):
    decision: str = Field(..., example="accepted", pattern="^(accepted|rejected)$")
    feedback: Optional[str] = Field(None, example="Quote looks too high.")

class ReviewResponse(BaseResponse):
    pass

# --- Helper Function ---
def _extract_response_data(state: AgentState, thread_id: str) -> Dict[str, Any]:
    """Extracts relevant data from AgentState for API response.

    If review is required for the step that just completed, this function
    sets 'review_required' to True and 'agent_response' to None,
    preventing the agent's generated content from being shown before approval.
    """
    agent_response: Union[Dict[str, Any], str, None] = None
    messages = state.get("messages", [])
    # Check if the step that just ran requires review
    step_review = state.get("step_requiring_review")
    # Determine if review is currently pending based on whether step_requiring_review is set
    review_is_currently_required = bool(step_review)

    logger.debug(f"Extracting response for {thread_id}. Step requiring review: '{step_review}', Review currently required: {review_is_currently_required}")

    if review_is_currently_required:
        # If review is required for the step that just ran, DO NOT return the agent's generated content.
        # The agency must call /review first.
        agent_response = None # Set to None (or could be a placeholder message)
        logger.info(f"Thread {thread_id}: Review required for step '{step_review}'. Hiding agent content.")
    elif messages:
        # If no review is required, return the last message from the agent.
        last_agent_message = next((msg for msg in reversed(messages) if msg.get("role") == "agent"), None)
        if last_agent_message:
            agent_response = last_agent_message.get("content")
            logger.info(f"Thread {thread_id}: No review required. Response is last agent message content.")
        else:
             logger.warning(f"Thread {thread_id}: No agent message found in state messages. Setting agent_response to None.")
             agent_response = None
    else:
        logger.warning(f"Thread {thread_id}: No messages found in state. Setting agent_response to None.")
        agent_response = None

    return {
        "thread_id": thread_id,
        "agent_response": agent_response, # Will be None if review is required
        "status": state.get("status", "unknown"),
        "review_required": review_is_currently_required, # Correctly reflects if review is pending
        "step_requiring_review": step_review,
    }


# --- API Endpoints ---

@app.post("/threads/start", response_model=StartResponse, tags=["Threads"])
async def start_thread(request: StartRequest):
    logger.info(f"Received /threads/start request for email: {request.email_content[:50]}...")
    try:
        final_state = agent.process_email(
            email_content=request.email_content
        )
        returned_thread_id = final_state.get("thread_id")
        if not returned_thread_id:
             logger.error("Agent did not return a thread_id in the state.")
             raise HTTPException(status_code=500, detail="Agent failed to provide a thread ID.")

        logger.info(f"Agent processing complete for new thread {returned_thread_id}. Final state status: {final_state.get('status')}")
        response_data = _extract_response_data(final_state, returned_thread_id)
        return StartResponse(**response_data)
    except Exception as e:
        logger.error(f"Error processing /threads/start: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/threads/{thread_id}/process", response_model=ProcessResponse, tags=["Threads"])
async def process_email_in_thread(thread_id: str, request: ProcessRequest):
    """
    Processes a subsequent customer email within an existing conversation thread.
    The agent updates the state based on the new email and returns the current status.
    """
    logger.info(f"Received /threads/{thread_id}/process request")
    try:
        final_state = agent.process_email(
            email_content=request.email_content,
            thread_id=thread_id
        )

        # Check for the specific error status returned by the modified agent
        if final_state.get("status") == "error_during_invoke":
            logger.error(f"Workflow invoke failed for thread {thread_id}. Returning error state.")
            # Return 500, but include the detailed error state in the response body
            raise HTTPException(status_code=500, detail=final_state)

        logger.info(f"Agent processing complete for thread {thread_id}. Final state status: {final_state.get('status')}")
        response_data = _extract_response_data(final_state, thread_id)
        return ProcessResponse(**response_data)

    except ValueError as ve:
        logger.warning(f"Value error processing thread {thread_id}: {ve}")
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Error processing /threads/{thread_id}/process: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/threads/{thread_id}/review", response_model=ReviewResponse, tags=["Threads"])
async def submit_review(thread_id: str, request: ReviewRequest):
    """
    Submits a human agent's review decision ('accepted' or 'rejected')
    for a step that required review. The agent resumes processing based
    on the decision and returns the subsequent state.
    """
    logger.info(f"Received /threads/{thread_id}/review request with decision: {request.decision}")
    try:
        final_state = agent.resume_after_review(
            thread_id=thread_id,
            decision=request.decision,
            feedback=request.feedback
        )
        logger.info(f"Agent resume complete for thread {thread_id}. Final state status: {final_state.get('status')}")
        response_data = _extract_response_data(final_state, thread_id)
        return ReviewResponse(**response_data)
    except ValueError as e:
        logger.warning(f"Value error processing review for thread {thread_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing /threads/{thread_id}/review: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# --- Root endpoint for basic check ---
@app.get("/", tags=["Status"])
async def read_root():
    """Basic health check endpoint."""
    return {"status": "PPA Agent HITL V2 Server is running"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server via __main__ (for testing only)")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True, log_config="log_config.yaml")
