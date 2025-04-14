"""Integration tests for the PPA Agent."""

import json
import os
import time # Potentially needed for delays between API calls
from pathlib import Path
from typing import Dict, Any

import pytest
from dotenv import load_dotenv

# Removed LangChain import
# from langchain_core.messages import AIMessage

from src.ppa_agent.agent import PPAAgent
from src.ppa_agent.config import GEMINI_MODEL_NAME, OPENAI_MODEL_NAME

# Load environment variables from .env file
load_dotenv()

# Define output directory for saving test results
OUTPUT_DIR = Path("test_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


@pytest.fixture
def openai_agent() -> PPAAgent:
    """Create an OpenAI agent instance with real API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in environment")
    return PPAAgent(provider="openai", model=OPENAI_MODEL_NAME)


@pytest.fixture
def gemini_agent() -> PPAAgent:
    """Create a Gemini agent instance with real API key."""
    # Check for API key in environment for integration tests
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not found in environment")
    # Pass provider and model, agent init will handle API key from env
    return PPAAgent(
        provider="gemini",
        model=GEMINI_MODEL_NAME,
    )

def save_test_output(test_name: str, state: Dict[str, Any]):
    """Helper to save final state to a JSON file."""
    output_path = OUTPUT_DIR / f"{test_name}_state.json"
    with open(output_path, "w") as f:
        # Use default=str to handle non-serializable types like datetime if they sneak in
        json.dump(state, f, indent=2, default=str)
    print(f"Saved final state for {test_name} to {output_path}")

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_complete_workflow_openai(openai_agent: PPAAgent) -> None:
    """Test complete workflow with OpenAI (single turn)."""
    test_name = "complete_openai"
    email = """
    Hi, I'm interested in getting auto insurance for my new car.
    My name is John Smith, I'm 22 years old, and I'm a full-time student
    with a 3.8 GPA. I live at 123 Main St, San Francisco, CA.

    I'm looking to insure my 2022 Tesla Model 3, which has all the latest
    safety features. I also have a clean driving record.

    I'm also interested in bundling this with renters insurance if possible.

    Thanks,
    John
    """

    state = openai_agent.process_email(email)
    save_test_output(test_name, state)

    # Verify basic state info extracted
    assert state["customer_email"] == email
    assert state["customer_info"].get("driver_name") == "John Smith"
    # LLM might interpret age differently (e.g., int vs str), allow flexibility
    assert str(state["customer_info"].get("driver_age", "")) == "22"
    assert state["customer_info"].get("vehicle_make") == "Tesla"
    assert state["customer_info"].get("vehicle_model") == "Model 3"
    assert str(state["customer_info"].get("vehicle_year", "")) == "2022"
    assert state["customer_info"].get("address") is not None # Address extraction can vary

    # Verify workflow progressed to quote generation
    assert not state.get("missing_info"), "Missing info should be empty for complete workflow"
    assert state.get("status") == "quote_generated", "Final status should be quote_generated"
    assert state.get("quote_data") is not None, "Quote data should be generated"
    assert state.get("quote_ready") is True, "Quote ready flag should be set"

    # Verify agency review message was generated
    messages = state.get("messages", [])
    review_msg = next(
        (msg for msg in messages if msg.get("type") == "quote_summary_for_review"), None
    )
    assert review_msg is not None, "Quote summary for review message not found"


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
def test_complete_workflow_gemini(gemini_agent: PPAAgent) -> None:
    """Test complete workflow with Gemini (single turn)."""
    test_name = "complete_gemini"
    email = """
    Hello, I need car insurance for my vehicle.
    I'm Sarah Johnson, 35 years old, living at 456 Oak Ave, Seattle, WA.

    I drive a 2021 Honda CR-V with advanced safety features and parking
    sensors. I've been driving for 15 years with no accidents.

    Looking forward to your quote.
    Best regards,
    Sarah
    """
    state = gemini_agent.process_email(email)
    save_test_output(test_name, state)

    # Verify basic state info extracted
    assert state["customer_email"] == email
    assert state["customer_info"].get("driver_name") == "Sarah Johnson"
    assert str(state["customer_info"].get("driver_age", "")) == "35"
    assert state["customer_info"].get("vehicle_make") == "Honda"
    assert state["customer_info"].get("vehicle_model") == "CR-V"
    assert str(state["customer_info"].get("vehicle_year", "")) == "2021"
    assert state["customer_info"].get("address") is not None

    # Verify workflow progressed to discount check / review
    assert not state.get("missing_info"), "Missing info should be empty"
    assert state.get("status") == "ready_for_review", f"Unexpected final status: {state.get('status')}"
    assert state.get("requires_review") is True, "Should require review after discount check"


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_incomplete_info_workflow(openai_agent: PPAAgent) -> None:
    """Test workflow with incomplete information (single turn)."""
    test_name = "incomplete_openai"
    email = "Hi, I want car insurance. My name is Mike Brown."

    state = openai_agent.process_email(email)
    save_test_output(test_name, state)

    # Verify basic state
    assert state["customer_email"] == email
    assert state["customer_info"].get("driver_name") == "Mike Brown"
    assert state.get("missing_info"), "Missing info list should not be empty"
    # Check specific missing items (LLM might vary slightly, check for most likely)
    assert "driver_age" in state["missing_info"]
    assert "vehicle_make" in state["missing_info"]
    assert "address" in state["missing_info"]

    # Check workflow status and messages
    assert state.get("status") == "ready_for_review", f"Expected ready_for_review, got {state.get('status')}"
    assert state.get("requires_review") is True, "Requires review flag should be set"
    messages = state.get("messages", [])
    info_request_msg = next(
        (msg for msg in messages if msg.get("type") == "info_request"), None
    )
    assert info_request_msg is not None, "Info request message not found"
    review_msg = next(
        (msg for msg in messages if msg.get("type") == "review_summary"), None
    )
    assert review_msg is not None, "Review summary message not found"


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
def test_non_ppa_workflow(gemini_agent: PPAAgent) -> None:
    """Test workflow with non-PPA inquiry (single turn)."""
    test_name = "non_ppa_gemini"
    email = """
    Hi, I'm interested in getting homeowners insurance for my new house.
    Can you help me with that?
    """

    state = gemini_agent.process_email(email)
    save_test_output(test_name, state)

    # Verify state after early exit
    assert state["customer_email"] == email
    # LOB identification might be HOME or OTHER depending on LLM
    assert state.get("lob") in ["HOME", "OTHER"], f"Unexpected LOB: {state.get('lob')}"
    # Check that the workflow ended, status might be 'new' or 'error' if routing failed early
    # Or it might have gone through review prep? Let's check messages.
    assert state.get("status") != "quote_generated", "Quote should not be generated for non-PPA"
    # Check that no quote data was generated
    assert state.get("quote_data") is None
    # Check messages - should not contain quote summary or PPA info request
    messages = state.get("messages", [])
    assert not any(msg.get("type") == "quote_summary_for_review" for msg in messages)
    assert not any(msg.get("type") == "info_request" for msg in messages)


# --- Multi-Turn Test ---
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
def test_multi_turn_info_request_gemini(gemini_agent: PPAAgent) -> None:
    """Test multi-turn workflow: incomplete info -> request -> provide info -> review."""
    test_name = "multi_turn_gemini"
    # Turn 1: Initial email missing vehicle model
    email_1 = """
    Subject: Need a car insurance quote!

    Hi there,
    Please provide a quote for my 2022 Subaru.
    My name is Alice Williams, age 40.
    I live at 789 Pine St, Portland, OR.
    Thanks!
    """
    print("\n--- Multi-Turn Test: Turn 1 ---")
    state_1 = gemini_agent.process_email(email_1)
    thread_id = state_1.get("thread_id")
    print(f"Turn 1 State (Thread ID: {thread_id}):\n{json.dumps(state_1, indent=2, default=str)}")
    save_test_output(f"{test_name}_turn1", state_1)

    assert thread_id is not None, "Thread ID should be generated"
    assert state_1.get("status") == "ready_for_review", f"Turn 1: Status should be ready_for_review, got {state_1.get('status')}"
    assert "vehicle_model" in state_1.get("missing_info", []), "Turn 1: Vehicle model should be missing"
    messages_1 = state_1.get("messages", [])
    info_request_msg = next((msg for msg in messages_1 if msg.get("type") == "info_request"), None)
    assert info_request_msg is not None, "Turn 1: Agent should have generated an info request message"
    assert state_1.get("requires_review") is True, "Turn 1: State should require review"

    time.sleep(1)

    # Turn 2: Customer replies with the missing vehicle model
    email_2 = "Sorry about that, it's an Outback Wilderness model."
    print("\n--- Multi-Turn Test: Turn 2 ---")
    state_2 = gemini_agent.process_email(email_2, thread_id=thread_id)
    print(f"Turn 2 State:\n{json.dumps(state_2, indent=2, default=str)}")
    save_test_output(f"{test_name}_turn2", state_2)

    assert state_2.get("thread_id") == thread_id, "Turn 2: Thread ID should match"
    assert state_2.get("customer_email") == email_2, "Turn 2: Latest customer email should be updated"

    # Verify information is now complete
    customer_info_2 = state_2.get("customer_info", {})
    assert customer_info_2.get("driver_name") == "Alice Williams", "Turn 2: Driver name should persist"
    assert customer_info_2.get("vehicle_make") == "Subaru", "Turn 2: Vehicle make should persist"
    assert str(customer_info_2.get("vehicle_year", "")) == "2022", "Turn 2: Vehicle year should persist"
    assert customer_info_2.get("vehicle_model") == "Outback Wilderness", "Turn 2: Vehicle model should be updated"
    assert not state_2.get("missing_info"), "Turn 2: Missing info list should be empty after reply"

    # Verify workflow proceeded to review after discount check
    assert state_2.get("status") == "ready_for_review", f"Turn 2: Unexpected final status: {state_2.get('status')}"
    assert state_2.get("requires_review") is True, "Turn 2: Should require review after discount check"
    assert state_2.get("quote_data") is None, "Turn 2: Quote data should not be generated yet"
    assert state_2.get("quote_ready") is False, "Turn 2: Quote ready flag should be false"
