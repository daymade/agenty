import os
import sys
import pytest
from dotenv import load_dotenv

# --- Path Hack to find the src directory ---
# Get the absolute path of the directory containing the current file (tests/)
tests_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root
project_root = os.path.dirname(tests_dir)
# Construct the path to the src directory
src_path = os.path.join(project_root, 'src')
# Insert the src path at the beginning of sys.path
if src_path not in sys.path:
    sys.path.insert(0, src_path)
# --- End Path Hack ---

from ppa_agent.agent import PPAAgent
from ppa_agent.state import AgentState

# Load environment variables (especially GEMINI_API_KEY)
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SKIP_TEST = not GEMINI_API_KEY
REASON = "GEMINI_API_KEY not found in environment variables"


@pytest.fixture(scope="module")
def gemini_agent():
    """Fixture to initialize the PPAAgent with Gemini provider."""
    if SKIP_TEST:
        pytest.skip(REASON)
    # Assuming default Gemini configuration is sufficient
    return PPAAgent(provider="gemini")


@pytest.mark.integration
@pytest.mark.skipif(SKIP_TEST, reason=REASON)
def test_hitl_v2_accept_flow(gemini_agent: PPAAgent):
    """Tests the HITL v2 flow with accept decisions using the Gemini API."""
    initial_email = "Hi, I'd like a quote for my car. It's a Toyota Camry."
    thread_id = None

    # --- Turn 1: Initial Email -> Analyze Info Review ---
    print("\n--- Turn 1: Processing initial email ---")
    state1 = gemini_agent.process_email(email_content=initial_email, thread_id=thread_id)
    thread_id = state1.get("thread_id") # Get the assigned thread_id
    print(f"State after initial process (expecting analyze_information review): {state1}")

    assert state1.get("step_requiring_review") == "analyze_information", \
        f"Expected review step 'analyze_information', got {state1.get('step_requiring_review')}"
    assert state1.get("data_for_review") is not None, "Expected data_for_review to be populated"

    # --- Turn 2: Accept Analyze Info -> Generate Info Request Review ---
    print("\n--- Turn 2: Resuming after accepting analyze_information ---")
    state2 = gemini_agent.resume_after_review(thread_id=thread_id, decision="accepted")
    print(f"State after resuming (expecting generate_info_request review): {state2}")

    assert state2.get("step_requiring_review") == "generate_info_request", \
        f"Expected review step 'generate_info_request', got {state2.get('step_requiring_review')}"
    assert state2.get("data_for_review") is not None, "Expected data_for_review to be populated for info request"
    assert "generated_info_request_message" in state2.get("data_for_review", {}), \
        "Expected generated message in review data"

    # --- Turn 3: Accept Generate Info Request -> Final State (Info Requested) ---
    print("\n--- Turn 3: Resuming after accepting generate_info_request ---")
    state3 = gemini_agent.resume_after_review(thread_id=thread_id, decision="accepted")
    print(f"State after final resume: {state3}")

    assert state3.get("step_requiring_review") is None, \
        f"Expected no review step after accepting info request, got {state3.get('step_requiring_review')}"
    # Check if the final message in the *state* indicates an info request was the last action
    # Note: The actual email sending is outside this agent's scope
    final_messages = state3.get("messages", [])
    assert final_messages, "Expected messages in the final state"
    assert final_messages[-1].get("type") == "info_request", \
        f"Expected last message type to be 'info_request', got {final_messages[-1].get('type')}"
    assert "year" in final_messages[-1].get("content", "").lower(), \
        "Expected info request content to ask about missing info (e.g., year)"
    # Status might vary, but should indicate waiting for customer info
    # assert state3.get("status") == "info_requested" # Or similar status reflecting waiting

    print("\nIntegration test test_hitl_v2_accept_flow completed successfully.")
