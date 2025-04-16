"""
Script to run a detailed PPA Agent HITL v2 workflow simulation 
and generate an HTML visualization.
"""

import json
import os
import logging
from datetime import datetime
from ppa_agent.agent import PPAAgent
from ppa_agent.state import AgentState
from typing import Dict, List
from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound
import traceback

# --- Configuration ---
LOG_LEVEL = logging.INFO
# Use absolute path for output directory
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "visualizations"))
OUTPUT_FILENAME = "hitl_v2_flow_detailed.html"
TEMPLATE_PATH = "templates/hitl_v2_visualization_template_fixed.html" # Using fixed template

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Simulation Scenario ---

initial_customer_email = "Hi, I'd like a quote for my car. It's a Toyota Camry."

customer_reply_with_info = ( 
    "Okay, here is the info:\n" 
    "Name: Bob Smith\n" 
    "Age: 35\n" 
    "Vehicle Year: 2022\n" 
    "Address: 123 Main St, Anytown, CA 90210\n"
    "Also, I have a good student discount proof, how can I provide it?"
)

customer_reply_with_discount = "Here is the attached proof of my good student discount."


def run_simulation():
    """Runs the multi-turn workflow simulation and captures data."""
    logger.info("Starting HITL v2 workflow simulation...")

    timeline = [] # Use a list for timeline events
    agent = PPAAgent(provider="gemini") # Use the real agent
    thread_id = None
    final_state = {}

    def add_timeline_event(event_type: str, description: str, actor: str = None, data: Dict = None, state: AgentState = None):
        timeline.append({
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "actor": actor,
            "description": description,
            "data": data.copy() if data else None,
            "state": state.copy() if state else None, # Store a copy of the state snapshot
        })

    try:
        # --- Turn 1: Initial Email -> Analyze Info Review ---
        logger.info("--- Turn 1: Processing initial email ---")
        event_data = {"email_content": initial_customer_email}
        state1 = agent.process_email(email_content=initial_customer_email, thread_id=thread_id)
        thread_id = state1.get("thread_id")
        add_timeline_event("Customer Input", "Customer sends initial email", actor="Customer", data=event_data, state=state1)
        logger.info(f"State after Turn 1 (expecting generate_info_request review): {state1.get('step_requiring_review')}")
        assert state1.get("step_requiring_review") == "generate_info_request", f"Expected generate_info_request review, got {state1.get('step_requiring_review')}"
        final_state = state1

        # --- Turn 2: Accept Analyze Info -> Generate Info Request Review ---
        logger.info("--- Turn 2: Agency accepts info analysis ---")
        logger.info("--- Turn 2: Agency accepts generate_info_request ---")
        event_data = {"decision": "accepted", "step": "generate_info_request"}
        state2 = agent.resume_after_review(thread_id=thread_id, decision="accepted")
        add_timeline_event("Agency Decision", "Agency accepts initial info analysis", actor="Agency", data=event_data, state=state2)
        add_timeline_event("Agency Decision", "Agency accepts generate_info_request email", actor="Agency", data=event_data, state=state2)
        logger.info(f"State after Turn 2 (expecting END state, waiting for customer): {state2.get('step_requiring_review')}")
        assert state2.get("step_requiring_review") is None, f"Expected no review step after accepting info request, got {state2.get('step_requiring_review')}"
        final_state = state2

        # --- Turn 3: Customer provides info -> Analyze Info -> Check Discounts (NO REVIEW) ---
        logger.info("--- Turn 3: Processing customer reply with info ---")
        event_data = {"email_content": customer_reply_with_info}
        state3 = agent.process_email(email_content=customer_reply_with_info, thread_id=thread_id)
        add_timeline_event("Customer Input", "Customer provides requested info & discount query", actor="Customer", data=event_data, state=state3)
        logger.info(f"State after Turn 3 (expecting NO review, discount status: {state3.get('discount_status')}): {state3.get('step_requiring_review')}")
        assert state3.get("step_requiring_review") is None, f"Expected NO review step after customer info + discount check, got {state3.get('step_requiring_review')}"
        assert state3.get("discount_status") == "proof_needed", f"Expected discount_status 'proof_needed', got {state3.get('discount_status')}"
        assert any(m.get('type') == 'discount_query' for m in state3.get('messages', [])), "Expected a discount_query message in state3"
        final_state = state3

        # --- Turn 4: Customer provides discount proof -> Process Response -> Check Discounts (NO REVIEW) -> Generate Quote Review ---
        logger.info("--- Turn 4: Customer provides proof ---")
        customer_proof_email = "Here is the attached proof of my good student discount."
        state4 = agent.process_email(email_content=customer_proof_email, thread_id=thread_id)
        logger.info(f"State4 received directly from process_email: {state4}")
        review_step = state4.get("step_requiring_review")
        logger.info(f"State after Turn 4 (expecting generate_quote review): {review_step}")
        add_timeline_event("Customer Input", "Customer provides discount proof", actor="Customer", data={"email_content": customer_proof_email}, state=state4)
        logger.info(f"State after Turn 4 (expecting generate_quote review): {state4.get('step_requiring_review')}")
        assert state4.get("step_requiring_review") == "generate_quote", f"Expected generate_quote review, got {state4.get('step_requiring_review')}"
        final_state = state4

        add_timeline_event("Agent Action", "Agent prepares quote for review", actor="Agent", data=event_data, state=state4)
        logger.info(f"State after Turn 4 (expecting generate_quote review): {state4.get('step_requiring_review')}")
        assert state4.get("step_requiring_review") == "generate_quote", f"Expected generate_quote review, got {state4.get('step_requiring_review')}"
        final_state = state4 # Keep state before review for the next step

        # --- Turn 5: Agency REJECTS generated quote (requesting change) ---
        logger.info("--- Turn 5: Agency rejects quote (requesting change) ---")
        reject_feedback = "Liability limits seem too low, please increase to 100/300."
        event_data = {"decision": "rejected", "step": "generate_quote", "feedback": reject_feedback}
        state5 = agent.resume_after_review(thread_id=thread_id, decision="rejected", feedback=reject_feedback)
        add_timeline_event("Agency Decision", "Agency rejects generated quote with feedback", actor="Agency", data=event_data, state=state5)
        logger.info(f"State after Turn 5 (expecting generate_quote review again): {state5.get('step_requiring_review')}")
        # After rejection, it should loop back to generate_quote and prepare_quote_review again
        assert state5.get("step_requiring_review") == "generate_quote", f"Expected generate_quote review after rejection, got {state5.get('step_requiring_review')}"
        final_state = state5 # Keep state before the second review

        # --- Turn 6: Agency ACCEPTS the RE-GENERATED quote ---
        logger.info("--- Turn 6: Agency accepts the re-generated quote ---")
        event_data = {"decision": "accepted", "step": "generate_quote"}
        state6 = agent.resume_after_review(thread_id=thread_id, decision="accepted") # Resume after the second quote review
        add_timeline_event("Agency Decision", "Agency accepts re-generated quote", actor="Agency", data=event_data, state=state6)
        logger.info(f"State after Turn 6 (expecting END state, quote generated): {state6.get('status')}")
        assert state6.get("step_requiring_review") is None, f"Expected no review step after accepting regenerated quote, got {state6.get('step_requiring_review')}"
        assert state6.get('status') == 'quote_generated', f"Expected status 'quote_generated', got {state6.get('status')}"
        final_state = state6 # Final state after successful completion

        # --- Turn 7: Simulate sending the final quote (No agent interaction) ---
        logger.info("--- Turn 7: Simulate sending the final quote ---")
        # Simulate the agent sending the final quote email based on the accepted state
        # This part is just for visualization and doesn't involve agent.resume_after_review
        quote_data = final_state.get('quote_data', {})
        premium = quote_data.get('premium', 'N/A')
        coverage = quote_data.get('coverage_summary', 'Standard') # Assuming coverage info is available
        discount = quote_data.get('discount_applied', 'N/A') # Assuming discount info is part of quote_data
        quote_email_content = f"Hi Bob,\n\nHere is your final auto insurance quote (revised per request):\nPremium: ${premium}/year\nCoverage: {coverage}\nDiscount Applied: {discount}\n\nPlease let us know if you'd like to proceed.\n\nThanks,\nYour Agent"
        event_data = {"email_content": quote_email_content}
        simulated_final_state = final_state.copy() # Use state from Turn 6 (final accepted state)
        simulated_final_state['messages'] = simulated_final_state.get('messages', []) + [{
            "role": "agent",
            "type": "quote_sent",
            "content": quote_email_content,
            "timestamp": datetime.now().isoformat()
        }]
        simulated_final_state['status'] = 'quote_sent' # Manually update status for visualization
        add_timeline_event("Agent Action", "Agent sends final quote email (Simulated)", actor="Agent", data=event_data, state=simulated_final_state)
        final_state = simulated_final_state # Update final state for the report


    except AssertionError as e:
        logger.error(f"Assertion failed during simulation: {e}")
        # Continue to generate visualization with data up to the failure point
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(traceback.format_exc())
        # Optionally capture the partial state if available before error
        # final_state = agent.get_state(thread_id) if thread_id else {}

    # Get final graph
    mermaid_graph = ""
    if agent.compiled_workflow:
        try:
            mermaid_graph = agent.compiled_workflow.get_graph().draw_mermaid()
        except Exception as graph_err:
            logger.error(f"Could not generate mermaid graph: {graph_err}")

    return timeline, final_state, mermaid_graph, thread_id

def generate_visualization(
    timeline: List[Dict],
    final_state: AgentState,
    mermaid_graph: str,
    thread_id: str,
    output_path: str = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME),
    template_path: str = TEMPLATE_PATH,
):
    """Generates the HTML visualization file."""
    logger.info(f"Generating HTML visualization to {output_path}")
    try:
        template_dir = os.path.dirname(template_path)
        template_file = os.path.basename(template_path)
        env = Environment(loader=FileSystemLoader(template_dir), autoescape=select_autoescape(['html', 'xml']))
        env.filters['tojson'] = json.dumps # Add json filter for state display
        template = env.get_template(template_file)

        rendered_html = template.render(
            timeline=timeline,
            final_state=final_state,
            mermaid_graph_markdown=mermaid_graph,
            thread_id=thread_id
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(rendered_html)
        logger.info("HTML file generated successfully.")
    except TemplateNotFound:
        logger.error(f"Template file not found at {template_path}")
    except Exception as e:
        logger.error(f"Error generating HTML: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Need to adjust path if running from src directory
    import sys
    if os.path.basename(os.getcwd()) == 'src':
        os.chdir('..') # Go up to project root
        sys.path.insert(0, os.path.abspath('.'))
        # Need to re-import with adjusted path
        from ppa_agent.agent import PPAAgent
        from ppa_agent.state import AgentState
        logger.info(f"Changed working directory to: {os.getcwd()}")

    
    timeline, final_state, mermaid_graph, thread_id = run_simulation()

    if timeline: # Check if simulation produced timeline data
        generate_visualization(
            timeline=timeline,
            final_state=final_state,
            mermaid_graph=mermaid_graph,
            thread_id=thread_id
        )
    else:
        logger.error("Simulation did not produce data. HTML generation skipped.")
