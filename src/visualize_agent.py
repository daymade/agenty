"""
Generates visualizations of the PPA Agent workflow using graphviz.
Saves the graph structure and example final states for different scenarios.
"""

import json
import logging
from pathlib import Path

from dotenv import load_dotenv

# Assuming PPAAgent is the main class and state definitions are accessible
from ppa_agent.agent import PPAAgent
from ppa_agent.state import \
    AgentState  # Import AgentState for type hint if needed

# Removed base64 import as we'll embed definition directly
# import base64


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Test Cases ---
TEST_CASES = {
    "complete_info": """
        Hello, I need car insurance for my vehicle.
        I'm John Smith, 35 years old, living at 123 Main St, Seattle, WA.

        I drive a 2020 Toyota Camry. I've been driving for 15 years with no accidents.

        Looking forward to your quote.
        Best regards,
        John
        """,
    "missing_info": """
        Hello, I'd like to get car insurance.
        I'm Sarah Johnson and I drive a Honda.

        Please let me know what information you need.

        Thanks,
        Sarah
        """,
    "non_ppa": """
        Hi, I'm interested in getting homeowners insurance for my new house.
        Can you help me with that?
        """,
    # Add multi-turn scenario base email
    "multi_turn_1": """
        Subject: Need a car insurance quote!

        Hi there,
        Please provide a quote for my 2022 Subaru.
        My name is Alice Williams, age 40.
        I live at 789 Pine St, Portland, OR.
        Thanks!
        """,
}

# Define the follow-up email for the multi-turn test
MULTI_TURN_EMAIL_2 = """
    Subject: Re: Need a car insurance quote!

    Sorry about that, it's an Outback Wilderness model.
    """

# Email 3: Customer confirms discount eligibility
MULTI_TURN_EMAIL_3 = """
Subject: Re: Need a car insurance quote!

Yes, that's correct, I have a clean driving record for the past 5 years - no accidents or violations.
Please proceed with the quote.
"""

# --- Visualization Generation ---

OUTPUT_DIR = Path("visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_visualization(agent: PPAAgent, test_name: str, final_state: AgentState) -> None:
    """Generates HTML report with Mermaid graph and final state."""
    json_path = OUTPUT_DIR / f"{test_name}_state.json"
    html_path = OUTPUT_DIR / f"{test_name}_result.html"

    # 1. Get Mermaid Graph Definition
    mermaid_definition = ""
    try:
        graph = agent.workflow
        mermaid_definition = graph.get_graph().draw_mermaid()
        logger.info(f"Generated Mermaid definition for {test_name}")
    except Exception as e:
        logger.error(f"Error generating Mermaid definition for {test_name}: {e}")
        mermaid_definition = f'graph TD\n    error["Error generating graph: {e}"]'

    # 2. Save Final State JSON
    state_json_str = ""
    try:
        serializable_state = dict(final_state)
        with open(json_path, "w") as f:
            json.dump(serializable_state, f, indent=2, default=str)
        logger.info(f"Saved final state JSON to {json_path}")
        state_json_str = json.dumps(serializable_state, indent=2, default=str)
    except Exception as e:
        logger.error(f"Error saving state JSON for {test_name}: {e}")
        state_json_str = f"<pre>Error saving state: {e}</pre>"

    # 3. Generate HTML for Conversation History
    conversation_html_parts = ['<h2>Conversation History</h2>']
    email_thread = final_state.get("email_thread", [])
    if not email_thread:
        conversation_html_parts.append("<p>No conversation history found.</p>")
    else:
        for i, message in enumerate(email_thread):
            role = message.get("role", "unknown").capitalize()
            timestamp = message.get("timestamp", "")
            content = message.get("content", "No content")
            msg_type = f' ({message.get("type", "")})' if role == 'Agent' else ''
            ts_display = f' ({timestamp.split(".")[0]})' if timestamp else ''

            # Determine CSS class based on role
            turn_class = "customer-turn" if role == "Customer" else "agent-turn"

            conversation_html_parts.append(f'''
                <div class="conversation-turn {turn_class}">
                    <h4>{role}{msg_type}{ts_display}</h4>
                    <pre>{content}</pre>
                </div>
                <hr style="border: none; border-top: 1px solid #eee; margin: 15px 0;">
            ''')
    conversation_history_html = "\n".join(conversation_html_parts)

    # 4. Generate HTML Report with Mermaid JS
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PPA Agent Result - {test_name.replace('_', ' ').title()}</title>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: system-ui, -apple-system, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f0f2f5;
            }}
            .main-container {{
                max-width: 1800px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                max-height: 600px;
                overflow: auto;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background: #fff;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }}
            .flex-container {{
                display: flex;
                gap: 20px;
                margin-top: 20px;
            }}
            .graph-container {{
                flex: 2;
                min-width: 0;
            }}
            .details-container {{
                flex: 1;
                min-width: 0;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
                margin-top: 0;
            }}
            .message {{
                background: #f8f9fa;
                padding: 15px;
                margin: 10px 0;
                border-radius: 4px;
                border-left: 4px solid #007bff;
            }}
            .message.requires-review {{
                border-left-color: #ffc107;
            }}
            pre {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 4px;
                overflow-x: auto;
                font-size: 13px;
                line-height: 1.5;
            }}
            .customer-info {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 10px;
                margin: 10px 0;
            }}
            .info-item {{
                background: #f8f9fa;
                padding: 10px;
                border-radius: 4px;
            }}
            .info-label {{
                font-weight: bold;
                color: #666;
            }}
            .mermaid {{
                background: white;
                padding: 20px;
                border-radius: 4px;
            }}
            .missing-info {{
                color: #dc3545;
                font-style: italic;
            }}
            .badge {{
                display: inline-block;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 500;
                margin-left: 8px;
            }}
            .badge.review {{
                background-color: #fff3cd;
                color: #856404;
            }}
            .badge.type {{
                background-color: #e2e3e5;
                color: #383d41;
            }}
            .conversation-turn {{ margin-bottom: 15px; }}
            .conversation-turn h4 {{ margin-bottom: 5px; color: #444; font-size: 1.1em; }}
            .conversation-turn pre {{ white-space: pre-wrap; word-wrap: break-word; background: #f9f9f9; padding: 10px; border: 1px solid #eee; border-radius: 4px; font-size: 0.95em; color: #333; }}
            .customer-turn h4 {{ color: #0056b3; }}
            .agent-turn h4 {{ color: #28a745; }}
            .customer-turn pre {{ background: #e7f3ff; border-color: #cce5ff; }}
            .agent-turn pre {{ background: #e6ffed; border-color: #c3e6cb; }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    </head>
    <body>
        <div class="main-container">
            <h1>PPA Agent Result - {test_name.replace('_', ' ').title()}</h1>
            
            <div class="section">
                {conversation_history_html}
            </div>

            <div class="section">
                <h2>Extracted Customer Information</h2>
                <div class="customer-info">
                    {generate_customer_info_html(final_state.get('customer_info', {}))}
                </div>
            </div>

            <div class="section">
                <h2>Missing Information</h2>
                {generate_missing_info_html(final_state.get('missing_info', []))}
            </div>

            <div class="section">
                <h2>Generated Messages ({len(final_state.get('messages', []))})</h2>
                {generate_messages_html(final_state.get('messages', []))}
            </div>

            <div class="flex-container">
                <div class="graph-container section">
                    <h2>Workflow Graph</h2>
                    <pre class="mermaid">
{mermaid_definition}
                    </pre>
                </div>
                
                <div class="details-container section">
                    <h2>Complete State</h2>
                    <pre>{state_json_str}</pre>
                </div>
            </div>
        </div>
        <script>
            mermaid.initialize({{ startOnLoad: true }});
        </script>
    </body>
    </html>
    """
    with open(html_path, "w") as f:
        f.write(html_content)
    logger.info(f"Saved HTML report to {html_path}")

def generate_customer_info_html(customer_info: dict) -> str:
    """Generate HTML for customer information display."""
    if not customer_info:
        return '<div class="info-item missing-info">No customer information available</div>'
    
    html_parts = []
    for key, value in customer_info.items():
        if value:  # Only show non-empty values
            formatted_key = key.replace('_', ' ').title()
            html_parts.append(f'''
                <div class="info-item">
                    <div class="info-label">{formatted_key}</div>
                    <div class="info-value">{value}</div>
                </div>
            ''')
    return '\n'.join(html_parts)

def generate_missing_info_html(missing_info: list) -> str:
    """Generate HTML for missing information display."""
    if not missing_info:
        return '<div>No missing information</div>'
    
    html_parts = ['<ul>']
    for item in missing_info:
        html_parts.append(f'<li class="missing-info">{item}</li>')
    html_parts.append('</ul>')
    return '\n'.join(html_parts)

def generate_messages_html(messages: list) -> str:
    """Generate HTML for messages display."""
    if not messages:
        return '<div>No messages generated</div>'
    
    html_parts = []
    for msg in messages:
        review_badge = '<span class="badge review">requires review</span>' if msg.get('requires_review') else ''
        type_badge = f'<span class="badge type">{msg.get("type", "unknown")}</span>'
        review_outcome = msg.get('review_outcome', '')
        if review_outcome:
            review_outcome_badge = f'<span class="badge {review_outcome}">review: {review_outcome}</span>'
        else:
            review_outcome_badge = ''
        
        html_parts.append(f'''
            <div class="message {' requires-review' if msg.get('requires_review') else ''}">
                <div style="margin-bottom: 8px">
                    <strong>agent</strong>
                    {type_badge}
                    {review_badge}
                    {review_outcome_badge}
                </div>
                <pre style="margin: 0">{msg.get("content", "No content")}</pre>
            </div>
        ''')
    return '\n'.join(html_parts)


def simulate_human_review(state: AgentState) -> AgentState:
    """Simulates a human review by setting review_outcome to 'accepted'.
    
    In a real implementation, this would be an external UI/API call where
    a human agent reviews the content and provides their decision.
    """
    logger.info("Simulating human review process...")
    
    # Check if review is required
    if not state.get("requires_review", False):
        logger.info("No review required for this state.")
        return state
    
    # In a real system, this is where we'd pause and wait for human input
    logger.info("State requires human review. Simulating human agent accepting the output...")
    
    # Create a copy of the state to modify
    updated_state = dict(state)
    
    # Simulate human accepting the output
    updated_state["review_outcome"] = "accepted"
    
    # For visualization purposes, tag the messages that were reviewed
    for msg in updated_state.get("messages", []):
        if msg.get("requires_review", False):
            msg["review_outcome"] = "accepted"
    
    logger.info("Human review simulation completed: Output ACCEPTED")
    return updated_state

if __name__ == "__main__":
    logger.info("Initializing PPA Agent...")
    # Revert initialization to use provider string
    # It will use the model name from config by default
    agent = PPAAgent(provider="gemini")

    # Run standard test cases
    for test_name, email_content in TEST_CASES.items():
        if test_name == "multi_turn_1":
            continue
        logger.info(f"Processing test email: {test_name}")
        try:
            final_state = agent.process_email(email_content)
            generate_visualization(agent, test_name, final_state)
        except Exception as e:
            logger.error(f"Error processing test case {test_name}: {e}", exc_info=True)
            html_path = OUTPUT_DIR / f"{test_name}_result.html"
            with open(html_path, "w") as f:
                f.write(f"<h1>Error processing {test_name}</h1><pre>{e}</pre>")

    # Run the multi-turn test case with human review simulation
    logger.info("Processing test email: multi_turn (Turn 1)")
    multi_turn_test_name = "multi_turn_final"
    human_review_test_name = "multi_turn_with_review" 
    try:
        # Turn 1: Initial customer email
        state_1 = agent.process_email(TEST_CASES["multi_turn_1"])
        thread_id_1 = state_1.get("thread_id")
        if not thread_id_1:
            raise ValueError("Could not get thread_id from multi_turn_1")
        
        # Turn 2: Customer provides missing vehicle model
        logger.info(f"Processing test email: multi_turn (Turn 2, Thread: {thread_id_1})")
        state_2 = agent.process_email(MULTI_TURN_EMAIL_2, thread_id=thread_id_1)
        thread_id_2 = state_2.get("thread_id")
        if not thread_id_2:
            raise ValueError("Could not get thread_id from multi_turn_2")
        
        # Turn 3: Customer confirms clean driving record
        logger.info(f"Processing test email: multi_turn (Turn 3, Thread: {thread_id_2})")
        state_3 = agent.process_email(MULTI_TURN_EMAIL_3, thread_id=thread_id_2)
        thread_id_3 = state_3.get("thread_id")
        if not thread_id_3:
            raise ValueError("Could not get thread_id from multi_turn_3")

        # Save the state before human review (may require review)
        logger.info("Generating visualization for multi-turn state before human review...")
        generate_visualization(agent, multi_turn_test_name, state_3)
        
        # Check if human review is required
        if state_3.get("requires_review", False):
            logger.info("State requires human review, simulating review process...")
            # Simulate human review (set review_outcome to 'accepted')
            reviewed_state = simulate_human_review(state_3)
            
            # In a real implementation, this is where we'd use LangGraph's thread
            # capabilities to resume the workflow with the updated state
            # For our simulation, we'll manually set the state and continue
            
            # Save the state after human review 
            logger.info("Generating visualization for state after human review...")
            generate_visualization(agent, human_review_test_name, reviewed_state)
            
            # For a complete simulation, we would pass the reviewed state back to
            # the LangGraph workflow and continue execution. In a real implementation,
            # this would involve LangGraph's thread management capabilities.
            logger.info("In a real implementation, execution would continue after human review.")
        else:
            logger.info("No human review required for this state.")
            
    except Exception as e:
        logger.error(f"Error processing multi-turn test case: {e}", exc_info=True)
        html_path = OUTPUT_DIR / f"{multi_turn_test_name}_result.html"
        with open(html_path, "w") as f:
            f.write(f"<h1>Error processing {multi_turn_test_name}</h1><pre>{e}</pre>")

    logger.info(f"Visualization files created in {OUTPUT_DIR.resolve()}")
    logger.info("Open the following files in your browser:")
    for html_file in OUTPUT_DIR.glob("*.html"):
        logger.info(f"  - {html_file.resolve()}")
