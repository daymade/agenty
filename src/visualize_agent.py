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
        # Use draw_mermaid() to get the definition string
        mermaid_definition = graph.get_graph().draw_mermaid()
        logger.info(f"Generated Mermaid definition for {test_name}")
    except Exception as e:
        logger.error(f"Error generating Mermaid definition for {test_name}: {e}")
        # Escape inner quotes for the f-string
        mermaid_definition = (
            f'graph TD\n    error["Error generating graph: {e}"]'  # Display error in diagram
        )

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

    # 3. Generate HTML Report with Mermaid JS
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PPA Agent Result: {test_name}</title>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: sans-serif; display: flex; flex-direction: column; align-items: center; }}
            .container {{ display: flex; width: 95%; max-width: 1600px; margin-top: 20px; gap: 20px; }}
            /* Adjust layout: Make graph container wider */
            .graph-container {{ flex: 3; overflow: auto; border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9; }}
            .state-container {{ flex: 1; border: 1px solid #ccc; padding: 10px; background-color: #f0f0f0; max-height: 90vh; overflow-y: auto; }}
            h1, h2 {{ text-align: center; }}
            pre {{ white-space: pre-wrap; word-wrap: break-word; background-color: #fff; padding: 10px; border-radius: 5px; font-size: 12px; }}
            /* Style for the mermaid diagram container */
            .mermaid {{ text-align: center; background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
            .error {{ color: red; font-weight: bold; }}
        </style>
        <!-- Load Mermaid JS from CDN -->
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    </head>
    <body>
        <h1>PPA Agent Result: {test_name}</h1>
        <div class="container">
            <div class="graph-container">
                <h2>Workflow Graph</h2>
                <!-- Embed Mermaid definition -->
                <pre class="mermaid">
{mermaid_definition}
                </pre>
            </div>
            <div class="state-container">
                <h2>Final State</h2>
                <pre>{state_json_str}</pre>
            </div>
        </div>
        <!-- Initialize Mermaid -->
        <script>
            mermaid.initialize({{ startOnLoad: true }});
        </script>
    </body>
    </html>
    """
    with open(html_path, "w") as f:
        f.write(html_content)
    logger.info(f"Saved HTML report to {html_path}")


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

    # Run the multi-turn test case
    logger.info("Processing test email: multi_turn (Turn 1)")
    multi_turn_test_name = "multi_turn_final"
    try:
        state_1 = agent.process_email(TEST_CASES["multi_turn_1"])
        thread_id_1 = state_1.get("thread_id")
        if thread_id_1:
            logger.info(f"Processing test email: multi_turn (Turn 2, Thread: {thread_id_1})")
            state_2 = agent.process_email(MULTI_TURN_EMAIL_2, thread_id=thread_id_1)
            logger.info("Generating visualization for multi-turn final state...")
            generate_visualization(agent, multi_turn_test_name, state_2)
        else:
            logger.error("Could not get thread_id from multi_turn_1, skipping Turn 2.")
            html_path = OUTPUT_DIR / f"{multi_turn_test_name}_result.html"
            with open(html_path, "w") as f:
                f.write(
                    f"<h1>Error processing {multi_turn_test_name}</h1><pre>Could not get thread_id</pre>"
                )

    except Exception as e:
        logger.error(f"Error processing multi-turn test case: {e}", exc_info=True)
        html_path = OUTPUT_DIR / f"{multi_turn_test_name}_result.html"
        with open(html_path, "w") as f:
            f.write(f"<h1>Error processing {multi_turn_test_name}</h1><pre>{e}</pre>")

    logger.info(f"Visualization files created in {OUTPUT_DIR.resolve()}")
    logger.info("Open the following files in your browser:")
    for html_file in OUTPUT_DIR.glob("*.html"):
        logger.info(f"  - {html_file.resolve()}")
