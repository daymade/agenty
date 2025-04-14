"""
Visualization script for testing the PPA Agent and visualizing its output.

This script runs the agent with a test email and generates a visualization
of the results for easier understanding and debugging.

Usage:
    python -m src.visualize_agent
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Import the agent
from ppa_agent.agent import PPAAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress AFC info messages from google_genai
logging.getLogger("google_genai").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()


def generate_html_table(data_dict: Dict[str, Any]) -> str:
    """Generate an HTML table from a dictionary."""
    if not data_dict:
        return '<div class="error">No data available</div>'

    html = '<div class="customer-info">'
    for k, v in data_dict.items():
        html += f'<div><span class="label">{k}:</span> {v}</div>'
    html += "</div>"
    return html


def generate_missing_info_list(missing_items: Optional[List[str]]) -> str:
    """Generate an HTML list of missing information."""
    if not missing_items:
        return '<li class="success">No missing information</li>'

    html = ""
    for item in missing_items:
        html += f"<li>{item}</li>"
    return html


def generate_messages_html(messages: Optional[List[Dict[str, Any]]]) -> str:
    """Generate HTML for messages."""
    if not messages:
        return '<div class="error">No messages generated</div>'

    html = ""
    for m in messages:
        html += '<div class="message">'
        html += f'<strong>{m["role"]}</strong> '
        html += f'<span class="tag type">{m.get("type", "unknown")}</span>'

        if m.get("requires_review"):
            html += '<span class="tag requires-review">requires review</span>'

        html += f'<p>{m["content"]}</p>'
        html += "</div>"

    return html


def main() -> None:
    """Run a test case and visualize the results."""
    logger.info("Initializing PPA Agent...")

    # Initialize the agent
    agent = PPAAgent(provider="gemini")

    # Make sure a visualizations directory exists
    vis_dir = Path("visualizations")
    vis_dir.mkdir(exist_ok=True)

    # Sample emails for testing
    test_emails = {
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
    }

    # Process each test email and create visualizations
    for name, email in test_emails.items():
        logger.info(f"Processing test email: {name}")

        # Process the email
        result = agent.process_email(email)

        # Save a JSON version of the result for debugging
        with open(vis_dir / f"{name}_result.json", "w") as f:
            # Convert to a serializable format
            serializable_result = {
                k: (v if k != "customer_info" else dict(v)) for k, v in result.items()
            }
            json.dump(serializable_result, f, indent=2)

        # Create an HTML visualization
        with open(vis_dir / f"{name}_result.html", "w") as f:
            # HTML header
            html_content = f"""
            <html>
            <head>
                <title>PPA Agent Result - {name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                    .box {{
                        background: #f5f5f5;
                        padding: 15px;
                        border-radius: 5px;
                        margin-bottom: 20px;
                    }}
                    .email {{
                        background: #e6f7ff;
                        padding: 15px;
                        border-radius: 5px;
                        white-space: pre-wrap;
                    }}
                    .message {{
                        background: white;
                        padding: 10px;
                        margin: 10px 0;
                        border-left: 3px solid #4CAF50;
                    }}
                    .customer-info {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
                    .label {{ font-weight: bold; }}
                    .error {{ color: red; }}
                    .success {{ color: green; }}
                    pre {{ white-space: pre-wrap; }}
                    h1, h2 {{ color: #333; }}
                    .tag {{
                        display: inline-block;
                        padding: 2px 8px;
                        border-radius: 12px;
                        font-size: 12px;
                        margin-right: 5px;
                    }}
                    .tag.requires-review {{ background: #ffe0e0; }}
                    .tag.type {{ background: #e0f0ff; }}
                </style>
            </head>
            <body>
                <h1>PPA Agent Result - {name.replace('_', ' ').title()}</h1>

                <h2>Input Email</h2>
                <div class="box email">
                    <pre>{email}</pre>
                </div>

                <h2>Extracted Customer Information</h2>
                <div class="box">
                    {generate_html_table(result["customer_info"])}
                </div>

                <h2>Missing Information</h2>
                <div class="box">
                    <ul>
                        {generate_missing_info_list(result.get("missing_info", []))}
                    </ul>
                </div>

                <h2>Generated Messages ({len(result.get("messages", []))})</h2>
                <div class="box">
                    {generate_messages_html(result.get("messages", []))}
                </div>

                <h2>Complete State</h2>
                <div class="box">
                    <pre>{json.dumps(serializable_result, indent=2)}</pre>
                </div>
            </body>
            </html>
            """

            f.write(html_content)

    logger.info(f"Visualization files created in {vis_dir.absolute()}")
    logger.info("Open the following files in your browser:")

    for name in test_emails.keys():
        logger.info(f"  - {vis_dir.absolute() / f'{name}_result.html'}")


if __name__ == "__main__":
    main()
