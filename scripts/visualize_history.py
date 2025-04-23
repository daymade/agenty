# scripts/visualize_history.py
import json
import os
import sys
import argparse
import logging
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound

# --- Path Setup ---
# Get the absolute path of the directory containing this script (scripts/)
scripts_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root
project_root = os.path.dirname(scripts_dir)
# Add project root to sys.path to allow imports from src
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Configuration ---
DEFAULT_INPUT_FILE = os.path.join(project_root, "tests", "output", "state-history.json")
DEFAULT_OUTPUT_DIR = os.path.join(project_root, "visualizations")
DEFAULT_OUTPUT_FILENAME = "state_history_visualization.html"
DEFAULT_TEMPLATE_DIR = os.path.join(project_root, "templates")
DEFAULT_TEMPLATE_NAME = "history_visualization_template.html" # Matches filename created above

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def format_timestamp(iso_ts_str: str) -> str:
    """Formats ISO timestamp for display."""
    if not iso_ts_str:
        return "N/A"
    try:
        dt_obj = datetime.fromisoformat(iso_ts_str.replace('Z', '+00:00'))
        return dt_obj.strftime('%Y-%m-%d %H:%M:%S %Z')
    except (ValueError, TypeError):
        return iso_ts_str # Return original if parsing fails

def preprocess_state_history(history_data: list) -> list:
    """Preprocesses state history for better display."""
    processed_history = []
    thread_id = None
    for i, step in enumerate(history_data):
        processed_step = step.copy() # Work with a copy

        # Extract values, default to empty dict if not present
        values = processed_step.get("values", {})
        if not isinstance(values, dict):
            logger.warning(f"Step {i+1} missing 'values' dictionary. Skipping.")
            continue

        # Extract thread_id from metadata if available
        if not thread_id and step.get("metadata", {}).get("thread_id"):
            thread_id = step["metadata"]["thread_id"]
        
        # Ensure messages list exists and format content
        messages = values.get("messages", [])
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict):
                    # Ensure content is treated as string, handle dicts/lists
                    content = msg.get("content")
                    if isinstance(content, (dict, list)):
                         try:
                            # Pretty-print JSON content
                            msg["content"] = json.dumps(content, indent=2)
                         except TypeError:
                            msg["content"] = str(content) # Fallback to string
                    elif content is None:
                         msg["content"] = "" # Ensure it's a string

                    # Ensure type exists and is lowercase for CSS class
                    msg_type = msg.get("type", "unknown")
                    msg["type"] = str(msg_type).lower() if msg_type else "unknown"

                else:
                     logger.warning(f"Message in step {i+1} is not a dictionary: {msg}")
        else:
             values["messages"] = [] # Ensure messages key exists as list

        # Process agent_scratchpad to handle None values properly
        if values.get("agent_scratchpad") is None:
            values["agent_scratchpad"] = "N/A"
            
        # Handle null values for better display
        for key in ["planned_tool_inputs", "action_pending_review", "human_feedback", "last_tool_outputs"]:
            if values.get(key) is None:
                values[key] = "None"
                
        # Format event_history for better display
        if "event_history" in values and isinstance(values["event_history"], list):
            # Process event history items if needed
            for event in values["event_history"]:
                if isinstance(event, dict) and "result" in event and isinstance(event["result"], dict):
                    # Ensure nested result objects are serializable
                    pass

        processed_step["values"] = values # Put potentially modified values back

        # Format timestamp if present
        processed_step["created_at"] = format_timestamp(processed_step.get("created_at"))
        
        # Also add thread_id to values for easy access in template
        if thread_id and "values" in processed_step:
            processed_step["values"]["thread_id"] = thread_id

        processed_history.append(processed_step)

    # Attempt to find thread_id if not found initially
    if not thread_id:
        for step in processed_history:
            if step.get("metadata",{}).get("thread_id"):
                 thread_id = step["metadata"]["thread_id"]
                 break
            elif step.get("values",{}).get("thread_id"):
                 thread_id = step["values"]["thread_id"]
                 break

    return processed_history, thread_id


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Generate HTML visualization for agent state history.")
    parser.add_argument(
        "-i", "--input",
        default=DEFAULT_INPUT_FILE,
        help=f"Path to the state history JSON file (default: {DEFAULT_INPUT_FILE})"
    )
    parser.add_argument(
        "-o", "--output",
        default=os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_FILENAME),
        help=f"Path to save the output HTML file (default: {DEFAULT_OUTPUT_FILENAME} in visualizations/)"
    )
    parser.add_argument(
        "--template-dir",
        default=DEFAULT_TEMPLATE_DIR,
        help=f"Directory containing the Jinja2 template (default: {DEFAULT_TEMPLATE_DIR})"
    )
    parser.add_argument(
        "--template-name",
        default=DEFAULT_TEMPLATE_NAME,
        help=f"Name of the Jinja2 template file (default: {DEFAULT_TEMPLATE_NAME})"
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    template_dir = os.path.abspath(args.template_dir)
    template_name = args.template_name

    logger.info(f"Input state history file: {input_path}")
    logger.info(f"Output HTML file: {output_path}")
    logger.info(f"Template directory: {template_dir}")
    logger.info(f"Template name: {template_name}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Load JSON Data ---
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
        if not isinstance(history_data, list):
            raise TypeError("JSON root is not a list.")
        logger.info(f"Successfully loaded {len(history_data)} state snapshots.")
    except FileNotFoundError:
        logger.error(f"Error: Input file not found at {input_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Error: Failed to parse JSON file {input_path}: {e}")
        sys.exit(1)
    except TypeError as e:
         logger.error(f"Error: JSON data does not seem to be a list of states: {e}")
         sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading JSON: {e}")
        sys.exit(1)

    # --- Preprocess Data ---
    processed_history, thread_id = preprocess_state_history(history_data)

    # --- Render HTML using Jinja2 ---
    try:
        env = Environment(loader=FileSystemLoader(template_dir), autoescape=select_autoescape(['html', 'xml']))
        # Add custom filters/globals if needed
        env.filters['tojson'] = lambda data, indent=2: json.dumps(data, indent=indent, default=str) # Ensure default=str for non-serializable types
        env.filters['format_timestamp'] = format_timestamp # Make formatter available

        template = env.get_template(template_name)
        rendered_html = template.render(
            history=processed_history,
            thread_id=thread_id
        )
        logger.info("HTML content rendered successfully.")

    except TemplateNotFound:
        logger.error(f"Error: Template '{template_name}' not found in directory '{template_dir}'")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during template rendering: {e}", exc_info=True)
        sys.exit(1)

    # --- Write HTML to File ---
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(rendered_html)
        logger.info(f"Successfully wrote visualization to {output_path}")
    except IOError as e:
        logger.error(f"Error: Failed to write HTML file to {output_path}: {e}")
        sys.exit(1)
    except Exception as e:
         logger.error(f"An unexpected error occurred while writing HTML: {e}")
         sys.exit(1)

if __name__ == "__main__":
    main()