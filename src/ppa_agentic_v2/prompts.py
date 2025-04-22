# src/ppa_agentic_v2/prompts.py
import json
from typing import List, Any, Optional, Dict
from typing_extensions import TYPE_CHECKING

# Use forward reference for AgentState to avoid circular import
if TYPE_CHECKING:
    from .state import AgentState

# --- Planner Prompt V3 (Simplified - Relies on OpenAI Tool Calling) ---
from langchain.prompts import PromptTemplate
PLANNER_PROMPT_TEMPLATE_V3 = PromptTemplate(
    input_variables=["history", "goal", "customer_info", "mercury_session_context", "last_event_result_str", "human_feedback_str", "tool_schemas_json"],
    template="""You are the 'Planner' component of the PPA New Business AI Agent.
Your goal: {goal}

**CRITICAL INSTRUCTION: Analyze the ENTIRE conversation history below, including all previous messages, your past plans, tool execution results, and any human feedback, before deciding the next step. Avoid repeating failed actions or getting stuck in loops.**

**Current State & Context:**
*   Customer Info: {customer_info}
*   Mercury API Session Context: {mercury_session_context}

**Conversation History (Oldest to Newest):**
{history}

**Result of the VERY LAST Event:**
{last_event_result_str}

**Human Feedback on Previous Action (if any):**
{human_feedback_str}

**Your Task:**
Based on the full context above, decide the single best next step. You have access to the following tools:

**Available Tools:**
```json
{tool_schemas_json}
```

**Decision Process:**
1.  **Review History & State:** Understand the full conversation flow and current status.
2.  **Identify Next Step:** Determine the most logical action towards the goal.
3.  **Select Tool (if needed):** Choose the appropriate tool and formulate the arguments required by its schema.
4.  **Agency Review Check:** Does this specific action require agency review? (Refer to tools marked with `Requires Agency Review: True` in their description). If yes, set `requires_agency_review` to `true`.
5.  **Wait Condition:** If the 'Result of the VERY LAST Event' indicates a tool like `ask_customer_tool` was successfully executed, your ONLY valid next step is usually to wait for the customer response. Set `action` to `null` in this case.
6.  **Formulate Output:** Structure your decision in the JSON format below.

**Output Format:** Always respond using the following JSON structure ONLY. Do not add any text before or after the JSON block.

```json
{{
  "thought": "Your reasoning for choosing the next action, considering the history, state, goal, and available tools. Explain *why* this is the best next step.",
  "action": {{ // Use null if waiting for customer
    "tool_name": "tool_name_to_use",
    "args": {{ /* Arguments matching the tool's schema */ }}
  }},
  "requires_agency_review": boolean // True if the chosen action needs human approval, False otherwise
}}
```

**Example Output (Calling a tool):**
```json
{{
  "thought": "The customer provided their name and DOB. I need their address next. I will use the ask_customer_tool.",
  "action": {{
    "tool_name": "ask_customer_tool",
    "args": {{"question": "Thank you! Could you please provide your current residential address?"}}
  }},
  "requires_agency_review": true
}}
```

**Example Output (Waiting for customer after asking a question):**
```json
{{
  "thought": "I have just asked the customer for their address using ask_customer_tool. Now I need to wait for their response.",
  "action": null, // Set action to null to wait
  "requires_agency_review": false // Waiting doesn't require review
}}
```

**Begin! Analyze the state and history, then provide your JSON output.**
"""
)

from langchain.tools.render import render_text_description

def format_planner_prompt(state: 'AgentState', tools: List['BaseTool']) -> Dict[str, Any]:
    """Formats the input for the planner LLM call based on the current AgentState and available tools."""
    from .state import AgentState # Import locally for type hint if needed
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage # For type checking

    messages = state.messages or []
    # Format messages including role
    msg_parts = []
    for m in messages:
         role = "Unknown"
         if isinstance(m, HumanMessage): role = "Human"
         elif isinstance(m, AIMessage): role = "AI"
         elif isinstance(m, ToolMessage):
             tool_call_id_str = f" ({m.tool_call_id[:8]})" if m.tool_call_id else ""
             role = f"Tool Result{tool_call_id_str}" # More descriptive
         # Safely access content, handle potential non-string content if necessary
         content_str = str(m.content) if m.content is not None else "" 
         msg_parts.append(f"{role}: {content_str}")

    history_str = "\n".join(msg_parts)
    if not history_str: history_str = "No messages yet."

    # Ensure complex objects are serialized safely (e.g., handling non-serializable types)
    customer_info_str = json.dumps(state.customer_info or {}, indent=2, default=str)
    mercury_session_str = json.dumps(state.mercury_session or {}, indent=2, default=str)

    # Format Last Event Result from Event History
    event_history = state.event_history
    if event_history:
        last_event = event_history[-1] # Get the most recent event
        last_event_result_str = f"Type: {last_event.get('type', 'unknown')}\nDetails: {json.dumps({k: v for k, v in last_event.items() if k != 'type'}, indent=2)}"
    else:
        last_event_result_str = "No previous tool execution or human feedback recorded in this session yet."

    # Use the passed human_feedback_str directly
    final_human_feedback_str = "None"
    if state.human_feedback:
        feedback = state.human_feedback
        feedback_status = "approved" if feedback.get("approved", False) else "rejected"
        feedback_comment = feedback.get("comment", "No comment provided.")
        final_human_feedback_str = f"Status: {feedback_status}. Comment: {feedback_comment}"

    # Format tool schemas for the prompt
    rendered_tools = render_text_description(tools) # <-- Use 'tools' argument

    # Format tool schemas for the prompt
    tool_schemas = {}
    for tool in tools:
        if hasattr(tool, 'args_schema') and hasattr(tool.args_schema, 'schema'):
            tool_schemas[tool.name] = tool.args_schema.schema()
        else:
            tool_schemas[tool.name] = {} # Tool has no args
    tool_schemas_json = json.dumps(tool_schemas, indent=2)

    # Prepare the final prompt inputs dictionary using the correct keys
    # Keys MUST match PLANNER_PROMPT_TEMPLATE_V3.input_variables
    prompt_inputs = {
        "goal": state.goal or "Process the PPA insurance quote request.",
        "history": history_str,                       # Use 'history' key
        "customer_info": customer_info_str,             # Use 'customer_info' key
        "mercury_session_context": mercury_session_str, # Use 'mercury_session_context' key
        "last_event_result_str": last_event_result_str,  # Use 'last_event_result_str' key
        "human_feedback_str": final_human_feedback_str, # Use 'human_feedback_str' key
        "tool_schemas_json": tool_schemas_json,
    }

    # Use the PromptTemplate's invoke method to format the prompt
    # This typically returns a PromptValue (e.g., StringPromptValue)
    # which is suitable for the LLM's ainvoke method.
    formatted_prompt_value = PLANNER_PROMPT_TEMPLATE_V3.invoke(prompt_inputs)

    # The LangChain node expects the formatted prompt object (e.g., PromptValue)
    return formatted_prompt_value

# Template for generating info request (needed by ask_customer_tool)
GENERATE_INFO_REQUEST_PROMPT_TEMPLATE = """
Generate a polite email asking the customer to provide these specific details: {missing_info_list}.
Keep the email concise and clear. Respond ONLY with the body of the email.
"""