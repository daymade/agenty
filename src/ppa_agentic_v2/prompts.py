# src/ppa_agentic_v2/prompts.py
import json
from typing import List, Any, TYPE_CHECKING

# Use forward reference for AgentState to avoid circular import
if TYPE_CHECKING:
    from .state import AgentState

# --- Planner Prompt V3 (Adds Human Feedback Handling & Review Decision) ---
PLANNER_PROMPT_TEMPLATE_V3 = """
You are a PPA Insurance Quoting Agent assistant. Your primary goal is: {goal}.
You interact with the customer via messages and use internal tools to call Mercury Insurance APIs.

**Current Conversation State:**

*   **Conversation History (Newest Last):**
{messages_str}
*   **Current Customer Information Extracted:**
{customer_info_str}
*   **Mercury API Session Context (e.g., Quote ID):**
{mercury_session_str}
*   **Result of the VERY LAST Action Taken:**
{last_tool_outputs_str}
*   **Feedback from Last Human Review (if any):**
{human_feedback_str}

**Your Task:**

Based on the current state, conversation, AND HUMAN FEEDBACK (if provided), decide the single next best action.

1.  **Process Human Feedback:** If feedback exists (not 'None'), **you MUST address it**. Re-evaluate your previous plan based on the comment or use the edited inputs if provided. Do NOT repeat the rejected action without changes unless the feedback allows it.
2.  **Analyze State & Goal:** Review history, current data, goal, and last action result. What's the next logical step in the quoting sequence (e.g., Initiate -> Add Driver -> Add Vehicle -> Rate -> Summarize)? Handle errors from the last step.
3.  **Determine Next Step & Formulate Plan:**
    *   If ready to call an API or summarize: Choose the tool and generate exact `args`.
    *   If customer information is needed: Plan to use `ask_customer_tool` with specific `missing_fields`.
    *   If stuck or error unrecoverable: Plan `request_human_review_tool` (if available and necessary).
    *   If goal met: Plan `prepare_final_summary_tool` or indicate completion (`tool_name: null`).
4.  **Decide on Agency Review:** Set `requires_review` to `true` if the PLANNED action needs human review before execution (e.g., policy dictates review for `ask_customer_tool`, `rate_quote_tool`, or if you are uncertain). Otherwise, set `requires_review` to `false`.

**Available Tools:**

{tools_str}

**Output Format:**

Respond ONLY with a JSON object:
```json
{{
  "thought": "Detailed reasoning, including how you processed human feedback (if any) and why you chose this action and review requirement.",
  "requires_review": true | false, // Does THIS planned action need agency review?
  "tool_name": "Name of the tool to use OR null.",
  "args": {{ ... }} // Args for the tool OR null.
}}
```

**Example (Processing Rejection):**
```json
{{
  "thought": "The human agent rejected the previous plan to call 'add_vehicle_tool' with VIN '123', commenting 'VIN is incorrect'. I must ask the customer for the correct VIN instead.",
  "requires_review": true, // Asking the customer always requires review by policy
  "tool_name": "ask_customer_tool",
  "args": {{ "missing_fields": ["vehicle_vin"] }}
}}
```

Analyze the current state and provide your decision in the specified JSON format ONLY.
"""

def format_planner_prompt(state: 'AgentState', tools: List[Any]) -> str:
    """Formats the prompt for the planner LLM (V3 - includes feedback)."""
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

    messages_str = "\n".join(msg_parts)
    if not messages_str: messages_str = "No messages yet."

    # Ensure complex objects are serialized safely (e.g., handling non-serializable types)
    customer_info_str = json.dumps(state.customer_info or {}, indent=2, default=str)
    mercury_session_str = json.dumps(state.mercury_session or {}, indent=2, default=str)
    last_tool_outputs_str = json.dumps(state.last_tool_outputs or {"status": "N/A"}, indent=2, default=str)

    # Format human feedback clearly
    human_feedback_str = "None"
    if state.human_feedback:
        feedback = state.human_feedback
        feedback_status = "approved" if feedback.get("approved", False) else "rejected"
        feedback_comment = feedback.get("comment", "No comment provided.")
        human_feedback_str = f"""IMPORTANT: Last plan was {feedback_status}.
Human feedback: {feedback_comment}
Consider this feedback when formulating your next plan."""

    tools_str = "\n".join([f"- {tool.name}: {tool.description}. Args schema: {tool.args_schema.schema() if tool.args_schema else '{}'}" for tool in tools])

    prompt = PLANNER_PROMPT_TEMPLATE_V3.format(
        goal=state.goal,
        messages_str=messages_str,
        customer_info_str=customer_info_str,
        mercury_session_str=mercury_session_str,
        last_tool_outputs_str=last_tool_outputs_str,
        human_feedback_str=human_feedback_str, # <-- Include formatted feedback
        tools_str=tools_str
    )
    return prompt

# Template for generating info request (needed by ask_customer_tool)
GENERATE_INFO_REQUEST_PROMPT_TEMPLATE = """
Generate a polite email asking the customer to provide these specific details: {missing_info_list}.
Keep the email concise and clear. Respond ONLY with the body of the email.
"""