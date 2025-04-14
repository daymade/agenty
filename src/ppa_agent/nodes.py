"""
Defines the nodes (functions) used in the PPA Agent LangGraph workflow.
"""

import json
import logging
from typing import Any, Dict, List

# Import requirements from config.py
from . import config
# Use BaseLLMProvider for type hinting
from .llm_providers import BaseLLMProvider
# Import state from state.py
from .state import AgentState

# Removed prompt import - prompts defined inline in this version
# from .prompts import (
#     ANALYZE_INFO_PROMPT,
#     CHECK_DISCOUNTS_PROMPT,
#     GENERATE_INFO_REQUEST_PROMPT,
#     IDENTIFY_INTENTION_PROMPT,
#     IDENTIFY_LOB_PROMPT,
#     PREPARE_REVIEW_PROMPT,
# )

logger = logging.getLogger(__name__)

# Define prompts directly here for now
# (Could be moved to prompts.py later)
IDENTIFY_INTENTION_PROMPT = """
Identify the primary intention of the *latest* customer message in the following conversation history.
Consider the context of previous messages if needed.

Conversation History (most recent message last):
{email_thread_formatted}

You MUST respond with ONLY a JSON object in this exact format:
{{
    "intent": "new_business" | "policy_change" | "provide_info" | "ask_question" | "other"
    # Added provide_info and ask_question intents
}}

Do not include any other text, markdown, or formatting.
"""

IDENTIFY_LOB_PROMPT = """
Identify the line of business being discussed in this conversation history, focusing on the initial request if present.

Conversation History (most recent message last):
{email_thread_formatted}

You MUST respond with ONLY a JSON object in this exact format:
{{
    "lob": "PPA" | "HOME" | "OTHER" | "UNKNOWN"
}}

Do not include any other text, markdown, or formatting.
"""

ANALYZE_INFO_PROMPT = """
Analyze the full conversation history provided below. Extract all available PPA insurance information mentioned by the customer across all their messages. Compare the extracted information against the requirements list.

Conversation History (most recent message last):
{email_thread_formatted}

Required Information Fields:
{requirements}

You MUST respond with ONLY a JSON object in this exact format:
{{
    "customer_info": {{
        "driver_name": string | null,
        "driver_age": string | null,
        "vehicle_make": string | null,
        "vehicle_model": string | null,
        "vehicle_year": string | null,
        "address": string | null
    }},
    "missing_info": [string]
}}

Aggregate all information provided by the customer throughout the history. If information for a key is not found anywhere in the customer's messages, use null as the value for that key in customer_info. List the keys for requirements that were *never* mentioned by the customer in missing_info.
Do not include any other text, markdown, or formatting.
"""

# This prompt might need context too, but primarily focuses on the *current* missing info
GENERATE_INFO_REQUEST_PROMPT = """
You are a helpful insurance agent assistant chatting with a customer.
Based on the current state where the following information is still missing: {missing_info}
Write a polite email asking the customer to provide these specific details so you can generate an accurate quote.
Keep the email concise and clear. Reference the ongoing conversation implicitly (don't explicitly say "based on our previous messages").
Example: "To continue with your PPA quote request, could you please provide the following: [list]?"
Respond ONLY with the body of the email.
"""

# This prompt should consider the full context to avoid asking redundant questions
CHECK_DISCOUNTS_PROMPT = """
Review the conversation history and the extracted customer information. Identify potential discounts (e.g., good student, safe driver) that *might* apply but haven't been confirmed or asked about yet.

Conversation History:
{email_thread_formatted}

Extracted Customer Info:
{customer_info}

You MUST respond with ONLY a JSON object in this exact format:
{{
    "status": "proof_needed" | "no_proof_needed" | "no_discount_applicable",
    "message_to_customer": string | null
}}

If you identify a likely discount requiring proof that hasn't been discussed, set status to "proof_needed" and craft a *short, specific* message asking for proof for *that discount*.
If no *new* potential discounts requiring proof are identified (either none apply, or they've already been addressed/asked about in the history), set status to "no_proof_needed" and message to null.
Do not include any other text, markdown, or formatting.
"""

# This prompt needs full context for an accurate summary
PREPARE_REVIEW_PROMPT = """
Generate a concise summary of the PPA quote case based on the entire conversation history and current state, intended for internal agency review. Highlight key customer details, the generated quote data, any remaining missing information, and any potential flags or follow-up items identified during the conversation.

Conversation History:
{email_thread_formatted}

Extracted Customer Info:
{customer_info}

Missing Info:
{missing_info}

Generated Quote Data:
{quote_data}

Internal Agent Messages Generated (for context):
{messages}

Respond ONLY with the summary text. Do not include JSON or markdown.
"""

PROCESS_RESPONSE_PROMPT = """
Analyze the latest customer message in the context of the full conversation history. Determine if the customer provided information previously requested by the agent, answered a specific question (like about discounts), or asked a new question.

Conversation History (most recent message last):
{email_thread_formatted}

Agent's Last Request/Question (if any):
{last_agent_message}

Required PPA Information Fields:
{requirements}

You MUST respond with ONLY a JSON object in this exact format:
{{
    "response_type": "provided_info" | "answered_question" | "asked_question" | "other",
    "provided_info": {{ "field_name": "value", ... }},
    "answered_discount_proof": boolean | null, # true if they confirmed having proof, false if not, null if not relevant
    "customer_question": string | null, # If they asked a question
    "updated_missing_info": [string] # List of *remaining* missing PPA requirements after considering the reply
}}

Analyze the latest customer message. Populate `provided_info` with any *new* PPA requirement values mentioned. If they responded to a discount proof request, set `answered_discount_proof`. If they asked a question, capture it in `customer_question`. Calculate `updated_missing_info` by seeing which requirements are still missing based on the entire conversation *including* the latest reply.
Do not include any other text, markdown, or formatting.
"""


def format_email_thread(email_thread: List[Dict[str, Any]]) -> str:
    """Helper function to format the email thread for prompts."""
    formatted = []
    for msg in email_thread:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        formatted.append(f"[{timestamp}] {role}:\n{content}\n--- ")
    return "\n".join(formatted)


def identify_intention(state: AgentState, llm: BaseLLMProvider) -> Dict[str, str]:
    """Node to identify the intention of the *latest* email in the thread."""
    logger.info("Identifying intention...")
    # Use the helper to format the history
    thread_formatted = format_email_thread(state["email_thread"])
    prompt = IDENTIFY_INTENTION_PROMPT.format(email_thread_formatted=thread_formatted)

    try:
        response = llm.generate_sync(prompt)
        logger.debug(f"Raw intention response from LLM: {response}")
        if isinstance(response, str):
            result = json.loads(response)
        elif hasattr(response, "content") and isinstance(response.content, str):
            result = json.loads(response.content)
        elif isinstance(response, dict):
            result = response
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")

        # Add provide_info / ask_question to expected intents if needed
        intent = result.get("intent", "other")
        logger.info(f"Intention result: {{'intent': '{intent}'}}")
        return {"intent": intent}
    except Exception as e:
        logger.error(f"Error identifying intention: {e}", exc_info=True)
        return {"intent": "error"}


def identify_lob(state: AgentState, llm: BaseLLMProvider) -> Dict[str, str]:
    """Node to identify the line of business from the *entire thread*."""
    logger.info("Identifying line of business...")
    thread_formatted = format_email_thread(state["email_thread"])
    prompt = IDENTIFY_LOB_PROMPT.format(email_thread_formatted=thread_formatted)

    try:
        response = llm.generate_sync(prompt)
        logger.debug(f"Raw LOB response from LLM: {response}")
        if isinstance(response, str):
            result = json.loads(response)
        elif hasattr(response, "content") and isinstance(response.content, str):
            result = json.loads(response.content)
        elif isinstance(response, dict):
            result = response
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")

        lob = result.get("lob", "UNKNOWN")  # Default to UNKNOWN if unsure
        logger.info(f"LOB result: {{'lob': '{lob}'}}")
        return {"lob": lob}
    except Exception as e:
        logger.error(f"Error identifying LOB: {e}", exc_info=True)
        return {"lob": "error"}


def analyze_information(state: AgentState, llm: BaseLLMProvider) -> Dict[str, Any]:
    """Node to analyze customer information from the *entire thread*."""
    logger.info("Analyzing information...")
    # Use requirements from config
    requirements_json = json.dumps(config.PPA_REQUIREMENTS, indent=2)
    thread_formatted = format_email_thread(state["email_thread"])

    prompt = ANALYZE_INFO_PROMPT.format(
        requirements=requirements_json, email_thread_formatted=thread_formatted
    )

    try:
        response = llm.generate_sync(prompt)
        logger.debug(f"Raw analysis response from LLM: {response}")
        if isinstance(response, str):
            result = json.loads(response)
        elif hasattr(response, "content") and isinstance(response.content, str):
            result = json.loads(response.content)
        elif isinstance(response, dict):
            result = response
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")

        updates = {}
        # Update customer_info based on LLM extraction from thread
        if "customer_info" in result:
            # IMPORTANT: Don't just update, potentially *replace* based on full history analysis?
            # Or merge carefully? For now, let's assume replacement/latest state from LLM.
            # Filter out null values before updating state? Or keep them?
            # Keeping nulls as per prompt for now.
            updates["customer_info"] = result["customer_info"]

        # Determine missing info based on LLM output OR re-calculate
        if "missing_info" in result:
            updates["missing_info"] = result["missing_info"]
        else:
            # Fallback: Re-calculate missing based on current info and requirements
            current_info = updates.get("customer_info", state.get("customer_info", {}))
            # Use requirements from config
            updates["missing_info"] = [
                req for req in config.PPA_REQUIREMENTS if not current_info.get(req)
            ]

        # Determine status based on the calculated missing_info
        status = "info_incomplete" if updates["missing_info"] else "info_complete"
        updates["status"] = status

        logger.info(f"Info analysis result: {updates}")
        return updates

    except Exception as e:
        logger.error(f"Error analyzing information: {e}", exc_info=True)
        # Preserve existing state on error where possible
        return {
            "status": "error",
            "missing_info": state.get("missing_info", []),
            "customer_info": state.get("customer_info", {}),
        }


def generate_info_request(state: AgentState, llm: BaseLLMProvider) -> Dict[str, Any]:
    """Node to generate an email requesting missing information."""
    logger.info("Generating information request...")
    # Use the current 'missing_info' list from the state
    missing_info_str = ", ".join(state["missing_info"])
    prompt = GENERATE_INFO_REQUEST_PROMPT.format(missing_info=missing_info_str)

    try:
        response_content = llm.generate_sync(prompt)
        if hasattr(response_content, "content"):
            response_content = response_content.content

        logger.debug(f"Generated info request email content: {response_content}")
        message = {
            "role": "agent",
            "content": response_content,
            "type": "info_request",
            "requires_review": True,
        }
        current_messages = state.get("messages", [])
        return {"messages": current_messages + [message], "requires_review": True}
    except Exception as e:
        logger.error(f"Error generating info request: {e}", exc_info=True)
        current_messages = state.get("messages", [])
        return {
            "messages": current_messages
            + [
                {
                    "role": "system",
                    "content": f"Error generating info request: {e}",
                    "type": "error",
                    "requires_review": True,
                }
            ],
            "requires_review": True,
        }


def check_for_discounts(state: AgentState, llm: BaseLLMProvider) -> Dict[str, Any]:
    """Node to check for potential discounts based on history and info."""
    logger.info("Checking for discounts...")
    thread_formatted = format_email_thread(state["email_thread"])
    # Ensure customer_info is a dict for json.dumps, even if it came back as None from analyze
    customer_info_json = json.dumps(state.get("customer_info") or {})

    prompt = CHECK_DISCOUNTS_PROMPT.format(
        email_thread_formatted=thread_formatted, customer_info=customer_info_json
    )

    try:
        response = llm.generate_sync(prompt)
        logger.debug(f"Raw discount check response from LLM: {response}")
        if isinstance(response, str):
            result = json.loads(response)
        elif hasattr(response, "content") and isinstance(response.content, str):
            result = json.loads(response.content)
        elif isinstance(response, dict):
            result = response
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")

        status = result.get("status", "error")
        logger.info(f"Discount result: {result}")

        updates = {"discount_status": status}
        current_messages = state.get("messages", [])
        new_message = None
        requires_review_update = False

        if result.get("message_to_customer"):
            new_message = {
                "role": "agent",
                "content": result["message_to_customer"],
                "type": "discount_query",  # Changed type? Or use proof_request?
                "requires_review": True,
            }
            requires_review_update = True
        # We removed the explicit 'proof_needed' message generation from here,
        # as the prompt now asks the LLM to generate the message directly if needed.
        # elif status == "proof_needed": ... (removed)

        if new_message:
            updates["messages"] = current_messages + [new_message]
            updates["requires_review"] = requires_review_update
        else:
            updates["messages"] = current_messages
            updates["requires_review"] = state.get("requires_review", False)

        return updates
    except Exception as e:
        logger.error(f"Error checking for discounts: {e}", exc_info=True)
        # Preserve state on error
        return {
            "discount_status": "error",
            "messages": state.get("messages", []),
            "requires_review": state.get("requires_review", False),
        }


def generate_quote(state: AgentState, llm: BaseLLMProvider) -> Dict[str, Any]:
    """Node to generate a mock quote and review summary using full context."""
    logger.info("Generating quote...")
    mock_quote = {
        "premium": 1234.56,
        "coverage_details": "Standard PPA Coverage",
        "term": "6 months",
        "based_on": state.get("customer_info") or {},
        "discount_applied": state.get("proof_of_discount", False),
    }
    thread_formatted = format_email_thread(state["email_thread"])
    customer_info_json = json.dumps(state.get("customer_info") or {})
    missing_info_json = json.dumps(state.get("missing_info") or [])
    messages_json = json.dumps(state.get("messages") or [])  # Internal agent messages
    quote_data_json = json.dumps(mock_quote)

    prompt = PREPARE_REVIEW_PROMPT.format(
        email_thread_formatted=thread_formatted,
        customer_info=customer_info_json,
        missing_info=missing_info_json,
        messages=messages_json,
        quote_data=quote_data_json,
    )

    try:
        response_content = llm.generate_sync(prompt)
        if hasattr(response_content, "content"):
            response_content = response_content.content

        logger.debug(f"Generated quote summary content: {response_content}")
        current_messages = state.get("messages", [])
        updates = {
            "quote_data": mock_quote,
            "quote_ready": True,
            "messages": current_messages
            + [
                {
                    "role": "agent",
                    "content": response_content,
                    "type": "quote_summary_for_review",
                    "requires_review": True,
                }
            ],
            "requires_review": True,
            "status": "quote_generated",
        }
        logger.info("Quote result: {'status': 'quote_generated'}")
        return updates

    except Exception as e:
        logger.error(f"Error generating quote: {e}", exc_info=True)
        # Preserve state on error
        return {
            "status": "error",
            "messages": state.get("messages", []),
            "requires_review": state.get("requires_review", False),
        }


def prepare_agency_review(state: AgentState, llm: BaseLLMProvider) -> Dict[str, Any]:
    """Node to finalize state for agency review."""
    logger.info("Preparing agency review...")
    # This node's logic might simplify now, as prompts generate summaries earlier.
    # Its main role is ensuring the review flag is set correctly before ending (for now).

    if state.get("requires_review"):
        logger.info("State already marked for review.")
        # Determine final status based on the message that triggered review
        last_message_type = state["messages"][-1]["type"] if state.get("messages") else "unknown"
        status = "ready_for_review"
        if last_message_type == "error":
            status = "error"
        return {"status": status}
    else:
        # This path shouldn't ideally be taken if previous nodes correctly set the flag
        logger.warning(
            "prepare_agency_review reached but requires_review is False. Setting status based on previous node."
        )
        # Use the status set by the previous node (e.g., 'quote_generated', 'no_proof_needed') if available
        # Or default to ready_for_review if we somehow got here.
        status = state.get("status", "ready_for_review") or "ready_for_review"
        return {"requires_review": True, "status": status}


def process_customer_response(state: AgentState, llm: BaseLLMProvider) -> Dict[str, Any]:
    """Processes the latest customer email reply in the context of the thread."""
    logger.info("Processing customer response...")
    thread_formatted = format_email_thread(state["email_thread"])
    # Use requirements from config
    requirements_json = json.dumps(config.PPA_REQUIREMENTS, indent=2)

    # Find the last agent message to provide context
    last_agent_message = "No previous agent message found."
    # Iterate backwards through the thread history copy
    for msg in reversed(state.get("email_thread", [])):
        if msg.get("role") == "agent":
            last_agent_message = msg.get("content", "")
            break

    prompt = PROCESS_RESPONSE_PROMPT.format(
        email_thread_formatted=thread_formatted,
        last_agent_message=last_agent_message,
        requirements=requirements_json,
    )

    try:
        response = llm.generate_sync(prompt)
        logger.debug(f"Raw response processing result from LLM: {response}")
        if isinstance(response, str):
            result = json.loads(response)
        elif hasattr(response, "content") and isinstance(response.content, str):
            result = json.loads(response.content)
        elif isinstance(response, dict):
            result = response
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")

        logger.info(f"Parsed customer response analysis: {result}")

        updates: Dict[str, Any] = {}

        # Update customer_info with newly provided info
        provided_info = result.get("provided_info", {})
        if provided_info:
            # Merge new info into existing customer_info
            current_customer_info = state.get("customer_info", {}).copy()
            for key, value in provided_info.items():
                if value:  # Only update if LLM provided a non-empty value
                    current_customer_info[key] = value
            updates["customer_info"] = current_customer_info
        else:
            # Ensure customer_info is preserved if nothing new was provided
            updates["customer_info"] = state.get("customer_info", {})

        # Update missing_info based on LLM's calculation
        if "updated_missing_info" in result:
            updates["missing_info"] = result["updated_missing_info"]
            # Recalculate status based on newly updated missing_info
            current_status = "info_incomplete" if updates["missing_info"] else "info_complete"
            updates["status"] = current_status
        else:
            # If LLM failed to provide updated_missing_info, recalculate manually
            current_info = updates.get("customer_info", state.get("customer_info", {}))
            # Use requirements from config
            current_missing = [req for req in config.PPA_REQUIREMENTS if not current_info.get(req)]
            updates["missing_info"] = current_missing
            updates["status"] = "info_incomplete" if current_missing else "info_complete"
            logger.warning("LLM did not provide updated_missing_info, recalculated manually.")

        # Update discount proof status if answered
        if result.get("answered_discount_proof") is not None:
            updates["proof_of_discount"] = result["answered_discount_proof"]
            logger.info(f"Customer answered discount proof: {updates['proof_of_discount']}")

        # Handle customer questions (for future routing or response generation)
        if result.get("customer_question"):
            logger.info(f"Customer asked a question: {result['customer_question']}")
            # We might add this question to the messages list or handle it specifically
            # For now, just log it. Could set a specific status too.
            updates["customer_question"] = result["customer_question"]
            updates["status"] = "question_asked"  # Example new status for routing

        # Reset requires_review flag as we've processed the response
        updates["requires_review"] = False
        # Clear agent messages from the previous turn
        updates["messages"] = []

        logger.info(f"State updates after processing response: {updates}")
        return updates

    except Exception as e:
        logger.error(f"Error processing customer response: {e}", exc_info=True)
        # Preserve existing state but signal error
        return {
            "status": "error_processing_response",
            "customer_info": state.get("customer_info", {}),
            "missing_info": state.get("missing_info", []),
            "messages": state.get("messages", []),
            "requires_review": state.get("requires_review", False),
        }
