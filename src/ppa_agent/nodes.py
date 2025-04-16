"""
Defines the nodes (functions) used in the PPA Agent LangGraph workflow.
"""

import json
import logging
from typing import Any, Dict, List, cast
from datetime import datetime

# Import requirements from config.py
from . import config
# Use BaseLLMProvider for type hinting
from .llm_providers import BaseLLMProvider
# Import state from state.py
from .state import AgentState # noqa
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
# Ensure prompts request ONLY JSON object output
IDENTIFY_INTENTION_PROMPT = """
Identify the primary intention of the *latest* customer message in the following conversation history.
Consider the context of previous messages if needed.

Conversation History (most recent message last):
{email_thread_formatted}

You MUST respond with ONLY a JSON object in this exact format:
{{
    "intent": "new_business" | "policy_change" | "provide_info" | "ask_question" | "other"
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
Also, determine if the LATEST customer message contains a question that is not just providing information. If so, include the question text.

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
    "missing_info": [string],
    "detected_customer_question": string | null
}}

Aggregate all information provided by the customer throughout the history. If information for a key is not found anywhere in the customer's messages, use null as the value for that key in customer_info. List the keys for requirements that were *never* mentioned by the customer in missing_info.
If the latest customer message contains a question, put the full question text in `detected_customer_question`, otherwise use null.
Do not include any other text, markdown, or formatting.
"""

# Note: GENERATE_INFO_REQUEST_PROMPT expects plain text, not JSON
GENERATE_INFO_REQUEST_PROMPT = """
You are an AI assistant helping an insurance agent. Your task is to request missing information needed for a Personal Private Auto (PPA) quote based on the conversation history and currently extracted info.

Conversation History:
{email_thread_formatted}

Extracted Customer Info:
{customer_info}

PPA Requirements (all needed):
{ppa_requirements}

Based on the current state where the following information is still missing: {missing_info}
Write a polite email asking the customer to provide these specific details so you can generate an accurate quote.
Keep the email concise and clear. Reference the ongoing conversation implicitly (don't explicitly say "based on our previous messages").
Example: "To continue with your PPA quote request, could you please provide the following: [list]?"
Respond ONLY with the body of the email.
"""

CHECK_DISCOUNTS_PROMPT = """
You are an AI assistant helping an insurance agent. Your task is to analyze a customer conversation and determine the status of potential discounts for a Personal Private Auto (PPA) policy.

Conversation History:
{email_thread_formatted}

Extracted Customer Info:
{customer_info}

Current Discount Status: {current_discount_status}

Checked Discount Name: {checked_discount_name}

Your Tasks:

1.  **If Current Discount Status is 'proof_needed':**
    *   Examine the LATEST customer message in the Conversation History.
    *   Did the customer provide the necessary proof for the discount (`{checked_discount_name}`) that was previously requested? Base this ONLY on the latest message content (e.g., mentioning an attachment, providing details related to the proof needed).
    *   If yes: Respond with status 'validated'.
    *   If no: Respond with status 'no_change' (we are still waiting).

2.  **If Current Discount Status is NOT 'proof_needed' (i.e., it's None or 'not_applicable' or 'validated'):**
    *   Review the Conversation History AND Extracted Customer Info for potential PPA discounts (e.g., Good Student, Multi-Car, Safety Features, Low Mileage, specific affiliations mentioned).
    *   If a potential discount is identified AND requires proof that has NOT been provided yet: Respond with status 'proof_needed' and the 'checked_discount_name'.
    *   If a potential discount is identified BUT it clearly does NOT apply based on other Extracted Customer Info (e.g., age doesn't match Good Student criteria): Respond with status 'not_applicable' and the 'checked_discount_name'.
    *   If no potential discounts requiring proof are found OR all found discounts are already validated or not applicable: Respond with status 'no_change'.

3.  **Override based on Latest Message Action (IMPORTANT):**
    *   REGARDLESS of the logic in step 2, examine the LATEST customer message VERY carefully.
    *   If the customer explicitly states they are providing proof for a specific discount (e.g., "attached is my proof", "here is the info for X discount") OR asks HOW to provide proof for a specific discount:
        *   Identify that specific `checked_discount_name` if possible.
        *   Set the `status` to `proof_needed`.
        *   Do NOT set status to `not_applicable` in this specific case, even if other extracted info (like age) seems to contradict eligibility for that discount. Prioritize the customer's explicit action regarding proof.
        *   Set `message_to_customer` to null unless a direct clarification is absolutely needed.

4.  **Message Generation:** ONLY generate a `message_to_customer` if the status is becoming 'proof_needed' for the first time, asking politely for the specific proof required for the 'checked_discount_name'. Keep it concise.

You MUST respond with ONLY a JSON object in this exact format:
{{
    "status": "validated" | "proof_needed" | "not_applicable" | "no_change",
    "checked_discount_name": string | null,
    "message_to_customer": string | null
}}
"""

# Note: PREPARE_REVIEW_PROMPT expects plain text, not JSON
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
Analyze the latest customer message in the context of the full conversation history AND the agent's immediately preceding request/question (if any). Determine if the customer provided information, answered a question, or asked something new.

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
    "answered_discount_proof": boolean | null,
    "customer_question": string | null, # If they asked a question
    "updated_missing_info": [string] # List of *remaining* missing PPA requirements after considering the reply
}}

Instructions:
1.  Examine the *latest customer message* carefully.
2.  Populate `provided_info` ONLY with *new values* for the `Required PPA Information Fields` mentioned in the latest message.
3.  Check if the `Agent's Last Request/Question` was specifically asking for confirmation to verify a discount (e.g., "may we verify..."). If it was, AND the customer's latest message gives an *affirmative response* (e.g., "Yes", "Sure", "Okay", "Please proceed", "I can provide that"), set `answered_discount_proof` to `true`. If they respond negatively or evasively to the verification request, set it to `false`. If the last agent message wasn't asking for discount verification, set `answered_discount_proof` to `null`.
4.  If the customer asks a clear question in their latest message, capture it in `customer_question`. Otherwise, set it to `null`.
5.  Determine the `response_type` based on the primary action in the customer's latest message (prioritize `answered_question` if applicable).
6.  Calculate `updated_missing_info` by checking which `Required PPA Information Fields` are still null after considering all `provided_info` from the entire conversation history (including the latest message).

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
    thread_formatted = format_email_thread(state["email_thread"])
    prompt = IDENTIFY_INTENTION_PROMPT.format(email_thread_formatted=thread_formatted)

    try:
        # Use generate_sync with structured=True for JSON output
        response_str = llm.generate_sync(prompt, structured=True)
        logger.debug(f"Raw intention response from LLM: {response_str}")
        result = json.loads(response_str)

        intent = result.get("intent", "other")
        logger.info(f"Intention result: {{'intent': '{intent}'}}")
        return {"intent": intent}
    except (json.JSONDecodeError, ValueError, Exception) as e:
        logger.error(f"Error identifying intention: {e}", exc_info=True)
        return {"intent": "error"}


def identify_lob(state: AgentState, llm: BaseLLMProvider) -> Dict[str, str]:
    """Node to identify the line of business from the *entire thread*."""
    logger.info("Identifying line of business...")
    thread_formatted = format_email_thread(state["email_thread"])
    prompt = IDENTIFY_LOB_PROMPT.format(email_thread_formatted=thread_formatted)

    try:
        # Use generate_sync with structured=True for JSON output
        response_str = llm.generate_sync(prompt, structured=True)
        logger.debug(f"Raw LOB response from LLM: {response_str}")
        result = json.loads(response_str)

        lob = result.get("lob", "UNKNOWN")
        logger.info(f"LOB result: {{'lob': '{lob}'}}")
        return {"lob": lob}
    except (json.JSONDecodeError, ValueError, Exception) as e:
        logger.error(f"Error identifying LOB: {e}", exc_info=True)
        return {"lob": "error"}


def analyze_information(state: AgentState, llm: BaseLLMProvider) -> Dict[str, Any]:
    """Node to analyze customer information from the *entire thread*, handling retries."""
    step_name = "analyze_information"
    logger.info(f"Executing node: {step_name}...")
    
    # --- HITL v2 Retry Logic --- 
    decisions = state.get("last_human_decision_for_step", {})
    feedback_dict = state.get("rejection_feedback_for_step", {})
    retries = state.get("retry_counts", {})
    feedback_prompt_addition = ""
    # Get the count *before* this attempt
    previous_retry_count = retries.get(step_name, 0)
    current_retry_count = previous_retry_count # Initialize for logging

    if decisions.get(step_name) == "rejected":
        logger.warning(f"Retrying step '{step_name}' due to previous rejection.")
        feedback = feedback_dict.get(step_name)
        if feedback:
            logger.info(f"Incorporating feedback: {feedback}")
            # Ensure feedback is clearly separated and marked
            feedback_prompt_addition = (
                f"\n\n--- HUMAN REVIEW FEEDBACK ---\n"
                f"The previous analysis was rejected. Please revise based on this feedback: "
                f"'{feedback}'\n"
                f"--- END FEEDBACK ---"
            )
        # Increment retry count for the state update we will return
        current_retry_count = previous_retry_count + 1
    # --- End HITL v2 Retry Logic ---
    
    logger.info("Analyzing information...")
    requirements_json = json.dumps(config.PPA_REQUIREMENTS, indent=2)
    thread_formatted = format_email_thread(state["email_thread"])

    prompt = ANALYZE_INFO_PROMPT.format(
        requirements=requirements_json, email_thread_formatted=thread_formatted
    ) + feedback_prompt_addition # Add feedback if present

    try:
        # Use generate_sync with structured=True for JSON output
        response_str = llm.generate_sync(prompt, structured=True)
        logger.debug(f"Raw analysis response from LLM: {response_str}")
        result = json.loads(response_str)

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

        # --- Preserve Customer Question --- 
        incoming_question = state.get("customer_question")
        detected_question = result.get("detected_customer_question")

        final_question = None
        if detected_question:
            final_question = detected_question
            logger.info(f"LLM detected new customer question: {final_question}")
        elif incoming_question:
            final_question = incoming_question
            logger.info(f"Preserving incoming customer question: {final_question}")
        
        updates["customer_question"] = final_question # Add/update in state

        # Determine status based on the calculated missing_info
        # TODO: Revisit status logic if a question is present? For now, base on missing info only.
        status = "info_incomplete" if updates["missing_info"] else "info_complete"
        updates["status"] = status

        # --- Prepare state updates including HITL clear/update ---
        updated_decisions = decisions.copy()
        updated_feedback = feedback_dict.copy()
        updated_retries = retries.copy()

        if step_name in updated_decisions:
            del updated_decisions[step_name]
        if step_name in updated_feedback:
            del updated_feedback[step_name]

        updated_retries[step_name] = current_retry_count

        updates["last_human_decision_for_step"] = updated_decisions
        updates["rejection_feedback_for_step"] = updated_feedback
        updates["retry_counts"] = updated_retries

        logger.info(f"Info analysis result: {updates}")
        return updates

    except (json.JSONDecodeError, ValueError, Exception) as e:
        logger.error(f"Error analyzing information: {e}", exc_info=True)
        # Preserve existing HITL state on error, let error handler decide overall state
        return {
            "status": "error",
            "missing_info": state.get("missing_info", []),
            "customer_info": state.get("customer_info", {}),
            "customer_question": state.get("customer_question"), # Preserve customer question
            "last_human_decision_for_step": decisions,
            "rejection_feedback_for_step": feedback_dict,
            "retry_counts": retries,
        }


def generate_info_request(state: AgentState, llm: BaseLLMProvider) -> Dict[str, Any]:
    """Node to generate an email requesting missing information, handling retries."""
    step_name = "generate_info_request"
    logger.info(f"Executing node: {step_name}...")
    
    # --- HITL v2 Retry Logic --- 
    decisions = state.get("last_human_decision_for_step", {})
    feedback_dict = state.get("rejection_feedback_for_step", {})
    retries = state.get("retry_counts", {})
    feedback_prompt_addition = ""
    previous_retry_count = retries.get(step_name, 0)
    current_retry_count = previous_retry_count

    if decisions.get(step_name) == "rejected":
        logger.warning(f"Retrying step '{step_name}' due to previous rejection.")
        feedback = feedback_dict.get(step_name)
        if feedback:
            logger.info(f"Incorporating feedback: {feedback}")
            # Add feedback context for the LLM (it's generating free text)
            feedback_prompt_addition = (
                f"\n\n--- HUMAN REVIEW FEEDBACK ---\n"
                f"The previous request message was rejected. Please revise based on this feedback: "
                f"'{feedback}'\n"
                f"Remember to ONLY generate the email body.\n"
                f"--- END FEEDBACK ---"
            )
        current_retry_count = previous_retry_count + 1
    # --- End HITL v2 Retry Logic ---
    
    logger.info("Generating information request...")
    missing_info_str = ", ".join(state.get("missing_info", []))
    customer_info_json = json.dumps(state.get("customer_info") or {})
    thread_formatted = format_email_thread(state["email_thread"])
    ppa_requirements_str = ", ".join(state.get("ppa_requirements", []))

    # Format the prompt with all required variables
    prompt = GENERATE_INFO_REQUEST_PROMPT.format(
        email_thread_formatted=thread_formatted,
        customer_info=customer_info_json,
        ppa_requirements=ppa_requirements_str,
        missing_info=missing_info_str
    ) + feedback_prompt_addition # Add feedback if present

    try:
        # Use generate_sync with structured=False for plain text email body
        response_text = llm.generate_sync(prompt, structured=False)
        logger.debug(f"Raw info request response from LLM: {response_text}")

        message = {
            "role": "agent",
            "content": response_text,
            "type": "info_request",
            "requires_review": True,
        }
        updates = {"messages": state.get("messages", []) + [message]}

        # --- Prepare state updates including HITL clear/update ---
        updated_decisions = decisions.copy()
        updated_feedback = feedback_dict.copy()
        updated_retries = retries.copy()

        if step_name in updated_decisions:
            del updated_decisions[step_name]
        if step_name in updated_feedback:
            del updated_feedback[step_name]

        updated_retries[step_name] = current_retry_count

        updates["last_human_decision_for_step"] = updated_decisions
        updates["rejection_feedback_for_step"] = updated_feedback
        updates["retry_counts"] = updated_retries

        logger.info(f"Info request generation result: {updates}")
        return updates

    except (ValueError, Exception) as e:
        logger.error(f"Error generating info request: {e}", exc_info=True)
        updates = {
            "messages": state.get("messages", [])
            + [
                {
                    "role": "system",
                    "content": f"Error generating info request: {e}",
                    "type": "error",
                    "requires_review": False, # Error message doesn't need review
                }
            ],
            "status": "error_generating_request", # Signal specific error
            # Preserve existing HITL state on error
            "last_human_decision_for_step": decisions,
            "rejection_feedback_for_step": feedback_dict,
            "retry_counts": retries,
        }
        return updates


def check_for_discounts(state: AgentState, llm: BaseLLMProvider) -> Dict[str, Any]:
    """Node to check for potential discounts based on history and info, handling retries."""
    step_name = "check_for_discounts"
    logger.info(f"Executing node: {step_name}...")
    
    # --- HITL v2 Retry Logic --- 
    decisions = state.get("last_human_decision_for_step", {})
    feedback_dict = state.get("rejection_feedback_for_step", {})
    retries = state.get("retry_counts", {})
    feedback_prompt_addition = ""
    previous_retry_count = retries.get(step_name, 0)
    current_retry_count = previous_retry_count

    if decisions.get(step_name) == "rejected":
        logger.warning(f"Retrying step '{step_name}' due to previous rejection.")
        feedback = feedback_dict.get(step_name)
        if feedback:
            logger.info(f"Incorporating feedback: {feedback}")
            feedback_prompt_addition = (
                f"\n\n--- HUMAN REVIEW FEEDBACK ---\n"
                f"The previous discount check was rejected. Please revise based on this feedback: "
                f"'{feedback}'\n"
                f"--- END FEEDBACK ---"
            )
        current_retry_count = previous_retry_count + 1
    # --- End HITL v2 Retry Logic ---
    
    logger.info("Checking for discounts...")
    thread_formatted = format_email_thread(state["email_thread"])
    customer_info_json = json.dumps(state.get("customer_info") or {})
    current_discount_status = state.get("discount_status") # Get current status
    checked_discount_name_from_state = state.get("checked_discount_name", "the previously requested discount")

    prompt = CHECK_DISCOUNTS_PROMPT.format(
        email_thread_formatted=thread_formatted,
        customer_info=customer_info_json,
        current_discount_status=str(current_discount_status), # Pass current status
        checked_discount_name=checked_discount_name_from_state # Add the discount name
    ) + feedback_prompt_addition # Add feedback if present

    try:
        # Use generate_sync with structured=True for JSON output
        response_str = llm.generate_sync(prompt, structured=True)
        logger.debug(f"Raw discount check response from LLM: {response_str}")
        result = json.loads(response_str)
        logger.info(f"Parsed discount check result from LLM: {result}") # Log parsed result

        new_status = result.get("status")
        checked_discount = result.get("checked_discount_name")
        new_message_content = result.get("message_to_customer")

        updates = {
            # Update status only if changed meaningfully (not 'no_change')
            "discount_status": new_status if new_status and new_status != "no_change" else current_discount_status,
            # Set proof_of_discount to the name if validated, otherwise keep existing
            "proof_of_discount": checked_discount if new_status == "validated" else state.get("proof_of_discount"),
        }

        # --- Add fallback logic for proof request message ---
        if new_status == "proof_needed" and not new_message_content:
            discount_name = checked_discount or "the relevant discount"
            default_message = f"To apply the {discount_name}, please provide the necessary proof (e.g., documentation, ID card)."
            logger.warning(f"LLM set status to proof_needed but provided no message. Using default: {default_message}")
            new_message_content = default_message
        # --- End fallback logic ---

        # Generate a message *only* if the LLM decides one is needed now (for a *new* discount)
        if new_message_content:
            message = {
                "role": "agent",
                "content": new_message_content,
                "type": "discount_query",
                "requires_review": True, # Keep review for discount queries for now
            }
            # Avoid adding duplicate messages if the content is identical to the last agent message
            last_agent_message = None
            if state.get("messages"):
                for msg in reversed(state.get("messages")):
                    if msg.get('role') == 'agent':
                        last_agent_message = msg
                        break

            if not last_agent_message or last_agent_message.get("content") != new_message_content:
                 updates["messages"] = state.get("messages", []) + [message]
            else:
                logger.warning("Duplicate discount query message detected, not adding.")

        # --- Prepare state updates including HITL clear/update ---
        updated_decisions = decisions.copy()
        updated_feedback = feedback_dict.copy()
        updated_retries = retries.copy()

        if step_name in updated_decisions:
            del updated_decisions[step_name]
        if step_name in updated_feedback:
            del updated_feedback[step_name]

        updated_retries[step_name] = current_retry_count

        updates["last_human_decision_for_step"] = updated_decisions
        updates["rejection_feedback_for_step"] = updated_feedback
        updates["retry_counts"] = updated_retries

        logger.info(f"Discount check result state updates: {updates}")
        return updates

    except (json.JSONDecodeError, ValueError, Exception) as e:
        logger.error(f"Error checking discounts: {e}", exc_info=True)
        # Preserve existing HITL state on error
        return {
            "status": "error_checking_discounts",
            "discount_status": state.get("discount_status"),
            "proof_of_discount": state.get("proof_of_discount"),
            "last_human_decision_for_step": decisions,
            "rejection_feedback_for_step": feedback_dict,
            "retry_counts": retries,
        }

def generate_quote(state: AgentState, llm: BaseLLMProvider) -> Dict[str, Any]:
    """Node to generate a PPA quote (placeholder), handling retries."""
    step_name = "generate_quote"
    logger.info(f"Executing node: {step_name}...")
    
    # --- HITL v2 Retry Logic --- 
    decisions = state.get("last_human_decision_for_step", {})
    feedback_dict = state.get("rejection_feedback_for_step", {})
    retries = state.get("retry_counts", {})
    feedback_note = ""
    previous_retry_count = retries.get(step_name, 0)
    current_retry_count = previous_retry_count

    if decisions.get(step_name) == "rejected":
        logger.warning(f"Retrying step '{step_name}' due to previous rejection.")
        feedback = feedback_dict.get(step_name)
        if feedback:
            logger.info(f"Incorporating feedback: {feedback}")
            # Add a note to the quote message indicating revision
            feedback_note = f"(Revised based on feedback: {feedback})\n\n"
        current_retry_count = previous_retry_count + 1
    # --- End HITL v2 Retry Logic ---
    
    logger.info("Generating quote (placeholder)...")
    # Placeholder logic: Create dummy quote data based on available info
    # In a real scenario, this would involve complex calculations or API calls.
    customer_info = state.get("customer_info", {})
    timestamp = datetime.now().isoformat() # Add timestamp

    # Basic structure for a dummy quote
    quote_data = {
        "policy_number": f"PPA-{hash(json.dumps(customer_info, sort_keys=True)) % 100000}",
        "premium_annual": 1200.50 + (hash(customer_info.get('driver_age', '')) % 100), # Vary premium slightly
        "coverage_details": {
            "bodily_injury": "100k/300k",
            "property_damage": "50k",
            "deductible_collision": 500,
            "deductible_comprehensive": 250,
        },
        "vehicle": f"{customer_info.get('vehicle_year', 'N/A')} {customer_info.get('vehicle_make', 'N/A')} {customer_info.get('vehicle_model', 'N/A')}",
        "driver": f"{customer_info.get('driver_name', 'N/A')} (Age: {customer_info.get('driver_age', 'N/A')})",
        "quote_timestamp": timestamp,
        # Prepend feedback note if applicable
        "message": feedback_note + f"Great news! We've generated a preliminary quote for your {customer_info.get('vehicle_year', 'N/A')} {customer_info.get('vehicle_make', 'N/A')} {customer_info.get('vehicle_model', 'N/A')}. Your estimated annual premium is ${1200.50 + (hash(customer_info.get('driver_age', '')) % 100):,.2f}. This quote includes standard coverages (Bodily Injury: 100k/300k, Property Damage: 50k, Collision Deductible: $500, Comprehensive Deductible: $250). Please let us know if you'd like to adjust coverages or proceed."
    }

    logger.info(f"Generated placeholder quote data and flagged for review: {quote_data}")

    try:
        # Prepare initial updates
        updates = {
            "quote_data": quote_data,
            "quote_ready": True,
            "status": "quote_generated",
            # Preserve messages from previous steps
            "messages": state.get("messages", []).copy() 
        }

        # Add the generated quote message to the list (for context/display later)
        quote_message_content = quote_data.get("message")
        if quote_message_content:
            updates["messages"].append(
                {
                    "role": "agent",
                    "content": quote_message_content,
                    "type": "quote_generated",
                    # This specific message doesn't require review; 
                    # the *state change* triggers the prepare_review node.
                    "requires_review": False, 
                }
            )
        else:
            logger.warning("Generated quote data missing 'message' key.")

        # --- Prepare state updates including HITL clear/update ---
        updated_decisions = decisions.copy()
        updated_feedback = feedback_dict.copy()
        updated_retries = retries.copy()

        if step_name in updated_decisions:
            del updated_decisions[step_name]
        if step_name in updated_feedback:
            del updated_feedback[step_name]

        updated_retries[step_name] = current_retry_count

        updates["last_human_decision_for_step"] = updated_decisions
        updates["rejection_feedback_for_step"] = updated_feedback
        updates["retry_counts"] = updated_retries

        # The review itself is triggered by the graph structure leading to 
        # a 'prepare_review' node based on the 'quote_generated' status.
        # Do NOT set requires_review=True here.
        
        logger.info(f"Quote generation result: {updates}")
        return updates
        
    except Exception as e:
        logger.error(f"Error generating quote: {e}", exc_info=True)
        # Preserve existing HITL state on error
        return {
            "status": "error_generating_quote",
            "quote_data": state.get("quote_data"),
            "quote_ready": False,
            "messages": state.get("messages", []),
            "last_human_decision_for_step": decisions,
            "rejection_feedback_for_step": feedback_dict,
            "retry_counts": retries,
        }


def prepare_info_review(state: AgentState) -> Dict[str, Any]:
    """Prepares the state for human review after the 'analyze_information' step."""
    step_reviewed = "analyze_information"
    logger.info(f"Preparing review data for step: {step_reviewed}")

    # Extract relevant data generated by analyze_information
    customer_info = state.get("customer_info", {})
    missing_info = state.get("missing_info", [])
    current_status = state.get("status")

    review_data = {
        "review_target": step_reviewed,
        "customer_info_analysis": customer_info,
        "missing_info_analysis": missing_info,
        "calculated_status": current_status,
        "context_email_thread": state.get("email_thread", []) # Provide context
    }

    logger.debug(f"Data prepared for review: {review_data}")

    return {
        "step_requiring_review": step_reviewed,
        "data_for_review": review_data,
        # Ensure status isn't overwritten if it was 'error'
        "status": current_status 
    }


def prepare_discount_review(state: AgentState) -> Dict[str, Any]:
    """Prepares the state for human review after the 'check_for_discounts' step."""
    step_reviewed = "check_for_discounts"
    logger.info(f"Preparing review data for step: {step_reviewed}")

    # Extract relevant data generated by check_for_discounts
    discount_status = state.get("discount_status")
    proof_of_discount = state.get("proof_of_discount")
    messages = state.get("messages", [])
    # Find the message generated by the check_discounts node, if any
    discount_query_message = None
    if messages and messages[-1].get("type") == "discount_query":
        discount_query_message = messages[-1].get("content")

    review_data = {
        "review_target": step_reviewed,
        "discount_status_analysis": discount_status,
        "proof_of_discount_status": proof_of_discount,
        "generated_discount_query": discount_query_message,
        "context_customer_info": state.get("customer_info", {}),
        "context_email_thread": state.get("email_thread", []) # Provide context
    }

    logger.debug(f"Data prepared for review: {review_data}")

    return {
        "step_requiring_review": step_reviewed,
        "data_for_review": review_data,
        # Preserve status
        "status": state.get("status")
    }


def prepare_quote_review(state: AgentState) -> Dict[str, Any]:
    """Prepares the state for human review after the 'generate_quote' step."""
    step_reviewed = "generate_quote"
    logger.info(f"Preparing review data for step: {step_reviewed}")

    # Extract relevant data generated by generate_quote
    quote_data = state.get("quote_data")

    if not quote_data:
        logger.error(f"Cannot prepare review for '{step_reviewed}': quote_data is missing.")
        # Return error state or handle appropriately? For now, log and don't set review.
        return {
            "step_requiring_review": None,
            "data_for_review": None,
            "status": "error_preparing_review" # Signal specific error
        }

    review_data = {
        "review_target": step_reviewed,
        "generated_quote_details": quote_data,
        "context_customer_info": state.get("customer_info", {}),
        "context_email_thread": state.get("email_thread", []) # Provide context
    }

    logger.debug(f"Data prepared for review: {review_data}")

    return {
        "step_requiring_review": step_reviewed,
        "data_for_review": review_data,
        # Preserve status (should be 'quote_generated')
        "status": state.get("status")
    }


def prepare_info_request_review(state: AgentState) -> Dict[str, Any]:
    """Prepares the state for human review after the 'generate_info_request' step."""
    step_reviewed = "generate_info_request"
    logger.info(f"Preparing review data for step: {step_reviewed}")

    # Extract the generated info request message
    messages = state.get("messages", [])
    info_request_content = None
    if messages and messages[-1].get("type") == "info_request":
        info_request_content = messages[-1].get("content")
    else:
        logger.error(f"Cannot prepare review for '{step_reviewed}': Latest message is not of type 'info_request'.")
        return {
            "step_requiring_review": None,
            "data_for_review": None,
            "status": "error_preparing_review" # Signal specific error
        }

    if not info_request_content:
         logger.error(f"Cannot prepare review for '{step_reviewed}': info_request message content is missing.")
         return {
            "step_requiring_review": None,
            "data_for_review": None,
            "status": "error_preparing_review" # Signal specific error
        }

    review_data = {
        "review_target": step_reviewed,
        "generated_info_request_message": info_request_content,
        "context_missing_info": state.get("missing_info", []),
        "context_email_thread": state.get("email_thread", []) # Provide context
    }

    logger.debug(f"Data prepared for review: {review_data}")

    return {
        "step_requiring_review": step_reviewed,
        "data_for_review": review_data,
        # Preserve status
        "status": state.get("status")
    }


def prepare_agency_review(state: AgentState, llm: BaseLLMProvider) -> Dict[str, Any]:
    """Node to prepare a final summary or specific data for agency review. (DEPRECATED by granular review nodes)"""
    logger.info("Preparing data for agency review...")
    last_message = state["messages"][-1] if state["messages"] else None

    if not last_message:
        logger.error("No last message found to prepare for review.")
        # This node is deprecated, but returning empty HITLv2 state for safety
        return {
            "step_requiring_review": None,
            "data_for_review": None,
        }

    review_data = {}
    review_type = "general_review"
    if last_message.get("type") == "quote_generated":
        review_type = "quote_summary_for_review"

    logger.info(f"Prepared data for {review_type}")
    # This node is deprecated. Return empty HITLv2 state.
    return {
        "step_requiring_review": None,
        "data_for_review": None,
    }


def process_customer_response(state: AgentState, llm: BaseLLMProvider) -> Dict[str, Any]:
    """Processes the latest customer email reply in the context of the thread."""
    logger.info("Processing customer response...")
    thread_formatted = format_email_thread(state["email_thread"])
    requirements_json = json.dumps(config.PPA_REQUIREMENTS, indent=2)

    last_agent_message = "No previous agent message found."
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
        # Use generate_sync with structured=True for JSON output
        response_str = llm.generate_sync(prompt, structured=True)
        logger.debug(f"Raw response processing result from LLM: {response_str}")
        result = json.loads(response_str)

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

        logger.info(f"State updates after processing response: {updates}")
        return updates

    except (json.JSONDecodeError, ValueError, Exception) as e:
        logger.error(f"Error processing customer response: {e}", exc_info=True)
        # Preserve existing state but signal error
        return {
            "status": "error_processing_response",
            "customer_info": state.get("customer_info", {}),
            "missing_info": state.get("missing_info", []),
            "messages": state.get("messages", []),
            "requires_review": state.get("requires_review", False),
        }

def final_state_passthrough(state: AgentState) -> Dict[str, Any]:
    """Dummy node to force one last state resolution before END."""
    logger.info(f"Executing node: final_state_passthrough...")
    logger.info(f"State entering passthrough: step_requiring_review={state.get('step_requiring_review')}")
    # Add more detailed logging if needed:
    # logger.debug(f"Full state entering passthrough: {state}")
    return {} # Return no updates
