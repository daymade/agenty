"""Defines the state structure for the PPA Agent."""

from typing import Any, Dict, List, Optional, TypedDict


# Moved from agent.py
class AgentState(TypedDict):
    """State maintained throughout the agent's execution."""

    # Core Conversation Data
    thread_id: Optional[str]  # Unique ID for the conversation thread
    customer_email: str  # Raw content of the *latest* customer email received
    email_thread: List[
        Dict[str, Any]
    ]  # History of emails in this thread ({'role': 'customer'/'agent', 'content': ...})

    # Extracted & Derived Information
    customer_info: Dict[str, Any]  # Information extracted about the customer/vehicle
    missing_info: List[str]  # List of required fields still missing
    ppa_requirements: List[str]  # List of fields needed for a PPA quote
    quote_ready: bool  # Flag indicating if a quote can be generated
    quote_data: Optional[Dict[str, Any]]  # Holds the generated (mock) quote data
    proof_of_discount: Optional[bool]  # Status of discount proof (True/False/None if unknown)

    # Agent Messages & Status
    messages: List[
        Dict[str, Any]
    ]  # History of agent-generated messages (internal and drafts for customer)
    requires_review: bool  # Flag indicating if the current state requires human review

    # Internal state for routing/tracking
    intent: Optional[str]
    lob: Optional[str]
    status: Optional[str]  # e.g., info_complete, info_incomplete, quote_generated
    discount_status: Optional[str]  # e.g., proof_needed, no_proof_needed
    # Added for handling questions if needed
    customer_question: Optional[str]


# PPA_REQUIREMENTS constant moved to config.py
