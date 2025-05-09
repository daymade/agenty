from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # Core state related to the email thread and processing
    thread_id: str
    customer_email: str  # The most recent customer email content
    email_thread: List[Dict[str, Any]]  # Chronological list of messages {role, content, timestamp, type?}
    intent: Optional[str]  # Detected intent (e.g., new_business, inquiry)
    lob: Optional[str]  # Detected Line of Business (e.g., PPA)
    status: Optional[str]  # Overall status (e.g., info_incomplete, info_complete, quote_generated, question_asked)

    # PPA specific state
    customer_info: Dict[str, Any]  # Extracted customer/vehicle info {field: value}
    missing_info: List[str]  # List of fields still needed
    ppa_requirements: List[str]  # List of all required fields for a PPA quote
    quote_ready: bool  # Flag indicating if enough info is present for quoting
    quote_data: Optional[Dict[str, Any]]  # Generated quote details

    # Discount related state
    discount_status: Optional[str]  # Status of discount check (e.g., none_mentioned, proof_needed, validated, rejected)
    proof_of_discount: Optional[str]  # Name of discount if proof provided/validated
    customer_question: Optional[str]  # Specific question asked by customer in last email

    # Agent interaction state
    messages: List[Dict[str, Any]]  # List of messages generated by the agent in the current turn {role, content, type, timestamp, requires_review?}

    # HITL V2 State
    step_requiring_review: Optional[str]  # Name of the step needing human review (e.g., 'generate_quote')
    data_for_review: Optional[Dict[str, Any]]  # Data context needed for the human reviewer
    last_human_decision_for_step: Dict[str, str]  # Record of last decision per step {step_name: 'accepted'/'rejected'}
    rejection_feedback_for_step: Dict[str, str]  # Feedback if rejected {step_name: feedback_text}
    retry_counts: Dict[str, int]  # Track retries per step {step_name: count}
