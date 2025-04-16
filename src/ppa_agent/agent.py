"""
PPA New Business AI Agent implementation.

This module implements a LangGraph-based agent for handling new PPA insurance business inquiries.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Literal, Optional, cast
import copy

from langgraph.graph import END, StateGraph

from . import config
from .nodes import ( # noqa
    analyze_information,
    check_for_discounts,
    generate_info_request,
    generate_quote,
    identify_intention,
    identify_lob,
    process_customer_response,
    prepare_quote_review,
    prepare_info_request_review,
    final_state_passthrough, # Import new node
)
from .llm_providers import (
    BaseLLMProvider,
    GeminiConfig,
    GeminiProvider,
    OpenAIProvider
)
from .state import AgentState # noqa

logger = logging.getLogger(__name__)

HANDLE_MAX_RETRIES = "handle_max_retries"
# Node names for graph
ANALYZE_INFORMATION_NODE = "analyze_information_node"
CHECK_DISCOUNTS_NODE = "check_discounts_node"
GENERATE_QUOTE_NODE = "generate_quote_node"
GENERATE_INFO_REQUEST_NODE = "generate_info_request_node"
IDENTIFY_INTENTION_NODE = "identify_intention_node"
IDENTIFY_LOB_NODE = "identify_lob_node"
PROCESS_CUSTOMER_RESPONSE_NODE = "process_customer_response_node"
ERROR_NODE = "error_node"

PREPARE_QUOTE_REVIEW_NODE = "prepare_quote_review_node"
PREPARE_INFO_REQUEST_REVIEW_NODE = "prepare_info_request_review_node"
FINAL_STATE_PASSTHROUGH_NODE = "final_state_passthrough_node" # Define name

MAX_RETRIES = 2

class PPAAgent:
    """Agent for handling PPA insurance new business inquiries."""

    def __init__(
        self,
        provider: str = "gemini",
        model: Optional[str] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
    ) -> None:
        """Initialize the PPA Agent."""
        self.conversation_threads: Dict[str, Dict[str, Any]] = {}
        
        # Set up state for thread tracking - will be used with LangGraph checkpointing
        # We'll use thread_id as the key in the checkpointing mechanism
        self.threads = {}
        logger.info("Initialized agent with thread tracking")
        
        if llm_provider:
            self.llm: BaseLLMProvider = llm_provider
        else:
            provider_type = provider.lower()
            if provider_type == "gemini":
                gemini_config = GeminiConfig.from_env()
                model_to_use = model or gemini_config.model
                logger.info(f"Initializing GeminiProvider with model: {model_to_use}")
                self.llm = GeminiProvider(
                    api_key=gemini_config.api_key,
                    model=model_to_use
                )
            elif provider_type == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                model_name = model or config.OPENAI_MODEL_NAME
                if not api_key:
                    raise ValueError("OpenAI API key not found in environment variables.")
                logger.info(f"Initializing OpenAIProvider with model: {model_name}")
                self.llm = OpenAIProvider(
                    api_key=api_key,
                    model=model_name
                )
            else:
                raise ValueError(f"Unsupported LLM provider type: {provider_type}")

        # Build and compile the workflow with the checkpointer
        self.compiled_workflow = self._build_workflow()

    def _decide_intent_branch(self, state: AgentState) -> str:
        intent = state.get("intent", "error")
        logger.info(f"Routing based on intent: {intent}")
        if intent == "new_business":
            return "identify_lob_node"
        elif intent == "provide_info":
            return "process_customer_response_node"
        elif intent == "error":
            return "error_node"
        else:
            return str(END)

    def _decide_lob_branch(self, state: AgentState) -> str:
        lob = state.get("lob", "error")
        logger.info(f"Routing based on LOB: {lob}")
        
        # Handle non-PPA cases with appropriate message
        if lob != "PPA" and lob != "error":
            message = {
                "role": "agent",
                "content": f"I apologize, but I specialize in Personal Property & Auto (PPA) insurance. For {lob.lower()} insurance inquiries, I recommend speaking with one of our insurance agents who can better assist you with your needs.",
                "type": "non_ppa_response",
                "requires_review": True
            }
            state["messages"] = state.get("messages", []) + [message]
            return str(END)
        
        if lob == "PPA":
            return "analyze_information_node"
        elif lob == "error":
            return "error_node"
        else:
            return str(END)

    def _decide_after_response(self, state: AgentState) -> str:
        """Decides the next step after processing a customer's response."""
        status = state.get("status", "error_processing_response")
        missing_info = state.get("missing_info", [])
        proof_of_discount = state.get("proof_of_discount") # Populated by process_customer_response
        customer_question = state.get("customer_question")
        # Get discount status from the *previous* check node if available
        discount_status_from_check = state.get("discount_status") 

        logger.info(f"Routing after processing response. Status: {status}, Missing Info: {missing_info}, Discount Proof: {proof_of_discount}, Prev Discount Status: {discount_status_from_check}, Question: {bool(customer_question)}")

        if status == "error_processing_response":
            logger.error("Error processing response, routing to error node.")
            return ERROR_NODE # Use constant

        if missing_info:
            # If info became incomplete again (unlikely but possible), ask again
            logger.info(f"Information still missing ({missing_info}), routing to generate_info_request_node.")
            return GENERATE_INFO_REQUEST_NODE # Use constant
        else: # Info is complete
            # Check if discount proof was just provided in this response
            if proof_of_discount is not None:
                # Info complete, but new proof provided -> Re-run discount check to validate
                logger.info("Info complete and discount proof provided, routing to check_discounts_node for validation.")
                return CHECK_DISCOUNTS_NODE # Use constant
            else:
                # Info complete, no new proof provided. Check previous discount status.
                discount_fully_handled = discount_status_from_check in ["no_proof_needed", "validated"]

                if discount_fully_handled:
                    # Info complete and discount situation already handled -> Generate Quote
                    logger.info("Info complete and discount previously handled, routing to generate_quote_node.")
                    if customer_question and "quote" in customer_question.lower():
                        logger.info(f"Customer asked '{customer_question}', proceeding to generate quote as requested.")
                    elif customer_question:
                        # Handle other questions later if needed, proceed with quote for now
                        logger.warning(f"Customer asked '{customer_question}' but proceeding to quote generation. Question may need review later.")
                    # Ensure GENERATE_QUOTE_NODE is a valid target in add_conditional_edges if uncommenting this path
                    # For now, routing to CHECK_DISCOUNTS_NODE as a safe intermediate step if quote generation isn't directly reachable.
                    # return GENERATE_QUOTE_NODE # Uncomment if graph allows direct routing here
                    logger.warning("Routing to check_discounts_node temporarily as GENERATE_QUOTE_NODE might not be directly reachable from PROCESS_CUSTOMER_RESPONSE.")
                    return CHECK_DISCOUNTS_NODE # Safer intermediate route for now
                else:
                    # Info complete, but discount status is still unclear (e.g., query unanswered)
                    # Re-run discount check to potentially ask again or flag for review
                    logger.info("Info complete but discount status unclear after response, routing back to check_discounts_node.")
                    return CHECK_DISCOUNTS_NODE # Use constant

        # Fallback/Error case (should not be reached)
        logger.error("Reached unexpected state in _decide_after_response, routing to review.")
        return PREPARE_AGENCY_REVIEW_NODE # Use constant

    def _decide_after_discount_check(self, state: AgentState) -> str:
        """Decides the next step after checking for discounts (no review step here)."""
        discount_status = state.get("discount_status", "error")
        status = state.get("status") # Get overall status
        # Check if the discount node generated a message (e.g., asking for proof)
        messages = state.get("messages", [])
        # Find the *last* message added by the agent in this state potentially
        last_agent_message = next((m for m in reversed(messages) if m.get("role") == "agent"), None)
        discount_query_generated = last_agent_message and last_agent_message.get("type") == "discount_query"

        logger.info(f"Routing after discount check. Status: {discount_status}, Query Generated: {discount_query_generated}, Overall Status: {status}")

        if discount_status == "error_checking_discounts" or discount_status == "error":
            logger.error("Error occurred during discount check, routing to error node.")
            return ERROR_NODE
        elif discount_status in ["validated", "no_proof_needed"]:
            # Discount resolved, proceed to quote generation
            logger.info("Discount check complete/resolved, routing to generate_quote_node.")
            return GENERATE_QUOTE_NODE
        elif discount_status == "proof_needed" or discount_query_generated:
            # Discount needs proof, or a question was generated.
            # MODIFIED LOGIC: If info is complete, proceed to quote anyway.
            if status == "info_complete":
                logger.info("Info complete, proceeding to generate quote even though discount proof is pending.")
                return GENERATE_QUOTE_NODE
            else:
                # Info is not complete, OR discount needs proof but info isn't complete yet.
                # End the turn to wait for customer response (either proof or more info).
                logger.info("Discount requires proof or query generated, AND info is not yet complete. Ending turn.")
                return END # End the current flow, agent should send msg, wait for reply
        else:
            # Unhandled status, maybe needs clarification? Route to error for safety.
            logger.warning(f"Unhandled discount status '{discount_status}' after check, routing to error node.")
            return ERROR_NODE

    def _decide_after_quote(self, state: AgentState) -> str:
        requires_review = state.get("requires_review", True)
        logger.info(f"Routing after quote generation. Requires Review: {requires_review}")

        if not requires_review:
            logger.info("Quote generated and no review flagged, ending workflow.")
            return str(END)
        else:
            logger.info("Quote generated but review flagged, proceeding to prepare review.")
            return "prepare_agency_review_node"

    def _decide_after_step_review(self, state: AgentState) -> str:
        """Decides the next routing step after human review of a specific step.

        Reads the decision and feedback from the state, checks retry counts,
        and routes either back to the reviewed step (on rejection) or to the
        next logical step (on acceptance).
        """
        step_reviewed = state.get("step_requiring_review")
        decisions = state.get("last_human_decision_for_step", {})
        feedback = state.get("rejection_feedback_for_step", {})
        retries = state.get("retry_counts", {})

        if not step_reviewed or step_reviewed not in decisions:
            logger.error(
                f"Decision function called but no decision found for step '{step_reviewed}'. Routing to END."
            )
            # TODO: Should route to a specific error handling node
            return str(END)

        decision = decisions.get(step_reviewed)
        logger.info(f"Routing after human review for step '{step_reviewed}'. Decision: {decision}")

        # --- Clear review flags before routing ---
        # state["step_requiring_review"] = None  # LangGraph state is immutable within a node
        # state["data_for_review"] = None
        # We need to return updates to the state in the node that *calls* this decision function, or the nodes themselves.
        # This conditional function *only* returns the name of the next node.

        if decision == "accepted":
            logger.info(f"Human accepted output for '{step_reviewed}'. Proceeding.")
            # Logic to determine the *actual* next step based on 'step_reviewed' will go here
            # during the _build_workflow update.
            # Placeholder for now:
            if step_reviewed == "analyze_information":
                # After analyzing info, decide if we need more info or can check discounts
                if state.get("status") == "info_complete":
                    return CHECK_DISCOUNTS_NODE 
                else:
                    return GENERATE_INFO_REQUEST_NODE
            elif step_reviewed == "check_for_discounts":
                # After checking discounts, decide if we need more info or can generate quote
                # This replicates the old _decide_after_discount_check logic path
                if state.get("status") == "discount_query_needed":
                    # If a query was generated, it needs review? Assume yes for now.
                    logger.info("Discount check accepted, query generated. Ending to send message.")
                    return str(END)
                elif state.get("quote_ready"):
                    return GENERATE_QUOTE_NODE
                else: # Should not happen if discount check is done? Maybe request info?
                    logger.warning("Routing unclear after accepted discount check. Going to END.")
                    return str(END)
            elif step_reviewed == "generate_quote":
                logger.info("Quote accepted by human. Ending workflow.")
                return str(END) # Placeholder - End of PPA specific flow after quote acceptance
            elif step_reviewed == "generate_info_request":
                logger.info("Info request accepted by human. Ending workflow.")
                return str(END) # Placeholder - End of PPA specific flow after info request acceptance
            else:
                logger.warning(f"Accepted step '{step_reviewed}' has no defined next step. Routing to END.")
                return str(END)

        elif decision == "rejected":
            current_retry_count = retries.get(step_reviewed, 0)
            feedback_text = feedback.get(step_reviewed, "No feedback provided.")
            logger.warning(
                f"Human rejected output for '{step_reviewed}'. Retry count: {current_retry_count}. Feedback: {feedback_text}"
            )

            if current_retry_count >= MAX_RETRIES:
                logger.error(
                    f"Max retries ({MAX_RETRIES}) reached for step '{step_reviewed}'. Routing to handler."
                )
                return HANDLE_MAX_RETRIES # Route to max retries handler
            else:
                # Increment retry count happens implicitly when state is updated by the calling node/graph
                logger.info(f"Routing back to step '{step_reviewed}' for retry.")
                # Return the name of the node that needs to be retried
                # This mapping will be defined in _build_workflow
                if step_reviewed == "analyze_information":
                    return ANALYZE_INFORMATION_NODE
                elif step_reviewed == "check_for_discounts":
                    return CHECK_DISCOUNTS_NODE
                elif step_reviewed == "generate_quote":
                    return GENERATE_QUOTE_NODE
                elif step_reviewed == "generate_info_request":
                    return GENERATE_INFO_REQUEST_NODE
                else:
                    logger.error(f"Rejected step '{step_reviewed}' cannot be retried. Routing to END.")
                    return str(END)

        else:
            logger.error(f"Unknown decision '{decision}' for step '{step_reviewed}'. Routing to END.")
            return str(END)

    def _decide_after_analyze_info(self, state: AgentState) -> str:
        """Decides next step after analyzing information (no review step here)."""
        status = state.get("status", "error")
        logger.info(f"Routing after info analysis. Status: {status}")
        if status == "info_complete":
            logger.info("Info complete after analysis, routing to check_discounts_node.")
            return CHECK_DISCOUNTS_NODE
        elif status == "info_incomplete":
            logger.info("Info incomplete after analysis, routing to generate_info_request_node.")
            return GENERATE_INFO_REQUEST_NODE
        else: # error or unexpected
            logger.error(f"Unexpected status '{status}' after analyze_information, routing to error_node.")
            return ERROR_NODE

    def _handle_error(self, state: AgentState) -> Dict[str, Any]:
        logger.error(f"Entering error handling node. Current state: {state}")
        error_message = {
            "role": "system",
            "content": "An unexpected error occurred during processing.",
            "type": "final_error",
            "requires_review": False,
        }
        messages = state.get("messages", [])
        messages.append(error_message)
        return {"messages": messages}

    def _handle_max_retries(self, state: AgentState) -> Dict[str, Any]:
        """Handles the situation when max retries are reached for a step."""
        # Determine which step actually failed based on the retry counts or last decision?
        # Let's assume the step that triggered this path is implicitly known by the decision logic 
        # that routed here. We need to find the step name for logging/message.
        # A simple way is to look at the retry counts dict.
        retries = state.get("retry_counts", {})
        failed_step = "unknown_step"
        for step, count in retries.items():
            if count >= MAX_RETRIES:
                failed_step = step
                break
        
        logger.error(f"Max retries ({MAX_RETRIES}) reached for step: {failed_step}. Requires manual intervention.")
        
        error_message = {
                "role": "system",
                "content": f"Agent stopped: Max retries ({MAX_RETRIES}) reached for step '{failed_step}'. Manual handling required.",
                "type": "error",
                "timestamp": datetime.now().isoformat(),
            }
        
        # Update state to reflect this terminal error state
        return {
            "status": "error_max_retries", 
            "messages": state.get("messages", []) + [error_message],
            # Clear review flags as it's no longer waiting for review input
            "step_requiring_review": None,
            "data_for_review": None,
            # Optionally clear decision/feedback for the failed step?
            # No, keep them for diagnosis.
        }

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        # --- Node Function Partials (Binding LLM) ---
        identify_intention_node = partial(identify_intention, llm=self.llm)
        identify_lob_node = partial(identify_lob, llm=self.llm)
        analyze_information_node = partial(analyze_information, llm=self.llm)
        generate_info_request_node = partial(generate_info_request, llm=self.llm)
        check_for_discounts_node = partial(check_for_discounts, llm=self.llm)
        generate_quote_node = partial(generate_quote, llm=self.llm)
        process_customer_response_node = partial(process_customer_response, llm=self.llm)

        # --- Review Node Partials (Binding Step Name) ---
        prepare_quote_review_node = partial(prepare_quote_review)
        prepare_info_request_review_node = partial(prepare_info_request_review)
        final_state_passthrough_node = partial(final_state_passthrough) # Add the new node

        # --- Add Nodes to Graph ---
        workflow.add_node(IDENTIFY_INTENTION_NODE, identify_intention_node)
        workflow.add_node(IDENTIFY_LOB_NODE, identify_lob_node)
        workflow.add_node(ANALYZE_INFORMATION_NODE, analyze_information_node)
        workflow.add_node(GENERATE_INFO_REQUEST_NODE, generate_info_request_node)
        workflow.add_node(CHECK_DISCOUNTS_NODE, check_for_discounts_node)
        workflow.add_node(GENERATE_QUOTE_NODE, generate_quote_node)
        workflow.add_node(PROCESS_CUSTOMER_RESPONSE_NODE, process_customer_response_node)
        workflow.add_node(ERROR_NODE, self._handle_error)
        # HITL Nodes
        workflow.add_node(PREPARE_QUOTE_REVIEW_NODE, prepare_quote_review_node)
        workflow.add_node(PREPARE_INFO_REQUEST_REVIEW_NODE, prepare_info_request_review_node)
        workflow.add_node(FINAL_STATE_PASSTHROUGH_NODE, final_state_passthrough_node) # Add the new node
        workflow.add_node(HANDLE_MAX_RETRIES, self._handle_max_retries)

        # --- Define Graph Edges --- 
        workflow.set_entry_point(IDENTIFY_INTENTION_NODE)

        # Initial routing based on intent and LOB
        workflow.add_conditional_edges(
            IDENTIFY_INTENTION_NODE,
            self._decide_intent_branch,
            {
                "identify_lob_node": IDENTIFY_LOB_NODE,
                "process_customer_response_node": PROCESS_CUSTOMER_RESPONSE_NODE,
                "error_node": ERROR_NODE,
                END: END,
            },
        )
        workflow.add_conditional_edges(
            IDENTIFY_LOB_NODE,
            self._decide_lob_branch,
            {
                "analyze_information_node": ANALYZE_INFORMATION_NODE,
                "error_node": ERROR_NODE,
                END: END,
            },
        )

        # --- Analyze Info -> Decide (No Review Prep) --- 
        workflow.add_conditional_edges(
            ANALYZE_INFORMATION_NODE, # Source node
            self._decide_after_analyze_info, # NEW decision function
            {
                CHECK_DISCOUNTS_NODE: CHECK_DISCOUNTS_NODE,
                GENERATE_INFO_REQUEST_NODE: GENERATE_INFO_REQUEST_NODE,
                ERROR_NODE: ERROR_NODE,
            }
        )

        # --- Check Discounts -> Decide (No Review Prep) ---
        workflow.add_conditional_edges(
            CHECK_DISCOUNTS_NODE, # Source node
            self._decide_after_discount_check, # Use existing (modified) decision function
            {
                GENERATE_QUOTE_NODE: GENERATE_QUOTE_NODE,
                ERROR_NODE: ERROR_NODE,
                END: END # If discount check generated a query, pause here
            }
        )

        # --- Generate Quote -> Prepare Review -> Decide --- 
        # Keep this path for mandatory quote review
        workflow.add_edge(GENERATE_QUOTE_NODE, PREPARE_QUOTE_REVIEW_NODE)
        workflow.add_edge(PREPARE_QUOTE_REVIEW_NODE, FINAL_STATE_PASSTHROUGH_NODE) # Change edge to passthrough node

        # --- Generate Info Request -> Prepare Review -> Decide --- 
        workflow.add_edge(GENERATE_INFO_REQUEST_NODE, PREPARE_INFO_REQUEST_REVIEW_NODE)
        workflow.add_edge(PREPARE_INFO_REQUEST_REVIEW_NODE, FINAL_STATE_PASSTHROUGH_NODE) # Change edge to passthrough node

        # --- Customer Response Handling --- (Ensure GENERATE_QUOTE_NODE is not a direct target)
        workflow.add_conditional_edges(
            PROCESS_CUSTOMER_RESPONSE_NODE, # Where should this go after processing?
            self._decide_after_response,
            {
                # Typically re-analyze info based on new customer input?
                ANALYZE_INFORMATION_NODE: ANALYZE_INFORMATION_NODE,
                # Old paths for reference (may need review)
                # GENERATE_INFO_REQUEST_NODE: GENERATE_INFO_REQUEST_NODE,
                CHECK_DISCOUNTS_NODE: CHECK_DISCOUNTS_NODE,
                # GENERATE_QUOTE_NODE: GENERATE_QUOTE_NODE, 
                ERROR_NODE: ERROR_NODE,
                # END is not a direct target here; subsequent nodes decide END
            } # TODO: Revisit routing from customer response processing
        )

        # --- Error and Max Retries Handling ---
        workflow.add_edge(ERROR_NODE, END)
        workflow.add_edge(HANDLE_MAX_RETRIES, END) # Max retries also terminates flow for now

        # --- Add edge from passthrough node to END ---
        workflow.add_edge(FINAL_STATE_PASSTHROUGH_NODE, END)

        # Compile the workflow - we'll handle checkpointing via the config
        logger.info("Compiling workflow")
        compiled = workflow.compile()
        return compiled

    def process_email(self, email_content: str, thread_id: Optional[str] = None) -> AgentState:
        """Process an incoming customer email, handling conversation threads and state."""
        logger.info(f"Processing email for thread_id: {thread_id or 'New Thread'}")

        current_thread_id: str
        email_history: List[Dict[str, Any]]
        previous_customer_info: Dict[str, Any] = {}

        if thread_id and thread_id in self.conversation_threads:
            current_thread_id = thread_id
            logger.info(f"Resuming existing thread: {current_thread_id}")
            thread_data = self.conversation_threads[current_thread_id]
            email_history = thread_data.get("history", [])
            previous_customer_info = thread_data.get("last_customer_info", {})
        else:
            current_thread_id = uuid.uuid4().hex
            logger.info(f"Creating new thread: {current_thread_id}")
            email_history = []
            self.conversation_threads[current_thread_id] = {
                "history": email_history,
                "last_customer_info": {},
                "last_state": None,
            }

        new_email_entry = {
            "role": "customer",
            "content": email_content,
            "timestamp": datetime.now().isoformat(),
        }
        email_history.append(new_email_entry)

        initial_state: AgentState = {
            "thread_id": current_thread_id,
            "customer_email": email_content,
            "email_thread": email_history.copy(), # Will be updated with AI/Human turns
            "customer_info": previous_customer_info.copy(),
            "missing_info": [],
            "ppa_requirements": config.PPA_REQUIREMENTS,
            "quote_ready": False,
            "quote_data": None,
            "proof_of_discount": None,
            "messages": [],
            "requires_review": False,
            # v2 HITL State Initialization
            "step_requiring_review": None,
            "data_for_review": None,
            "last_human_decision_for_step": {},
            "rejection_feedback_for_step": {},
            "retry_counts": {},
            "intent": None,
            "lob": None,
            "status": "new",
            "discount_status": None,
            "customer_question": None,
        }

        if thread_id and thread_id in self.conversation_threads:
            logger.info(f"Continuing existing thread: {current_thread_id}")
            # Update history for existing thread before invoking
            self.conversation_threads[current_thread_id]["history"] = email_history
            # Load previous state if needed (or just use current info)
            initial_state = initial_state # State contains the new email and thread context
            # For existing threads, start by processing the response
            workflow_config = {"configurable": {"thread_entry_point": "process_customer_response_node"}}
            logger.info(f"Setting workflow entry point to: process_customer_response_node")
        else:
            logger.info(f"Creating new thread: {current_thread_id}")
            self.conversation_threads[current_thread_id] = {
                "history": email_history,
                "last_customer_info": {},
                "last_state": None,
            }
            initial_state = initial_state
            # For new threads, start normally
            workflow_config = None

        # Run the workflow with config and memory for this thread
        # Store state for this thread to enable checkpointing
        if thread_id not in self.threads:
            self.threads[current_thread_id] = {}
            
        # Setup config with checkpointer
        if not workflow_config:
            workflow_config = {}
        
        # Add thread_id to config for checkpointing
        workflow_config["configurable"] = workflow_config.get("configurable", {})
        workflow_config["configurable"]["thread_id"] = current_thread_id
            
        final_state = self.compiled_workflow.invoke(initial_state, config=workflow_config)

        # Log the state immediately after invoke returns
        logger.info(f"Invoke completed for thread {current_thread_id}. State: {final_state}")
        logger.info(f"---> Invoke result review step: {final_state.get('step_requiring_review')}")
        logger.info(f"---> Invoke result status: {final_state.get('status')}")

        # === Workaround: Extract critical value, deepcopy storage, reconstruct return ===
        # 1. Extract the crucial value BEFORE potential modification by LangGraph internals
        original_review_step = final_state.get("step_requiring_review")
        original_data_for_review = final_state.get("data_for_review") # Also save data
        logger.info(f"Extracted original_review_step: {original_review_step}")

        # 2. Create a deep copy IMMEDIATELY for internal storage
        state_for_storage = copy.deepcopy(final_state)
        logger.info("Created deepcopy for internal storage.")

        # 3. Update internal storage using the deep copy
        self.conversation_threads[current_thread_id]["last_state"] = state_for_storage
        logger.info(f"Updated conversation_threads using the DEEP copy for {current_thread_id}")

        # 4. Construct a NEW dictionary for return, ensuring critical values are preserved
        return_state = final_state.copy() # Start with a shallow copy is fine here
        return_state["step_requiring_review"] = original_review_step # Explicitly set saved value
        return_state["data_for_review"] = original_data_for_review # Explicitly set saved value
        logger.info(f"Reconstructed return_state with review step: {return_state.get('step_requiring_review')}")
        # === End Workaround ===

        # Log the final state for debugging again before returning
        # Log the RECONSTRUCTED state
        logger.info(
            f"Final state JUST BEFORE RETURN for thread {current_thread_id}: {return_state}"
        )
        logger.info(f"---> Returning final state with step_requiring_review: {return_state.get('step_requiring_review')}")
        logger.info(f"---> Returning final state with status: {return_state.get('status')}")
        logger.info(f"---> Returning final state with discount_status: {return_state.get('discount_status')}")

        return cast(AgentState, return_state) # Return the RECONSTRUCTED state

    def resume_after_review(
        self,
        thread_id: str,
        decision: Literal["accepted", "rejected"],
        feedback: Optional[str] = None,
    ) -> AgentState:
        """Resumes the workflow after a human review decision.

        Args:
            thread_id: The ID of the conversation thread to resume.
            decision: The human reviewer's decision ('accepted' or 'rejected').
            feedback: Optional feedback text if the decision was 'rejected'.

        Returns:
            The final AgentState after the workflow resumes and completes or pauses again.

        Raises:
            ValueError: If the thread_id is invalid or not awaiting review.
        """
        logger.info(f"Resuming workflow for thread {thread_id} after human review. Decision: {decision}")

        # Verify thread exists in conversation data
        if thread_id not in self.conversation_threads:
            raise ValueError(f"Invalid thread_id: {thread_id}")
            
        # Initialize thread memory if needed
        if thread_id not in self.threads:
            self.threads[thread_id] = {}
            
        # Get the last state from the conversation threads
        last_state = self.conversation_threads[thread_id].get("last_state")
        if not last_state:
            raise ValueError(f"No state found for thread {thread_id}")

        step_reviewed = last_state.get("step_requiring_review")
        if not step_reviewed:
            raise ValueError(f"Thread {thread_id} is not currently awaiting review.")

        logger.debug(f"Last state loaded for thread {thread_id} before update: {last_state}")

        # ------------- Direct Routing Logic -------------
        # Instead of relying on the graph's conditional edges, let's determine the
        # next step explicitly based on the step reviewed and the decision.
        next_step = None
        
        if decision == "accepted":
            logger.info(f"Human accepted output for '{step_reviewed}'. Determining next step.")
            # Determine next step based on the step that was reviewed
            if step_reviewed == "analyze_information":
                # If analyze_information was accepted:
                # - If info is complete: go to check_discounts
                # - If info is incomplete: go to generate_info_request
                if last_state.get("status") == "info_complete":
                    next_step = CHECK_DISCOUNTS_NODE
                else:  # Default to info_request for missing information
                    next_step = GENERATE_INFO_REQUEST_NODE
            elif step_reviewed == "check_for_discounts":
                # If check_for_discounts was accepted, go to generate_quote
                next_step = GENERATE_QUOTE_NODE
            elif step_reviewed == "generate_quote":
                # If generate_quote was accepted, we're done (END)
                logger.info("Quote accepted. Workflow complete.")
                # Just update state and return, no next step needed
                last_state["last_human_decision_for_step"] = last_state.get("last_human_decision_for_step", {})
                last_state["last_human_decision_for_step"][step_reviewed] = decision
                last_state["step_requiring_review"] = None
                last_state["data_for_review"] = None
                last_state["quote_ready"] = True # Explicitly set quote_ready to True
                self.conversation_threads[thread_id]["last_state"] = last_state.copy()
                return last_state
            elif step_reviewed == "generate_info_request":
                # If generate_info_request was accepted, we're done for now (END)
                logger.info("Info request accepted. Workflow paused until customer responds.")
                
                # Extract the generated message from data_for_review
                generated_message = last_state.get("data_for_review", {}).get("generated_info_request_message")
                if generated_message:
                    # Create a proper message in the state with the accepted content
                    if "messages" not in last_state:
                        last_state["messages"] = []
                    
                    # Add the info request message to the messages list
                    info_request_message = {
                        "role": "agent",
                        "type": "info_request",
                        "content": generated_message,
                        "timestamp": datetime.now().isoformat()
                    }
                    last_state["messages"].append(info_request_message)
                    
                    # Also add it to the email thread history
                    if thread_id in self.conversation_threads:
                        email_history = self.conversation_threads[thread_id].get("history", [])
                        email_history.append({
                            "role": "agent", 
                            "content": generated_message,
                            "type": "info_request",
                            "timestamp": datetime.now().isoformat()
                        })
                        self.conversation_threads[thread_id]["history"] = email_history
                    
                    logger.info("Added accepted info request message to state")
                else:
                    logger.warning("No generated message found in data_for_review")
                
                # Update state with decision and clear review flags
                last_state["last_human_decision_for_step"] = last_state.get("last_human_decision_for_step", {})
                last_state["last_human_decision_for_step"][step_reviewed] = decision
                last_state["step_requiring_review"] = None
                last_state["data_for_review"] = None
                last_state["status"] = "info_requested"  # Set appropriate status
                
                # Store updated state and return
                self.conversation_threads[thread_id]["last_state"] = last_state.copy()
                return last_state
        else:  # decision == "rejected"
            # Handle retries
            current_retries = last_state.get("retry_counts", {})
            current_count = current_retries.get(step_reviewed, 0)
            new_retry_count = current_count + 1
            
            # Update state for rejection
            if "retry_counts" not in last_state:
                last_state["retry_counts"] = {}
            last_state["retry_counts"][step_reviewed] = new_retry_count
            
            if feedback:
                if "rejection_feedback_for_step" not in last_state:
                    last_state["rejection_feedback_for_step"] = {}
                last_state["rejection_feedback_for_step"][step_reviewed] = feedback
                
            logger.info(f"Incremented retry count for '{step_reviewed}' to {new_retry_count}")
            
            # Check if max retries reached
            if new_retry_count >= MAX_RETRIES:
                logger.warning(f"Max retries reached for {step_reviewed}. Routing to error handler.")
                next_step = HANDLE_MAX_RETRIES  # Handle max retries reached
            else:
                # Retry the same step
                if step_reviewed == "analyze_information":
                    next_step = ANALYZE_INFORMATION_NODE
                elif step_reviewed == "check_for_discounts":
                    next_step = CHECK_DISCOUNTS_NODE
                elif step_reviewed == "generate_quote":
                    next_step = GENERATE_QUOTE_NODE
                elif step_reviewed == "generate_info_request":
                    next_step = GENERATE_INFO_REQUEST_NODE
        
        # Apply decision to state regardless of path
        last_state["last_human_decision_for_step"] = last_state.get("last_human_decision_for_step", {})
        last_state["last_human_decision_for_step"][step_reviewed] = decision
        last_state["step_requiring_review"] = None  # Clear review flags
        last_state["data_for_review"] = None

        # If we determined a next step, invoke that specific node with our updated state
        if next_step:
            logger.info(f"Explicitly routing to {next_step} after human decision")
            # First, store the updated state
            self.conversation_threads[thread_id]["last_state"] = last_state.copy()
            
            # Invoke that specific node with our state
            # This is more reliable than relying on LangGraph's graph routing
            if next_step == ANALYZE_INFORMATION_NODE:
                new_state = analyze_information(last_state, self.llm)
            elif next_step == CHECK_DISCOUNTS_NODE:
                new_state = check_for_discounts(last_state, self.llm)
            elif next_step == GENERATE_INFO_REQUEST_NODE:
                new_state = generate_info_request(last_state, self.llm)
            elif next_step == GENERATE_QUOTE_NODE:
                new_state = generate_quote(last_state, self.llm)
            elif next_step == HANDLE_MAX_RETRIES:
                new_state = self._handle_max_retries(last_state)
            else:
                raise ValueError(f"Unknown next step: {next_step}")
                
            # Most nodes then lead to a prepare_review node
            # Let's handle those transitions manually as well
            if next_step == ANALYZE_INFORMATION_NODE:
                new_state = prepare_info_review(new_state)
            elif next_step == CHECK_DISCOUNTS_NODE: 
                new_state = prepare_discount_review(new_state)
            elif next_step == GENERATE_INFO_REQUEST_NODE:
                new_state = prepare_info_request_review(new_state)
            elif next_step == GENERATE_QUOTE_NODE:
                new_state = prepare_quote_review(new_state)
                
            # Store the updated state and return it
            self.conversation_threads[thread_id]["last_state"] = new_state.copy()
            return new_state
        else:
            # If no next step was determined, just return the current state
            logger.warning(f"No explicit next step determined for {step_reviewed} with decision {decision}")
            return last_state

        # TODO: Update conversation history similar to process_email?
        # This depends on whether resume generates new messages.

        logger.info(
            f"Finished resuming workflow for thread {thread_id}. Final state: {final_state_after_resume}"
        )

        return cast(AgentState, final_state_after_resume)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    agent = PPAAgent(provider="gemini")

    # --- Test Case 1: New Thread ---
    print("\n--- Test Case 1: New Thread ---")
    test_email_1 = """
    Hello,
    I need car insurance for my 2021 Honda Civic.
    I'm Jane Doe, 28 years old.
    My address is 456 Oak Ave, Anytown, USA.
    """
    result_1 = agent.process_email(test_email_1)
    thread_id_1 = result_1.get("thread_id")
    print(f"Created Thread ID: {thread_id_1}")
    print(json.dumps(result_1, indent=2))

    # --- Test Case 2: Follow-up on the same thread ---
    print(f"\n--- Test Case 2: Follow-up (Thread: {thread_id_1}) --- Now processes the reply ---")
    test_email_2 = """
    Thanks for the quick response!
    My Honda Civic is the LX trim.
    My vehicle year is 2021.
    """
    if thread_id_1:
        # Process the second email using the thread_id from the first
        result_2 = agent.process_email(test_email_2, thread_id=thread_id_1)
        print(json.dumps(result_2, indent=2))
    else:
        print("Cannot run test case 2 without a thread ID from test case 1.")

    # --- Test Case 3: Different New Thread ---
    print("\n--- Test Case 3: Different New Thread ---")
    test_email_3 = """
    Hi, I need home insurance.
    """
    result_3 = agent.process_email(test_email_3)
    thread_id_3 = result_3.get("thread_id")
    print(f"Created Thread ID: {thread_id_3}")
    print(json.dumps(result_3, indent=2))
