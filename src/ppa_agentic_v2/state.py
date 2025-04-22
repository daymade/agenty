# src/ppa_agentic_v2/state.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Annotated
from langchain_core.messages import BaseMessage
import operator
import uuid
import logging

logger = logging.getLogger(__name__)

# Define a type hint for the structured feedback from agency review
HumanFeedback = Dict[str, Any] # e.g., {"approved": bool, "comment": str, "edited_inputs": dict}

class AgentState(BaseModel):
    """
    Represents the state of a single PPA quoting conversation thread.
    Managed by LangGraph and persisted using a checkpointer.
    """
    # ================= Core Conversation State ==================
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the conversation thread.")
    goal: str = Field(default="Get a PPA insurance quote for the customer.", description="The overall objective of the agent.")
    # Use Annotated List[BaseMessage] for automatic message appending by LangGraph
    messages: Annotated[List[BaseMessage], operator.add] = Field(default_factory=list, description="Full history of messages (Human, AI, Tool). Appended automatically.")

    # ==================== Extracted Data ======================
    customer_info: Optional[Dict[str, Any]] = Field(default=None, description="Accumulated structured data about customer, drivers, vehicles etc.")
    mercury_session: Optional[Dict[str, Any]] = Field(default=None, description="Stores context from Mercury APIs (e.g., Quote ID, session tokens). Updated by relevant tools.")

    # ============== Agent's Internal State & Planning ============
    agent_scratchpad: Optional[str] = Field(default=None, description="Internal reasoning or notes from the Planner for the current step. Cleared each cycle.")

    # ============ Action Execution & Control Flow ==============
    planned_tool_inputs: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Action details (`{'tool_name': ..., 'args': ...}`) set by Planner for *immediate* execution (if no review needed) OR transferred from `action_pending_review` upon approval. Cleared after execution attempt."
    )
    action_pending_review: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Action details (`{'tool_name': ..., 'args': ...}`) set by Planner when `requires_agency_review` is True. Read by `_handle_human_feedback`. Cleared after feedback processing or execution."
    )
    requires_agency_review: bool = Field(
        default=False, 
        description="Flag set by Planner indicating the action in `action_pending_review` needs human approval. Cleared after feedback processing."
    )
    human_feedback: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Raw feedback dictionary received via `/update` endpoint during an agency review pause. Processed and cleared by `_handle_human_feedback`."
    )
    is_waiting_for_customer: bool = Field(
        default=False, 
        description="Flag set by Planner when its explicit decision is to wait for customer input (e.g., after `ask_customer_tool`). Used for routing to wait pause."
    )

    # ================= Event History & Results ==================
    # Use Annotated List for automatic event appending by Nodes
    event_history: Annotated[List[Dict], operator.add] = Field(
        default_factory=list,
        description="Append-only list storing dicts representing significant events (tool calls, feedback processing). Used by Planner prompt context."
    )
    last_tool_outputs: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Result dictionary from the *most recent* node execution (tool call *or* feedback processing). Used for *immediate* conditional routing. Overwritten each step."
    )

    # =================== Internal Utilities =====================
    loop_count: int = Field(default=0, description="Loop counter, defaults to 0.")

    # Allow arbitrary types for LangChain messages etc.
    class Config:
        arbitrary_types_allowed = True

    def update_state(self, update_dict: Dict[str, Any]) -> "AgentState":
        """Helper method to update state immutably."""
        # Note: LangGraph often handles merging directly, but this can be useful
        updated_data = self.model_dump()
        updated_data.update(update_dict)
        # Re-validate if needed, handle potential errors
        new_state = AgentState(**updated_data)
        return new_state

    def clear_planning_state(self) -> "AgentState":
        """Clears fields related to the previous planning/execution cycle before the next Planner run."""
        updates = {
            "planned_tool_inputs": None,
            "action_pending_review": None,
            "requires_agency_review": False,
            "human_feedback": None, # Ensure feedback is cleared *after* processing
            "is_waiting_for_customer": False,
            "agent_scratchpad": None, # Clear thoughts from previous cycle
            # NOTE: Do NOT clear last_tool_outputs here, Planner uses it via event_history
            # NOTE: Do NOT clear event_history ever
        }
        return self.update_state(updates)