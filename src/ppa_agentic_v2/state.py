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
    # == Conversation Context ==
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the conversation thread.")
    goal: str = Field(default="Get a PPA insurance quote for the customer.", description="The overall objective of the agent.")
    # Use Annotated List[BaseMessage] for automatic message appending by LangGraph
    messages: Annotated[List[BaseMessage], operator.add] = Field(default_factory=list, description="Full history of messages (Human, AI, Tool).")

    # == Extracted & API Data ==
    customer_info: Optional[Dict[str, Any]] = Field(default=None, description="Accumulated structured data about customer, drivers, vehicles etc.")
    mercury_session: Optional[Dict[str, Any]] = Field(default=None, description="Stores context from Mercury APIs (e.g., Quote ID, session tokens).")

    # == Agent Planning & Execution State ==
    planned_tool_inputs: Optional[Dict[str, Any]] = Field(default=None, description="The tool name and arguments planned by the Planner for the *next* action. Reviewed during Agency Review.")
    last_tool_outputs: Optional[Dict[str, Any]] = Field(default=None, description="The dictionary result {'status': 'success'|'error', ...} from the *last executed* tool.")
    agent_scratchpad: Optional[str] = Field(default=None, description="Internal reasoning or notes from the Planner.")

    # == HITL Control Flags & Feedback ==
    requires_agency_review: bool = Field(default=False, description="Set by Planner if the planned action in 'planned_tool_inputs' needs human review before execution.")
    awaiting_customer_reply: bool = Field(default=False, description="Set by Planner *before* triggering the 'Wait for Customer' interrupt.")
    human_feedback: Optional[Dict[str, Any]] = Field(default=None, description="Feedback received from the Agency Review interrupt. Processed and cleared by the Planner.")

    # == Loop Counter ==
    loop_count: int = Field(default=0, description="Loop counter, defaults to 0.")

    # Allow arbitrary types for LangChain messages etc.
    class Config:
        arbitrary_types_allowed = True

    def update_state(self, update_dict: Dict[str, Any]) -> "AgentState":
        """Helper method to update state immutably."""
        # Note: LangGraph often handles merging directly, but this can be useful
        updated_data = self.model_dump()
        updated_data.update(update_dict)
        # logger.debug(f"Updating state with: {update_dict}")
        # logger.debug(f"Resulting state data: {updated_data}")

        # Re-validate if needed, handle potential errors
        new_state = AgentState(**updated_data)
        # logger.debug(f"New state object created: {new_state}")
        return new_state

    def clear_planning_state(self) -> "AgentState":
        """Clears fields related to the previous planning/execution cycle."""
        updates = {
            "planned_tool_inputs": None,
            "last_tool_outputs": None, # Or keep last output? Depends on planner needs
            "requires_agency_review": False,
            "awaiting_customer_reply": False,
            "human_feedback": None, # Ensure feedback is cleared after processing
            "agent_scratchpad": None # Optional: clear scratchpad each cycle?
        }
        return self.update_state(updates)