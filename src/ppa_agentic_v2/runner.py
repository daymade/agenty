# src/ppa_agentic_v2/runner.py

import logging
import uuid
import sqlite3
from typing import Dict, Any

from langgraph.graph import StateGraph

from .state import AgentState
from .config import logger, SQLITE_DB_NAME

class PPAAgentRunner:
    """A helper class to manage running the agent graph for a thread."""
    def __init__(self, graph: StateGraph):
        self.graph = graph
        # Store the checkpointer associated with the graph
        self.checkpointer = graph.checkpointer
        if not self.checkpointer:
             logger.warning("Graph provided to PPAAgentRunner does not have a checkpointer. State persistence may not work as expected.")

    def _get_thread_id(self) -> str:
        """Generates a unique thread ID."""
        return str(uuid.uuid4())

    def start_new_thread(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """Starts a new conversation thread."""
        thread_id = self._get_thread_id()
        logger.info(f"Starting new thread with ID: {thread_id}")
        config = {"configurable": {"thread_id": thread_id}}

        # Prepare the initial state based on AgentState defaults and initial_input
        initial_state = AgentState(**initial_input).dict()

        # Directly use graph.invoke for the first message to set the initial state correctly
        # Note: We don't use update_state here as invoke handles initial state setting
        final_state = self.graph.invoke(initial_state, config=config)
        logger.info(f"Initial processing complete for thread {thread_id}.")
        return final_state # Return the full state after the first run

    def process_message(self, thread_id: str, message_content: str) -> Dict[str, Any]:
        """Processes a single user message in an existing thread."""
        logger.info(f"Processing message for thread {thread_id}: '{message_content[:50]}...' ")
        config = {"configurable": {"thread_id": thread_id}}

        # Construct the input for the next step - typically just the new human message
        # The graph should merge this with the existing state
        # TODO: Revisit how messages are added. Need to conform to AgentState structure.
        # Assuming the graph's input schema expects something like {'messages': [HumanMessage(...)]}
        # For now, let's pass it as a dictionary that the planner node expects to find in state.messages
        # This might need refinement depending on the exact graph input requirements.
        step_input = {"messages": [("human", message_content)]}

        final_state = self.graph.invoke(step_input, config=config)
        logger.info(f"Message processing complete for thread {thread_id}.")
        return final_state

    def resume_from_review(self, thread_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Resumes a thread paused for agency review with feedback."""
        logger.info(f"Resuming thread {thread_id} from review with feedback: {feedback}")
        config = {"configurable": {"thread_id": thread_id}}

        # Update the state with the human feedback before resuming
        if self.checkpointer:
            self.checkpointer.update_state(config, {"human_feedback": feedback})
            logger.info(f"Updated state for thread {thread_id} with human feedback.")
        else:
            logger.error(f"Cannot update state with feedback for thread {thread_id}: No checkpointer available.")
            # Proceeding anyway, but feedback might be lost if not handled by graph logic itself

        # Invoke the graph with None input to continue from the interrupted state
        final_state = self.graph.invoke(None, config=config)
        logger.info(f"Resumed processing complete for thread {thread_id}.")
        return final_state

    def get_thread_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the current state of a thread."""
        logger.info(f"Getting state for thread {thread_id}")
        config = {"configurable": {"thread_id": thread_id}}
        if self.checkpointer:
            state_data = self.checkpointer.get_state(config)
            if state_data:
                # The state from SqliteSaver might be wrapped, extract the core values
                return state_data.values
            else:
                logger.warning(f"No state found for thread {thread_id}.")
                return None
        else:
            logger.error(f"Cannot get state for thread {thread_id}: No checkpointer available.")
            return None
