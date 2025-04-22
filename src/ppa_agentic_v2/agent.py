# src/ppa_agentic_v2/agent.py
import logging
import json
from typing import List, Dict, Any, Optional, Tuple, AsyncIterator
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError, BaseModel, Field 
from langgraph.prebuilt import ToolNode 
import uuid 
import re 
from langchain_core.exceptions import OutputParserException 
from pydantic import ValidationError 
from .llm_setup import llm, llm_with_tools # Import initialized LLMs
from .graph import build_agent_graph, PLANNER_NODE_NAME, EXECUTOR_NO_REVIEW_NODE_NAME, AGENCY_REVIEW_NODE_NAME, CUSTOMER_WAIT_NODE_NAME, START, END # Import the correct build_agent_graph
from .state import AgentState
from .tools import all_tools, TOOL_MAP 
from .config import ( 
    GOOGLE_API_KEY, OPENAI_API_KEY, 
    DEFAULT_LLM_PROVIDER, GOOGLE_MODEL_NAME, OPENAI_MODEL_NAME, logger
)
from .prompts import format_planner_prompt 
from .llm import get_llm_client 

# --- Constants ---
PLANNER_NODE_NAME = "planner"
EXECUTOR_NO_REVIEW_NODE_NAME = "execute_tool_no_review"
AGENCY_REVIEW_NODE_NAME = "agency_review_pause"

# --- Planner Node Helper Functions ---

def _handle_human_feedback(state: AgentState, update: Dict[str, Any]) -> Tuple[bool, Optional[List[BaseMessage]], Optional[Dict[str, Any]]]:
    """Processes human feedback if present, updates the state update dict, and determines if LLM call is needed."""
    logger.debug("Checking for human feedback.")
    call_llm = True
    prompt_input_messages = None
    current_tool_outputs = None

    if state.human_feedback:
        logger.info("Planner processing human feedback from previous step.")
        feedback = state.human_feedback
        approved = feedback.get("approved", False)
        comment = feedback.get("comment", "")
        planned_action = state.planned_tool_inputs # Action planned *before* review
        tool_name_reviewed = planned_action.get("tool_name", "unknown") if planned_action else "unknown"

        if approved:
            logger.info(f"Action '{tool_name_reviewed}' approved by human. Routing to executor.")
            # Keep the previously planned action but mark review as done
            update["planned_tool_inputs"] = planned_action
            update["requires_agency_review"] = False # Approved, no need for review *again*
            update["last_tool_outputs"] = {"status": "approved_by_human", "tool_name": tool_name_reviewed, "message": comment}
            update["human_feedback"] = None # Consume feedback
            call_llm = False # Skip LLM call, proceed directly to execution

        else: # Rejected
            logger.info(f"Action '{tool_name_reviewed}' rejected by human. Replanning needed.")
            current_tool_outputs = { # This becomes input for replanning prompt
                "status": "rejected_by_human",
                "message": comment or "Action rejected by human review.",
                "tool_name": tool_name_reviewed
            }
            update["last_tool_outputs"] = current_tool_outputs # Record rejection as last output
            update["planned_tool_inputs"] = None # Clear the rejected plan
            update["requires_agency_review"] = False # No action planned yet
            update["human_feedback"] = None # Consume feedback
            # Let call_llm remain True to proceed to LLM replanning below
            prompt_input_messages = format_planner_prompt(state, all_tools, current_tool_outputs=current_tool_outputs, human_feedback_str=None)
    else:
        logger.debug("No human feedback found.")

    return call_llm, prompt_input_messages, current_tool_outputs


# --- Helper Function to get Tools ---
def get_available_tools() -> List[BaseTool]:
    """Returns the list of available tools for the agent."""
    return all_tools

class PPAAgentRunner:
    """Manages the execution of the PPA Agentic V2 graph."""

    def __init__(self):
        logger.info("Initializing PPAAgentRunner...")

        # Get the graph builder
        self.app = build_agent_graph() # Call without checkpointer
        logger.info("Agent graph compiled.") # Update log message

        # Generate and store mermaid syntax *after* compilation
        try:
            # Use the compiled app to get the graph for drawing
            self.graph_mermaid_syntax = self.app.get_graph().draw_mermaid()
            logger.info("Generated Mermaid syntax for graph.")
        except Exception as e:
            logger.error(f"Failed to generate Mermaid syntax: {e}")
            self.graph_mermaid_syntax = "Error: Could not generate graph diagram."

    async def _invoke_llm_and_parse(self, state: AgentState, update: Dict[str, Any]):
        """(Instance Method) Invokes the LLM, parses the response, and updates the state update dict."""
        if not llm_with_tools:
            logger.error("LLM with tools is not initialized.")
            update["agent_scratchpad"] = "Error: LLM not configured."
            return

        # Format the prompt using the function from prompts.py
        # Pass necessary components from state and the generated mermaid syntax
        prompt_str = format_planner_prompt(
            state=state,
            tools=get_available_tools(),
            current_tool_outputs=state.last_tool_outputs,
            human_feedback_str=state.agent_scratchpad,
            graph_mermaid_syntax=self.graph_mermaid_syntax # Use instance attribute
        )
        logger.debug(f"Formatted Planner Prompt:\n{prompt_str}")

        # Prepare messages for LLM
        messages = [HumanMessage(content=prompt_str)]

        # --- Invoke LLM --- #
        try:
            logger.info("Invoking planner LLM...")
            response = await llm_with_tools.ainvoke(messages)
            logger.info("Planner LLM invocation successful.")

            # --- Manually check for tool calls --- #
            if response.tool_calls:
                # Assumption: Planner LLM only calls one tool at a time
                tool_call = response.tool_calls[0]
                tool_name = tool_call['name']
                try:
                    # Args are now already a dict when using native tool calls
                    tool_args = tool_call['args'] # Assign the dict directly
                    logger.info(f"Planner decided to call tool: {tool_name} with args: {tool_args}")
                    # Return the state update dictionary for tool execution
                    update["planned_tool_inputs"] = {
                        "tool_name": tool_name,
                        "tool_args": tool_args
                    }
                    update["requires_agency_review"] = tool_call.get("requires_review", False)
                    update["agent_scratchpad"] = tool_call.get("log", "Planning to use tool: " + tool_call.get("tool_name", "Unknown"))
                    update["is_waiting_for_customer"] = False
                except Exception as e:
                    logger.error(f"Failed to parse tool args: {e}")
                    update["agent_scratchpad"] = f"Error: Planner failed to parse tool args for {tool_name}. Args: {tool_call['args']}"
                    update["is_waiting_for_customer"] = False
            else:
                # No tool call, LLM provided a response or decided to wait
                logger.info("Planner decided on final answer/no tool.")
                llm_content = response.content
                update["agent_scratchpad"] = f"Planner response: {llm_content}"
                # Check if the LLM explicitly stated it is waiting
                if "wait" in llm_content.lower() or "waiting for customer" in llm_content.lower():
                    logger.info("LLM response indicates waiting for customer.")
                    update["is_waiting_for_customer"] = True
                else:
                    update["is_waiting_for_customer"] = False

        except Exception as e:
            logger.error(f"Unexpected error during planner LLM invocation/parsing: {e}", exc_info=True)
            update["agent_scratchpad"] = f"Error: An unexpected error occurred in the planner LLM step. Error: {e}"
            update["is_waiting_for_customer"] = False

    # --- Planner Node (Instance Method) --- #
    async def planner_node(self, state: AgentState) -> Dict[str, Any]:
        """(Instance Method) Determines the next action or tool call based on the current state via LLM call."""
        logger.info("--- Planner Node --- ")
        # Log input state details
        logger.debug(f"Planner Node - Input last_tool_outputs: {state.last_tool_outputs}")
        logger.debug(f"Planner Node - Input human_feedback: {state.human_feedback}")
        logger.debug(f"Planner Node - Input agent_scratchpad: {state.agent_scratchpad}")

        update: Dict[str, Any] = {}

        # --- Prepare state before LLM call based on feedback --- #
        if state.human_feedback:
            logger.info(f"Planner received human feedback: {state.human_feedback}")
            update["human_feedback"] = None # Always clear feedback after considering
            if not state.human_feedback.get("approved"):
                logger.info("Human denied previous plan. Clearing plan before replanning.")
                update["planned_tool_inputs"] = None
                update["last_tool_outputs"] = None
            else:
                logger.info("Human approved previous plan. LLM will confirm execution.")
                update["last_tool_outputs"] = None
        else:
            update["last_tool_outputs"] = None

        # --- Always Invoke LLM and Parse Output --- #
        logger.info("Planner proceeding to call LLM.")
        # Call the instance method _invoke_llm_and_parse
        await self._invoke_llm_and_parse(state, update)

        # Filter out None values before logging and returning
        final_update = {k: v for k, v in update.items() if v is not None}
        logger.info(f"Planner node final decision: {final_update}")
        return final_update

    # --- Executor Node (No Review Required) --- #
    # (Keep execute_tool_no_review as a static/module-level function or make it a method if needed)
    # ... (rest of PPAAgentRunner methods remain largely the same, using self.app) ...

    # --- Public Invocation Methods --- #
    async def stream(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Streams the agent's execution steps."""
        logger.info(f"Streaming agent with inputs: {inputs}, config: {config}")
        async for output in self.app.stream(inputs, config=config):
            yield output

    async def ainvoke(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Invokes the agent asynchronously for a single turn."""
        logger.info(f"Invoking agent with inputs: {inputs}, config: {config}")
        # Ensure interrupt configuration is included if needed, e.g., for review
        final_config = config.copy()
        final_config.setdefault("interrupt_before", ["agency_review_pause", "customer_wait_pause"])
        
        result = await self.app.ainvoke(inputs, config=final_config)
        logger.info(f"Agent invocation completed. Result keys: {list(result.keys())}")
        # logger.debug(f"Agent invocation result: {result}") # Might be too verbose
        return result

    def get_state(self, config: Dict[str, Any]) -> AgentState:
        """Retrieves the current state of the agent graph for a given thread."""
        logger.info(f"Getting agent state for config: {config}")
        state = self.app.get_state(config=config)
        # logger.debug(f"Retrieved state: {state}")
        return state # type: ignore

    def get_state_history(self, config: Dict[str, Any]) -> List[AgentState]:
        """Retrieves the history of states for a given thread."""
        logger.info(f"Getting agent state history for config: {config}")
        history = []
        for state in self.app.get_state_history(config=config):
            history.append(state) # type: ignore
        logger.info(f"Retrieved state history count: {len(history)}")
        return history

    async def update_state(self, config: Dict[str, Any], values: Dict[str, Any]) -> None:
        """Updates the agent state for a given thread."""
        # Note: 'values' should contain the fields to update, e.g., {'human_feedback': {...}}
        logger.info(f"Updating agent state for config: {config} with values: {values}")
        await self.app.update_state(config=config, values=values)
        logger.info("Agent state updated.")

    # --- Resume from Review Method --- # 
    async def resume_from_review(self, thread_id: str, feedback: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Resumes the agent execution after human review with provided feedback."""
        logger.info(f"Resuming thread {thread_id} from review with feedback: {feedback}")
        config = {"configurable": {"thread_id": thread_id}}

        # 1. Get the current state to format feedback message appropriately
        current_state = self.get_state(config)
        # Format feedback for scratchpad (ensure this logic matches planner expectations)
        feedback_status = "approved" if feedback.get("approved", False) else "rejected"
        feedback_comment = feedback.get("comment", "No comment provided.")
        human_feedback_str = f"Human Review Feedback -> Status: {feedback_status}. Comment: {feedback_comment}"

        # Prepare update dictionary
        update_values = {
            "human_feedback": feedback,
            "agent_scratchpad": human_feedback_str # Put formatted feedback here for LLM
        }

        # 2. Update the state with the feedback (and potentially clear scratchpad/tool plan if rejected?)
        # The planner node now handles clearing state based on feedback approval status
        await self.update_state(config, update_values)
        logger.info(f"State updated for thread {thread_id} with review feedback.")

        # 3. Resume streaming from the point after the interruption
        logger.info(f"Resuming stream for thread {thread_id}")
        async for output in self.app.stream(None, config=config):
             # The stream picks up from where it left off (after the interrupt)
             logger.debug(f"Stream output after resume: Node='{next(iter(output))}', Keys='{list(output.values())[0].keys() if output else 'N/A'}'")
             yield output
        logger.info(f"Stream finished for thread {thread_id} after resume.")

# --- Global Instance (Optional, depends on server setup) ---
agent_runner = PPAAgentRunner() # Instantiate the runner
graph = agent_runner.app # Expose the compiled graph for langgraph dev

# --- Helper Functions (Keep execute_tool_no_review, etc. here or move into class) ---
@tool
def ask_customer_tool(missing_fields: List[str]) -> str:
    """Asks the customer for the missing fields."""
    # ... (rest of the function remains the same)

async def execute_tool_no_review(state: AgentState) -> Dict[str, Any]:
    """Executes the planned tool without prior review."""
    # ... (rest of the function remains the same)

async def agency_review_pause(state: AgentState) -> Dict[str, Any]:
    """Node that pauses execution for agency review."""
    # ... (rest of the function remains the same)

async def customer_wait_pause(state: AgentState) -> Dict[str, Any]:
    """Node that pauses execution to wait for customer input."""
    # ... (rest of the function remains the same)