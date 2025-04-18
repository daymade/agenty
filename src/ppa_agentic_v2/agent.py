# src/ppa_agentic_v2/agent.py
import logging
import json
from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from pydantic import ValidationError # Use BaseException for broader catch if needed
from langgraph.prebuilt import ToolNode # Import ToolNode
from .state import AgentState
from .tools import all_tools # Import the list of tools
from .prompts import PLANNER_PROMPT_TEMPLATE_V1
from .llm import get_llm_client, MockPlannerLLM # Import mock LLM for now
import uuid # Import uuid for generating tool call IDs
from langgraph.checkpoint.sqlite import SqliteSaver # <-- Import Saver
from .config import SQLITE_DB_NAME # <-- Import DB name
import sqlite3 # <-- Add sqlite3 import
import os

logger = logging.getLogger(__name__)

# --- Create SQLite Checkpointer for standalone mode --- #
# Create SQLite connection
conn = sqlite3.connect(SQLITE_DB_NAME, check_same_thread=False)
# Create the SqliteSaver with the connection
sqlite_memory = SqliteSaver(conn)
logger.info(f"SQLite checkpointer initialized with database: {SQLITE_DB_NAME}")

# --- Agent Nodes ---

def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    The core planning node. Decides the next action.
    (Uses MOCK LLM in Milestone 2)
    """
    logger.info("--- Entering Planner Node ---")
    # logger.debug(f"Current State: {state}")

    # Prepare prompt inputs
    goal = state.goal
    messages = state.messages
    customer_info = state.customer_info or {}
    last_tool_outputs = state.last_tool_outputs or {}

    # Format messages for prompt
    messages_str = "\n".join([f"{type(m).__name__}: {m.content}" for m in messages])
    customer_info_str = json.dumps(customer_info, indent=2)
    last_tool_outputs_str = json.dumps(last_tool_outputs, indent=2)

    # Format tools for prompt
    tools_str = "\n".join([f"- {tool.name}: {tool.description}" for tool in all_tools])

    # --- MOCK LLM SETUP (Replace with real LLM call later) ---
    # Define a simple plan for testing
    mock_plan = [
        {"thought": "Initiate the quote.", "tool_name": "mock_quote_initiate", "args": {"detail": "from email"}},
        {"thought": "Ask for missing info.", "tool_name": "mock_ask_customer", "args": {"missing_fields": ["driver_age", "vehicle_make"]}},
        {"thought": "Run step 2.", "tool_name": "mock_api_step_2", "args": {}},
        {"thought": "Process complete.", "tool_name": None, "args": None} # Signal completion
    ]
    # Use a simple state counter to advance mock plan (replace with real LLM)
    if not hasattr(planner_node, "mock_step"):
        planner_node.mock_step = 0 # Initialize counter

    if planner_node.mock_step >= len(mock_plan):
         llm_decision = {"thought": "Plan exhausted, completing.", "tool_name": None, "args": None}
         planner_node.mock_step = 0 # Reset for next run if needed
    else:
         llm_decision = mock_plan[planner_node.mock_step]
         planner_node.mock_step += 1

    logger.info(f"[MOCK PLANNER] Decided action: {llm_decision}")
    # --- END MOCK LLM ---

    # # --- REAL LLM CALL (Enable in later milestones) ---
    # llm = get_llm_client() # Get configured LLM
    # prompt = PLANNER_PROMPT_TEMPLATE_V1.format(
    #     goal=goal,
    #     messages_str=messages_str,
    #     customer_info_str=customer_info_str,
    #     last_tool_outputs_str=last_tool_outputs_str,
    #     tools_str=tools_str
    # )
    # logger.debug(f"Planner Prompt:\n{prompt}")
    # try:
    #     ai_msg = llm.invoke(prompt)
    #     response_json = ai_msg.content
    #     logger.debug(f"Planner LLM Raw Response: {response_json}")
    #     llm_decision = json.loads(response_json) # Expecting JSON output directly
    #     logger.info(f"Planner LLM Decided Action: {llm_decision}")
    # except json.JSONDecodeError:
    #     logger.error(f"Planner LLM did not return valid JSON: {ai_msg.content}")
    #     # Decide how to handle error - maybe request human review? For now, end.
    #     llm_decision = {"tool_name": None, "args": None, "error": "LLM output parse error"}
    # except Exception as e:
    #     logger.error(f"Error invoking planner LLM: {e}", exc_info=True)
    #     llm_decision = {"tool_name": None, "args": None, "error": f"LLM invocation error: {e}"}
    # # --- END REAL LLM CALL ---


    # Prepare state update based on LLM decision
    update = {"last_tool_outputs": None} # Clear last output before next execution
    tool_name = llm_decision.get("tool_name")

    if tool_name:
        # Prepare tool call structure for AIMessage
        tool_call_id = str(uuid.uuid4())
        tool_calls = [{
            "name": tool_name,
            "args": llm_decision.get("args", {}),
            "id": tool_call_id
        }]
        ai_message = AIMessage(
            content=llm_decision.get("thought", ""), # Include thought if available
            tool_calls=tool_calls
        )
        update["messages"] = state.messages + [ai_message] # Add message to state
        logger.info(f"Planner adding AIMessage with tool_calls: {ai_message}")

        # Keep planned_tool_inputs for potential other uses, though ToolNode uses messages
        update["planned_tool_inputs"] = {
            "tool_name": tool_name,
            "args": llm_decision.get("args", {})
        }
        logger.info(f"Planner setting planned_tool_inputs: {update['planned_tool_inputs']}")
    else:
        # Planner decided to complete or errored
        update["planned_tool_inputs"] = None
        logger.info("Planner decided no tool to execute (complete/error).")
        # Optionally add a final AIMessage indicating completion/error
        final_thought = llm_decision.get("thought", "No further action planned.")
        if "error" in llm_decision:
            final_thought = f"Error in planning: {llm_decision['error']}"
        final_ai_message = AIMessage(content=final_thought)
        update["messages"] = state.messages + [final_ai_message]
        logger.info(f"Planner adding final AIMessage: {final_ai_message}")

    # Add thought process to scratchpad if available
    if "thought" in llm_decision:
         update["agent_scratchpad"] = llm_decision["thought"]

    return update

def execute_tool_node(state: AgentState) -> Dict[str, List[BaseMessage]]:
    """
    Executes tools based on the latest AIMessage tool calls and returns
    the result dictionary {'messages': [ToolMessage(...)]} for state update.
    """
    logger.info("--- Running Tool Executor Node ---")
    # Access messages using attribute access, default to empty list if None
    messages = state.messages if hasattr(state, 'messages') and state.messages is not None else []
    last_message = messages[-1] if messages else None

    # Check if the last message is an AIMessage with tool_calls
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logger.warning("Executor node called without pending tool calls in the last message.")
        # Return an empty update if no tools to run
        return {"messages": []}

    tool_node = ToolNode(all_tools)
    try:
        # Invoke ToolNode - It returns a dict {'messages': [ToolMessage(...)]}
        # This dictionary is the state update itself.
        update_dict = tool_node.invoke(state)

        # Optional: Add validation for the dict structure if desired
        if not isinstance(update_dict, dict) or 'messages' not in update_dict or \
           not isinstance(update_dict['messages'], list) or \
           not all(isinstance(msg, ToolMessage) for msg in update_dict['messages']):
            error_content = f"ToolNode returned unexpected dict structure: {update_dict}"
            logger.error(error_content)
            # Create error messages if structure is wrong
            tool_call_ids = [tc['id'] for tc in last_message.tool_calls]
            error_messages = [ToolMessage(content=error_content, tool_call_id=tc_id) for tc_id in tool_call_ids]
            return {"messages": error_messages}

        logger.info(f"Tool execution successful. State update generated: {update_dict}")
        # Return the update dictionary directly, as it's the expected format
        return update_dict

    except Exception as e:
        logger.exception(f"Error executing tool node: {e}")
        # Create error messages in the expected state update format
        tool_call_ids = [tc['id'] for tc in last_message.tool_calls]
        error_messages = [ToolMessage(content=f"Error executing tool: {e}", tool_call_id=tc_id) for tc_id in tool_call_ids]
        return {"messages": error_messages}


# --- Conditional Edges ---

def should_execute(state: AgentState) -> str:
    """Routes to execution if a tool is planned, otherwise ends."""
    if state.planned_tool_inputs:
        logger.info(f"Routing to executor for tool: {state.planned_tool_inputs.get('tool_name')}")
        return "execute"
    else:
        logger.info("Routing to END (no tool planned).")
        return END

# --- Graph Definition ---

def build_agent_graph(use_persistence=False):
    """Builds the LangGraph StateGraph with optional persistence.
    
    Args:
        use_persistence (bool): If True, use SQLite persistence. 
                               If False, no persistence (for LangGraph dev server).
    """
    workflow = StateGraph(AgentState)

    # Add nodes (same as before)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", execute_tool_node)

    # Setup explicit START and END routing for best practices
    workflow.add_edge(START, "planner")  # Explicit edge from START to planner

    # Define edges (same as before)
    workflow.add_conditional_edges(
        "planner",
        should_execute,
        {
            "execute": "executor",
            END: END
        }
    )
    workflow.add_edge("executor", "planner")

    # Compile the graph with conditionally using the checkpointer
    if use_persistence:
        # For standalone runs, use our SQLite checkpointer
        app = workflow.compile(checkpointer=sqlite_memory)
        logger.info("Agent graph compiled (Milestone 3 - WITH SQLite Persistence).")
    else:
        # For LangGraph API mode, don't provide a checkpointer
        app = workflow.compile()
        logger.info("Agent graph compiled (Milestone 3 - Using LangGraph built-in persistence).")
    
    return app

# --- Main Agent Class (Simple Runner for Now) ---
class PPAAgentRunner:
    def __init__(self, use_persistence=True):
        """Initialize the PPA Agent runner.
        
        Args:
            use_persistence (bool): Whether to use SQLite persistence.
        """
        # Remove mock planner step data if present (for clean testing)
        if hasattr(planner_node, "mock_step"):
             del planner_node.mock_step

        # Create the graph with appropriate persistence setting
        self.graph = build_agent_graph(use_persistence=use_persistence)

    def run_turn(self, thread_id: str, user_input: Optional[str] = None):
        """Runs a single turn for a given thread_id."""
        logger.info(f"--- Running Agent Turn (Thread: {thread_id}) ---")
        if user_input:
             logger.info(f"Input: '{user_input}'")

        # Prepare input message list if user input is provided
        input_messages = []
        if user_input:
            input_messages.append(HumanMessage(content=user_input))

        # Configuration includes the thread_id
        config = {"configurable": {"thread_id": thread_id}}

        final_state_dict = {}
        try:
            # The checkpointer handles loading state based on thread_id
            # Providing input adds the new message(s) to the loaded state
            for step in self.graph.stream({"messages": input_messages}, config=config):
                step_name = list(step.keys())[0]
                step_data = list(step.values())[0]
                logger.info(f"Step: {step_name}")
                # logger.debug(f"Step Data: {step_data}")
                final_state_dict = step_data

            logger.info(f"--- Agent Turn Complete (Thread: {thread_id}) ---")
            # Get the final state from the checkpointer for inspection
            final_state_persisted = self.graph.get_state(config)
            logger.info(f"Final Persisted Messages: {final_state_persisted.values['messages']}")
            logger.info(f"Last Tool Outputs: {final_state_persisted.values['last_tool_outputs']}")
            return final_state_persisted.values # Return the dict representation

        except Exception as e:
             logger.error(f"Error during agent run (Thread: {thread_id}): {e}", exc_info=True)
             try:
                 # Try to get state even if run failed mid-way
                 current_state = self.graph.get_state(config)
                 return current_state.values
             except Exception:
                  logger.error(f"Could not retrieve state for thread {thread_id} after error.")
                  return {"error": str(e)} # Return error dict

# --- Compile graphs at module level for different use cases ---
# For standalone runs with persistence (e.g. run_agent.py)
graph_with_persistence = build_agent_graph(use_persistence=True)

# For LangGraph dev server (needs to be named 'graph' for discovery)
graph = build_agent_graph(use_persistence=False)