# src/ppa_agentic_v2/agent.py
import logging
import json
from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from pydantic import ValidationError, BaseModel, Field 
from langgraph.prebuilt import ToolNode 
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI 
from langchain_core.utils.function_calling import convert_to_openai_tool 
from .state import AgentState
from .tools import all_tools, TOOL_MAP 
from .config import ( 
    SQLITE_DB_NAME, GOOGLE_API_KEY, OPENAI_API_KEY, 
    DEFAULT_LLM_PROVIDER, GOOGLE_MODEL_NAME, OPENAI_MODEL_NAME, logger
)
from .prompts import format_planner_prompt 
from .llm import get_llm_client 
from langchain_core.messages import AIMessage 
import uuid 
import re 
from langchain_core.exceptions import OutputParserException 
from pydantic import ValidationError 

# --- Instantiate LLM --- 
# Select LLM based on config and available keys
llm = None
llm_with_tools = None
try:
    if DEFAULT_LLM_PROVIDER == "google" and GOOGLE_API_KEY:
        logger.info(f"Initializing Google LLM: {GOOGLE_MODEL_NAME}")
        llm = ChatGoogleGenerativeAI(model=GOOGLE_MODEL_NAME, temperature=0, google_api_key=GOOGLE_API_KEY)
    elif DEFAULT_LLM_PROVIDER == "openai" and OPENAI_API_KEY:
        logger.info(f"Initializing OpenAI LLM: {OPENAI_MODEL_NAME}")
        llm = ChatOpenAI(model=OPENAI_MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)
    else:
        # Fallback or error if no provider is configured/available
        logger.warning("No suitable LLM provider configured or API key found. Planner will not function.")

    if llm:
        # Convert tools to OpenAI format for the LLM
        llm_with_tools = llm.bind_tools([convert_to_openai_tool(t) for t in all_tools])
        logger.info("LLM tools bound successfully.")

except Exception as e:
    logger.error(f"Failed to initialize LLM or bind tools: {e}", exc_info=True)
    llm = None
    llm_with_tools = None

# --- Planner Prompt Template --- 
# Define the system prompt instructing the LLM
async def planner_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Entering Planner Node ---")
    update: Dict[str, Any] = {
        "agent_scratchpad": "", # Default empty unless LLM provides one
        "requires_agency_review": False, # Default unless LLM or logic sets it
        "planned_tool_inputs": None, # Default unless LLM sets it or approval keeps it
        "human_feedback": None, # Always clear feedback after processing
        "last_tool_outputs": None # Default unless feedback/LLM sets it
    }
    prompt_input_messages = [] 
    current_tool_outputs = None
    call_llm = True # Flag to control whether to call LLM at the end

    # --- Handle Post-Review State --- #
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
            call_llm = False # Skip LLM call, proceed directly to execution via conditional edge
            
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

    # --- Standard Planner Logic or Replanning after Rejection --- # 
    if call_llm:
        if not prompt_input_messages: # If not already set by rejection handling
             logger.info("Planner running standard logic (no feedback to process or approved action).")
             # Use state.last_tool_outputs from the *previous* node (e.g., executor)
             current_tool_outputs = state.last_tool_outputs
             prompt_input_messages = format_planner_prompt(state, all_tools, current_tool_outputs=current_tool_outputs, human_feedback_str=None)
        
        # --- Invoke LLM --- #
        try:
            logger.debug(f"Planner Prompt Messages: {prompt_input_messages}")
            response = await llm_with_tools.ainvoke(prompt_input_messages)
            logger.debug(f"LLM Response: {response.content}")

            # --- Parse LLM Output --- # 
            parsed_output = None
            if isinstance(response.content, str):
                raw_llm_output = response.content
                json_block_match = re.search(r"```json\n(.*?)\n```", raw_llm_output, re.DOTALL)
                if json_block_match:
                    json_str = json_block_match.group(1).strip()
                else:
                    json_str = raw_llm_output.strip()
                
                try:
                    parsed_output = json.loads(json_str)
                    logger.info(f"Successfully parsed LLM JSON: {parsed_output}")
                except json.JSONDecodeError as json_e:
                    logger.error(f"Failed to parse JSON from LLM output: {json_e}")
                    logger.debug(f"LLM Raw output that failed parsing: {raw_llm_output}")
                    raise OutputParserException(f"LLM output could not be parsed as JSON. Raw: {raw_llm_output}") from json_e
            else:
                 logger.error(f"Unexpected LLM response content type: {type(response.content)}")
                 raise OutputParserException(f"Unexpected LLM response content type: {type(response.content)}")

            if not isinstance(parsed_output, dict):
                logger.error(f"Parsed output is not a dictionary: {parsed_output}")
                raise OutputParserException(f"Parsed output is not a dictionary: {parsed_output}")

            # --- Update State Based on LLM Output --- #
            update["agent_scratchpad"] = parsed_output.get("thought", "")
            # Update requires_review based on *this new plan* from LLM
            update["requires_agency_review"] = parsed_output.get("requires_review", False) 

            tool_name = parsed_output.get("tool_name")
            if tool_name and tool_name != "final_answer":
                update["planned_tool_inputs"] = {
                    "tool_name": tool_name,
                    "args": parsed_output.get("args", {})
                }
                logger.info(f"Planner planned tool: {tool_name} with review required: {update['requires_agency_review']}")
            else:
                logger.info("Planner decided final_answer or no tool action needed.")
                update["planned_tool_inputs"] = None # Ensure it's cleared
                update["requires_agency_review"] = False # No review needed if no tool planned

        except (OutputParserException, ValidationError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing planner output: {e}")
            update["agent_scratchpad"] = f"Error: Could not parse LLM output. Error: {e}. Raw output: {response.content if 'response' in locals() else 'N/A'}"
            update["requires_agency_review"] = True # Force review if parsing fails
            update["planned_tool_inputs"] = None 
        except Exception as e:
            logger.error(f"Unexpected error in planner node: {e}", exc_info=True)
            update["agent_scratchpad"] = f"Error: An unexpected error occurred in the planner. Error: {e}"
            update["requires_agency_review"] = True # Force review on unexpected errors
            update["planned_tool_inputs"] = None

    # Filter out None values before logging and returning
    final_update = {k: v for k, v in update.items() if v is not None}
    logger.info(f"--- Exiting Planner Node. State updates: {final_update} ---")
    return final_update


async def executor_node_wrapper(state: AgentState) -> Dict[str, Any]:
    """Executes the tool specified in state.planned_tool_inputs.
    This node is now ONLY called when the planner decides to execute a tool
    AND does NOT require agency review.
    """
    logger.info("--- Entering Executor Node Wrapper (Review NOT Required) ---")
    planned_tool_inputs = state.planned_tool_inputs

    if not planned_tool_inputs or not isinstance(planned_tool_inputs, dict):
        logger.warning("Executor called without valid planned_tool_inputs.")
        return {"last_tool_outputs": {"status": "error", "message": "No tool planned for execution."}}

    tool_name = planned_tool_inputs.get("tool_name")
    tool_args = planned_tool_inputs.get("args", {}) # Args might be empty

    if not tool_name:
        logger.error("Planned tool inputs missing 'tool_name'.")
        return {"last_tool_outputs": {"status": "error", "message": "Tool name missing in plan."}}

    logger.info(f"Executing tool (no review required): {tool_name} with args: {tool_args}")

    if tool_name in TOOL_MAP:
        selected_tool = TOOL_MAP[tool_name]
        try:
            # Ensure args are passed correctly, tool.ainvoke expects a dict
            # If tool_args is None or not a dict, provide an empty dict
            input_args = tool_args if isinstance(tool_args, dict) else {}
            tool_result = await selected_tool.ainvoke(input_args)
            logger.info(f"Tool '{tool_name}' executed successfully (no review). Result keys: {list(tool_result.keys()) if isinstance(tool_result, dict) else 'N/A'}")
            update = {
                "last_tool_outputs": tool_result,
                "planned_tool_inputs": None, # Clear plan after execution
                "agent_scratchpad": "", # Clear scratchpad
                "human_feedback": None # Clear any lingering feedback
            }
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
            update = {
                "last_tool_outputs": {"status": "error", "tool_name": tool_name, "message": f"Execution failed: {e}"},
                "planned_tool_inputs": None,
                "agent_scratchpad": "",
                "human_feedback": None
            }
    else:
        logger.error(f"Tool '{tool_name}' not found in TOOL_MAP.")
        update = {
            "last_tool_outputs": {"status": "error", "tool_name": tool_name, "message": "Tool definition not found."},
            "planned_tool_inputs": None,
            "agent_scratchpad": "",
            "human_feedback": None
        }

    logger.info(f"--- Exiting Executor Node Wrapper. State updates: {update} ---")
    return update

def agency_review_pause_node(state: AgentState) -> Dict[str, Any]:
    """Placeholder node for the agency review interrupt point.
    The graph pauses *before* AND *after* this node when run in dev mode.
    To simulate review:
    1. Click 'Continue' on the first pause (before).
    2. Manually edit the 'human_feedback' field in the state JSON in the UI
       (e.g., {"approved": false, "comment": "Needs VIN"}).
    3. Click 'Continue' on the second pause (after).
    """
    logger.info("--- Entering Agency Review Pause Node ---")
    logger.info("Graph paused BEFORE this node. Click 'Continue' to proceed to the 'after' pause.")
    logger.info("If simulating review, manually edit 'human_feedback' state *before* the SECOND 'Continue'.")
    # No state modification needed here by the node itself.
    return {}

def check_agency_review(state: AgentState) -> str:
    """
    Conditional edge logic after the planner node.
    Routes to review, execution, or end based on planner's decision.
    """
    logger.info("--- Checking Agency Review Requirement --- ")
    requires_review = state.requires_agency_review
    planned_tool_inputs = state.planned_tool_inputs

    logger.debug(f"Requires Review Flag: {requires_review}")
    logger.debug(f"Planned Tool Inputs: {planned_tool_inputs}")

    if requires_review and planned_tool_inputs:
        logger.info("Routing: Planner -> Agency Review Pause")
        return "review_needed"
    elif not requires_review and planned_tool_inputs:
        logger.info("Routing: Planner -> Execute Tool")
        return "execute_tool"
    else:
        logger.info("Routing: Planner -> END (No tool planned)")
        return END 

# Define node names for clarity (will be used in graph building)
PLANNER_NODE_NAME = "planner"
EXECUTOR_NODE_NAME = "execute_tool"
AGENCY_REVIEW_NODE_NAME = "agency_review_pause"

async def execute_tool_node(state: AgentState) -> Dict[str, Any]:
    """Executes tools based on the planner's decision in messages."""
    logger.warning("execute_tool_node called, but it should be obsolete.")
    return {"last_tool_outputs": {"status": "error", "error_message": "Obsolete function called"}}

def build_agent_graph() -> StateGraph:
    """Builds the LangGraph StateGraph for the PPA Agent (V3 with Agency Review)."""
    workflow = StateGraph(AgentState)

    workflow.add_node(PLANNER_NODE_NAME, planner_node)
    workflow.add_node(EXECUTOR_NODE_NAME, executor_node_wrapper)
    workflow.add_node(AGENCY_REVIEW_NODE_NAME, agency_review_pause_node)

    workflow.add_edge(START, PLANNER_NODE_NAME)

    workflow.add_conditional_edges(
        PLANNER_NODE_NAME,
        check_agency_review,
        {
            "review_needed": AGENCY_REVIEW_NODE_NAME,
            "execute_tool": EXECUTOR_NODE_NAME,
            END: END
        }
    )

    workflow.add_edge(AGENCY_REVIEW_NODE_NAME, PLANNER_NODE_NAME)
    workflow.add_edge(EXECUTOR_NODE_NAME, PLANNER_NODE_NAME)

    app = workflow.compile(
        interrupt_before=[AGENCY_REVIEW_NODE_NAME],
        interrupt_after=[AGENCY_REVIEW_NODE_NAME] # Add this interrupt
    )
    logger.info("Agent graph V3 compiled with agency review interrupt (before & after).")
    return app

# --- Main Agent Class (Simple Runner for Now) ---
class PPAAgentRunner:
    """A helper class to manage running the agent graph for a thread."""
    def __init__(self, graph: StateGraph):
        self.graph = graph

    def start_new_thread(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """Starts a new conversation thread."""
        logger.info("Starting new agent thread.")
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        initial_state = self.graph.invoke(initial_input, config)
        logger.info(f"New thread started. Initial state: {initial_state}")
        return initial_state

    async def process_message(self, thread_id: str, message_content: str) -> List[Dict[str, Any]]:
        """Processes a single user message in an existing thread."""
        logger.info(f"Processing message for thread_id: {thread_id}")
        config = {"configurable": {"thread_id": thread_id}}
        input_data = {"messages": [HumanMessage(content=message_content)]}

        output_chunks = []
        async for chunk in self.graph.astream(input_data, config):
            logger.debug(f"Stream chunk: {chunk}")
            output_chunks.append(chunk)
            current_state = await self.graph.aget_state(config)
            if current_state.next == (AGENCY_REVIEW_NODE_NAME,):
                logger.info(f"Agent paused for Agency Review on thread {thread_id}.")
                break

        logger.info(f"Finished processing message stream for thread {thread_id}. Chunks: {len(output_chunks)}")
        return output_chunks

    async def resume_from_review(self, thread_id: str, feedback: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Resumes a thread paused for agency review with feedback."""
        logger.info(f"Resuming thread {thread_id} from review with feedback: {feedback}")
        config = {"configurable": {"thread_id": thread_id}}
        
        await self.graph.aupdate_state(config, {"human_feedback": feedback})
        logger.info(f"State updated with human feedback for thread {thread_id}.")

        output_chunks = []
        async for chunk in self.graph.astream(None, config):
            logger.debug(f"Resumed stream chunk: {chunk}")
            output_chunks.append(chunk)
            current_state = await self.graph.aget_state(config)
            if current_state.next == (AGENCY_REVIEW_NODE_NAME,):
                logger.warning(f"Agent immediately paused again for review on thread {thread_id}. Check planner logic.")
                break

        logger.info(f"Finished resumed stream for thread {thread_id}. Chunks: {len(output_chunks)}")
        return output_chunks

    async def get_thread_state(self, thread_id: str) -> AgentState:
        """Retrieves the current state of a thread."""
        logger.debug(f"Getting state for thread_id: {thread_id}")
        config = {"configurable": {"thread_id": thread_id}}
        state = await self.graph.aget_state(config)
        logger.debug(f"Retrieved state for {thread_id}")
        return state

# --- Main Execution (Example / Standalone Test) --- #
graph = build_agent_graph()