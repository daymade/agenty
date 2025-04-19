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
def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    The core planning node. Uses LLM to decide the next action based on state,
    including history, tool outputs, and human feedback. Outputs JSON.
    """
    logger.info("--- Entering Planner Node (V3 Logic) ---")
    update: Dict[str, Any] = {}

    # Check if LLM initialized correctly
    if not llm_with_tools: 
        logger.error("LLM with tools not initialized. Cannot proceed with planning.")
        update["agent_scratchpad"] = "Internal error: Planner LLM not available."
        update["requires_agency_review"] = True
        update["planned_tool_inputs"] = None
        update["last_tool_outputs"] = None 
        update["human_feedback"] = None 
        return update

    # --- Prepare Prompt using V3 Formatter ---
    try:
        formatted_prompt = format_planner_prompt(state, all_tools)
        logger.debug(f"Formatted Planner Prompt V3:\n{formatted_prompt}")
    except Exception as e:
        logger.error(f"Error formatting planner prompt: {e}", exc_info=True)
        update["agent_scratchpad"] = f"Internal error: Failed to format prompt - {e}"
        update["requires_agency_review"] = True 
        update["planned_tool_inputs"] = None
        update["last_tool_outputs"] = None
        update["human_feedback"] = None
        return update

    # --- Invoke LLM ---
    logger.info("Invoking LLM planner...")
    raw_llm_output = ""
    parsed_data: Optional[Dict[str, Any]] = None
    try:
        llm_response = llm_with_tools.invoke(formatted_prompt)

        if isinstance(llm_response, AIMessage) and llm_response.content:
            raw_llm_output = llm_response.content
            logger.debug(f"Raw LLM Output:\n{raw_llm_output}")

            json_block_match = re.search(r"```json\n(.*?)\n```", raw_llm_output, re.DOTALL)
            if json_block_match:
                json_str = json_block_match.group(1).strip()
            else:
                json_str = raw_llm_output.strip()

            parsed_data = json.loads(json_str)
            logger.info(f"Successfully parsed LLM JSON: {parsed_data}")

        else:
            raise ValueError(f"Unexpected LLM response format: {type(llm_response)}")

    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.error(f"Failed to parse LLM JSON output: {e}", exc_info=True)
        logger.error(f"Raw LLM output was: {raw_llm_output}")
        update["agent_scratchpad"] = f"Error: Failed to parse planner JSON response. Raw: {raw_llm_output}"
        update["requires_agency_review"] = True
        update["planned_tool_inputs"] = None

    except OutputParserException as e: 
        logger.error(f"LangChain OutputParserException from LLM: {e}", exc_info=True)
        update["agent_scratchpad"] = f"Error: LLM output parsing failed. Details: {e}"
        update["requires_agency_review"] = True
        update["planned_tool_inputs"] = None

    except Exception as e: 
        logger.error(f"Unexpected error invoking planner LLM: {e}", exc_info=True)
        update["agent_scratchpad"] = f"Internal error: Planner invocation failed - {e}"
        update["requires_agency_review"] = True
        update["planned_tool_inputs"] = None


    if parsed_data:
        update["agent_scratchpad"] = parsed_data.get("thought", "No thought provided.")
        update["requires_agency_review"] = bool(parsed_data.get("requires_review", False))

        tool_name = parsed_data.get("tool_name")
        args = parsed_data.get("args") 

        if tool_name:
            update["planned_tool_inputs"] = {
                "tool_name": tool_name,
                "args": args if args is not None else {} 
            }
            logger.info(f"Planner decided action: {tool_name} with args: {update['planned_tool_inputs']['args']}")
            logger.info(f"Requires Agency Review: {update['requires_agency_review']}")
        else:
            update["planned_tool_inputs"] = None
            logger.info("Planner decided no tool action needed.")
            if "requires_agency_review" not in update: 
                update["requires_agency_review"] = False

    update["last_tool_outputs"] = None 
    update["human_feedback"] = None 

    logger.info(f"--- Exiting Planner Node. State updates: { {k: v for k, v in update.items() if v is not None} } ---")
    return update

def executor_node_wrapper(state: AgentState) -> Dict[str, Any]:
    """
    Executes the tool specified in state.planned_tool_inputs.
    Handles errors and updates state with the result and clears the plan.
    """
    logger.info("--- Entering Executor Node Wrapper ---")
    update: Dict[str, Any] = {}
    tool_result: Optional[Dict[str, Any]] = None 

    planned_inputs = state.planned_tool_inputs
    if not planned_inputs or not planned_inputs.get("tool_name"):
        logger.warning("Executor node called without planned_tool_inputs. No action taken.")
        update["last_tool_outputs"] = {
            "status": "error",
            "error_message": "Executor called without a planned tool."
        }
        update["planned_tool_inputs"] = None 
        return update

    tool_name = planned_inputs["tool_name"]
    tool_args = planned_inputs.get("args", {}) 

    logger.info(f"Attempting to execute tool: {tool_name} with args: {tool_args}")

    tool_to_execute = None
    for tool in all_tools:
        if tool.name == tool_name:
            tool_to_execute = tool
            break

    if not tool_to_execute:
        logger.error(f"Tool '{tool_name}' planned but not found in available tools.")
        tool_result = {
            "status": "error",
            "error_message": f"Tool '{tool_name}' not found.",
            "tool_name": tool_name 
        }
    else:
        try:
            output = tool_to_execute.invoke(tool_args)
            logger.info(f"Tool '{tool_name}' executed successfully. Output type: {type(output)}")
            logger.debug(f"Tool '{tool_name}' raw output: {output}")
            tool_result = {
                "status": "success",
                "output": output, 
                "tool_name": tool_name
            }

        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
            tool_result = {
                "status": "error",
                "error_message": str(e),
                "tool_name": tool_name
            }

    update["last_tool_outputs"] = tool_result
    update["planned_tool_inputs"] = None 

    logger.info(f"--- Exiting Executor Node Wrapper. Result: {tool_result['status']} ---")
    return update

def agency_review_pause_node(state: AgentState) -> Dict[str, Any]:
    """Placeholder node for the agency review interrupt point. Does nothing.
    The graph pauses *before* this node is executed.
    When resumed, the graph executes this node and proceeds to the next edge.
    """
    logger.info("--- Entering Agency Review Pause Node (post-resume) ---")
    # No state modification needed here, just passing through after resume
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

    workflow.add_edge(AGENCY_REVIEW_NODE_NAME, EXECUTOR_NODE_NAME)
    workflow.add_edge(EXECUTOR_NODE_NAME, PLANNER_NODE_NAME)

    app = workflow.compile(
        interrupt_before=[AGENCY_REVIEW_NODE_NAME]
    )
    logger.info("Agent graph V3 compiled with agency review interrupt.")
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