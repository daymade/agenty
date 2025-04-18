# src/ppa_agentic_v2/agent.py
import logging
import json
from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # <-- Added imports
from pydantic import ValidationError, BaseModel, Field # Use BaseException for broader catch if needed
from langgraph.prebuilt import ToolNode # Import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3 # Add sqlite3 import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI # Import OpenAI LLM
from langchain_core.utils.function_calling import convert_to_openai_tool # Import tool converter
from .state import AgentState
from .tools import all_tools, TOOL_MAP # Import TOOL_MAP
from .config import ( 
    SQLITE_DB_NAME, GOOGLE_API_KEY, OPENAI_API_KEY, 
    DEFAULT_LLM_PROVIDER, GOOGLE_MODEL_NAME, OPENAI_MODEL_NAME, logger
)
from .prompts import PLANNER_PROMPT_TEMPLATE_V2
import uuid # Import uuid for generating tool call IDs
from .prompts import format_planner_prompt # <-- Import prompt formatter
from .llm import get_llm_client # <-- Import real LLM getter
from langchain_core.messages import AIMessage # For parsing LLM response

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

    # --- Pydantic Model for Planner Output ---
    class PlannerDecision(BaseModel):
        """Represents the structured decision output from the planner LLM."""
        thought: str = Field(description="The reasoning process for the decision.")
        tool_name: Optional[str] = Field(None, description="The name of the tool to call, if any.")
        args: Optional[Dict[str, Any]] = Field(None, description="The arguments for the tool, if tool_name is provided.")

    # Bind structured output using Pydantic model
    if llm_with_tools:
        # Explicitly use function_calling method for structured output with OpenAI
        llm_with_tools = llm_with_tools.with_structured_output(PlannerDecision, method="function_calling")
        logger.info("LLM configured for structured output (PlannerDecision) using function_calling.")

except Exception as e:
    logger.error(f"Failed to initialize LLM or bind tools: {e}", exc_info=True)
    llm = None
    llm_with_tools = None

# --- Planner Prompt Template --- 
# Define the system prompt instructing the LLM
planner_system_prompt = """You are an expert insurance assistant agent tasked with collecting information for a Personal Private Auto (PPA) insurance quote.
Your goal is to gather all necessary information from the customer through conversation, using the available tools when appropriate.

Available Tools:
{tool_descriptions}

Conversation History:
{chat_history}

Based on the conversation history and the available tools, decide the next best action.

Your response MUST be a JSON object with the following structure:
{{
  "thought": "<Your reasoning for the chosen action>",
  "tool_name": "<name_of_tool_to_use_or_null>",
  "args": {{ <arguments_for_the_tool_if_calling_one> }}
}}

- If you need to use a tool, provide its name and arguments.
- If you have enough information or the conversation is complete, set 'tool_name' to null.
- If you need to ask the customer a question, use the 'ask_customer_tool'.
- Review the entire conversation history, including previous tool results, to avoid redundant actions.
- Be precise with tool arguments based on the information gathered.

Respond ONLY with the JSON object.
"""

# --- Create SQLite Checkpointer for standalone mode --- #
# Create SQLite connection
# conn = sqlite3.connect(SQLITE_DB_NAME, check_same_thread=False)
# # Create the SqliteSaver with the connection
# sqlite_memory = SqliteSaver(conn)
# logger.info(f"SQLite checkpointer initialized with database: {SQLITE_DB_NAME}")

# --- Agent Nodes ---

def planner_node(state: AgentState) -> Dict[str, Any]:
    """The core planning node. Uses LLM to decide the next action based on history and tools."""
    logger.info("--- Entering Planner Node ---")
    
    # Check if LLM initialized correctly
    if not llm_with_tools:
        logger.error("LLM with tools not initialized. Cannot proceed with planning.")
        # Return an error state or fallback behavior
        return {
            "messages": [AIMessage(content="Internal error: Planner LLM not available.", id=str(uuid.uuid4()))],
            "planned_tool_inputs": None,
            "agent_scratchpad": "Internal error: Planner LLM not available.", # Add scratchpad for clarity
        }
        
    update: Dict[str, Any] = {}
    messages: List[BaseMessage] = state.messages or []
    # Access agent_scratchpad using dot notation, default to empty string if None
    agent_scratchpad = state.agent_scratchpad if state.agent_scratchpad is not None else ""

    # --- Prepare Tool Descriptions --- 
    # Format tool descriptions for the prompt, ensuring no accidental format braces
    # Escape braces within the JSON schema itself to prevent formatting errors
    tool_description_lines = []
    for tool in all_tools:
        schema_str = str(tool.args_schema.schema()) if tool.args_schema else '{}'
        # Escape literal braces within the schema string
        escaped_schema_str = schema_str.replace('{', '{{').replace('}', '}}') 
        tool_description_lines.append(
            f"- {tool.name}: {tool.description}. Args schema: {escaped_schema_str}"
        )
    tool_descriptions = "\n".join(tool_description_lines)

    # --- Prepare Prompt for LLM --- 
    # Pass the raw system prompt template. It expects 'tool_descriptions' and 'chat_history'.
    prompt = ChatPromptTemplate.from_messages([
        # System prompt expects 'tool_descriptions' and 'chat_history' to be filled by invoke
        ("system", planner_system_prompt), 
        MessagesPlaceholder(variable_name="chat_history"),
    ])

    # Chain the prompt and LLM
    planner_chain = prompt | llm_with_tools
    
    # --- Invoke LLM --- 
    logger.info("Invoking LLM planner...")
    llm_decision: Optional[PlannerDecision] = None # Expecting PlannerDecision or None
    try:
        # --- Debugging: Log message types before invoking --- 
        logger.info("Messages being passed to planner chain:")
        for i, msg in enumerate(messages):
            logger.info(f"  Message {i}: Type={type(msg)}, Content='{str(msg.content)[:100]}...'" ) 
        # --- End Debugging --- 
        
        # --- Manually format messages to dict format --- 
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                # Handle potential tool calls in AI messages
                message_dict = {"role": "assistant", "content": msg.content or ""} # Ensure content is not None
                if msg.tool_calls:
                    message_dict["tool_calls"] = msg.tool_calls
                formatted_messages.append(message_dict)
            elif isinstance(msg, ToolMessage):
                formatted_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content
                })
            # Add other types like SystemMessage if needed, though usually handled by the prompt template
            # else: # Optional: Log unhandled types
            #     logger.warning(f"Unhandled message type in manual formatting: {type(msg)}")
        # --- End Manual Formatting ---

        # Pass the formatted message history AND tool descriptions to invoke
        invoke_input = {
            "tool_descriptions": tool_descriptions,
            "chat_history": formatted_messages # Use manually formatted list
        }
        logger.debug(f"Planner invoke input keys: {invoke_input.keys()}")
        # Invoke should now return a PlannerDecision object directly
        llm_decision = planner_chain.invoke(invoke_input)
        logger.debug(f"LLM Structured Decision: {llm_decision}")

        # --- Use Structured Decision --- 
        if llm_decision:
            # Update state with tool call information if present
            if llm_decision.tool_name:
                tool_name = llm_decision.tool_name
                # Use getattr to safely access potentially None args, default to {}
                tool_args = llm_decision.args if llm_decision.args is not None else {}
                tool_id = str(uuid.uuid4())
                tool_calls = [{
                    "name": tool_name,
                    "args": tool_args,
                    "id": tool_id
                }]
                
                # Create AIMessage with tool_calls for the executor to use
                # Use the 'thought' from the decision as the content
                # Safely access thought, default to a generic message
                ai_message_content = llm_decision.thought if llm_decision.thought else f"Planning to use tool: {tool_name}"
                ai_message = AIMessage(
                    content=ai_message_content,
                    tool_calls=tool_calls,
                    id=str(uuid.uuid4()) # Ensure AIMessage has a unique ID
                )
                
                # Add the AIMessage to state.messages
                update["messages"] = [ai_message]
                logger.info(f"Adding AIMessage with tool_calls to state: {ai_message}")
                
                # Store planned inputs for potential debugging/tracking (optional)
                update["planned_tool_inputs"] = {
                    "tool_name": tool_name,
                    "args": tool_args
                }
                logger.info(f"Planner setting planned_tool_inputs: {update['planned_tool_inputs']}")
            else:
                # No tool to call, just add the thought as a message
                update["planned_tool_inputs"] = None
                # Safely access thought, default to a generic message
                final_thought = llm_decision.thought if llm_decision.thought else "No further action planned."
                ai_message = AIMessage(content=final_thought, id=str(uuid.uuid4())) # Ensure unique ID
                update["messages"] = [ai_message]
                logger.info("Planner decided no tool to execute. Adding final thought.")
        else:
            # Handle case where LLM invocation failed or returned None
            logger.error("LLM planner returned None or failed to produce a structured decision.")
            update["messages"] = [AIMessage(content="Internal error: Planner failed to produce a decision.", id=str(uuid.uuid4()))]
            update["planned_tool_inputs"] = None
            # Update scratchpad too if needed
            current_scratch = agent_scratchpad
            update["agent_scratchpad"] = f"{current_scratch}\nThought: Planner failed.".strip()

    except Exception as e:
        logger.error(f"Error invoking LLM planner or processing structured output: {e}", exc_info=True)
        # Add error message to state
        ai_message = AIMessage(content=f"Error during planning: {e}", id=str(uuid.uuid4()))
        update["messages"] = [ai_message]
        # Potentially add a message asking the user to retry or informing about the error

    # Update scratchpad with the thought from the decision
    if llm_decision and llm_decision.thought:
        current_scratch = agent_scratchpad # Get current scratchpad from state
        update["agent_scratchpad"] = f"{current_scratch}\nThought: {llm_decision.thought}".strip()
    
    # We only return the *changes* for LangGraph to merge
    logger.info("--- Exiting Planner Node ---")
    return update

async def execute_tool_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes tools based on the tool_calls in the last AIMessage.
    Also increments the loop counter.
    """
    logger.info("--- Entering Execute Tool Node ---")
    # logger.debug(f"Current State: {state}")
    
    # Initialize update for what the node changes in the state
    update = {
        "last_tool_outputs": None
    }
    
    # Find the last AIMessage with tool_calls
    last_ai_message = None
    for msg in reversed(state.messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            last_ai_message = msg
            break
    
    if not last_ai_message or not getattr(last_ai_message, "tool_calls", None):
        logger.warning("Executor node called without pending tool calls in the last message.")
        # Return early if there are no tool calls to execute
        return update
    
    logger.info(f"Found AI message with tool calls: {last_ai_message}")
    tool_call = last_ai_message.tool_calls[0]  # Assume single tool call for now
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args", {})
    tool_id = tool_call.get("id")
    
    # Find the tool in our available tools
    tool_fn = None
    for tool in all_tools:
        if tool.name == tool_name:
            tool_fn = tool
            break
    
    if not tool_fn:
        logger.error(f"Tool '{tool_name}' not found in available tools")
        error_msg = f"Tool '{tool_name}' not found."
        update["last_tool_outputs"] = {"status": "error", "message": error_msg}
        # Add a ToolMessage with the error
        update["messages"] = [ToolMessage(
            content=json.dumps({"status": "error", "message": error_msg}),
            tool_call_id=tool_id,
            name=tool_name
        )]
        return update
    
    # Execute the tool
    logger.info(f"Executing tool '{tool_name}' with args: {tool_args}")
    try:
        # Use invoke() method for LangChain tools, passing the args dictionary
        tool_result = await tool_fn.ainvoke(tool_args) # Use ainvoke for async
        logger.info(f"Tool execution result: {tool_result}")
        
        # Add a ToolMessage with the result
        update["last_tool_outputs"] = tool_result
        update["messages"] = [ToolMessage(
            content=json.dumps(tool_result),
            tool_call_id=tool_id,
            name=tool_name
        )]
    except Exception as e:
        logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
        error_msg = f"Error executing tool '{tool_name}': {str(e)}"
        update["last_tool_outputs"] = {"status": "error", "message": error_msg}
        update["messages"] = [ToolMessage(
            content=json.dumps({"status": "error", "message": error_msg}),
            tool_call_id=tool_id,
            name=tool_name
        )]
    
    # Increment loop counter
    # Access loop_count using dot notation for Pydantic/TypedDict
    current_loop_count = state.loop_count 
    next_loop_count = current_loop_count + 1
    logger.info(f"Incrementing loop counter from {current_loop_count} to {next_loop_count}")

    # Update state with the tool results and incremented loop count
    update["loop_count"] = next_loop_count

    logger.info("--- Exiting Execute Tool Node ---")
    return update

# Define a maximum loop count to prevent infinite loops
MAX_LOOPS = 5 

def should_execute_tool(state: AgentState) -> str:
    """Determines whether to execute a tool or end the flow.
    
    Checks loop count first, then if the last message is an AIMessage with tool_calls.
    """
    logger.info("--- Checking routing: execute tool or end? ---")
    
    # Check loop count first
    # Access loop_count using dot notation for Pydantic/TypedDict
    loop_count = state.loop_count
    if loop_count >= MAX_LOOPS:
        logger.warning(f"Loop count ({loop_count}) reached maximum ({MAX_LOOPS}). Forcing END.")
        return END
        
    # Original check: Does the last message have tool calls?
    last_message = state.messages[-1] if state.messages else None
    
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        logger.info(f"Tool calls found (Loop {loop_count}). Routing to 'execute_tool'.")
        return "execute_tool"
    else:
        logger.info(f"No tool calls found (Loop {loop_count}). Routing to END.")
        return END # Use the predefined END constant

def build_agent_graph(checkpointer=None):
    """
    Builds the LangGraph workflow according to the design document.
    Accepts an optional checkpointer instance.
    Flow: START -> planner -> (conditional edge) -> execute_tool -> planner
                                         -> END
    """
    # Define the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("execute_tool", execute_tool_node)
    
    # Configure graph connections
    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges(
        "planner",
        should_execute_tool,
        {
            "execute_tool": "execute_tool",  # Map string "execute_tool" to node
            END: END                     # Map END constant to END node
        }
    )
    workflow.add_edge("execute_tool", "planner")
    
    # --- Compile Graph --- 
    # Pass the checkpointer instance (or None) directly to compile
    if checkpointer:
        compiled = workflow.compile(checkpointer=checkpointer)
    else:
        compiled = workflow.compile()
        
    return compiled

# --- Main Agent Class (Simple Runner for Now) ---
class PPAAgentRunner:
    def __init__(self, use_sqlite_persistence=True):
        """Initialize the PPA Agent runner.
        
        Args:
            use_sqlite_persistence (bool): Whether to use SQLite persistence.
        """
        self.conn = None
        checkpointer = None
        self._graph_compiled = False # Flag to track compilation
        
        try:
            if use_sqlite_persistence:
                # Establish connection here, keep it open for the runner's lifetime
                # check_same_thread=False is important for potential async usage (like FastAPI)
                self.conn = sqlite3.connect(SQLITE_DB_NAME, check_same_thread=False)
                checkpointer = SqliteSaver(conn=self.conn)
                logger.info(f"Initialized SQLite checkpointer (DB: {SQLITE_DB_NAME})")
            else:
                 logger.info("Using in-memory checkpointer.")
            
            # Build the graph, passing the checkpointer instance (or None)
            self.graph = build_agent_graph(checkpointer=checkpointer)
            self._graph_compiled = True
            logger.info("PPAAgentRunner graph compilation successful.")
            
        except Exception as e:
            logger.error(f"Error during PPAAgentRunner initialization: {e}", exc_info=True)
            # Ensure connection is closed if initialization fails partially
            if self.conn:
                self.conn.close()
            raise # Re-raise the exception after logging and cleanup

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
            for step in self.graph.stream({"messages": input_messages}, config=config, stream_mode="values"):
                final_state_dict = step  # Keep latest state dict
                step_name = list(step.keys())[0]  # Get node name
                logger.info(f"Step: {step_name} completed.")
                # logger.debug(f"State after {step_name}: {step}")  # Log intermediate state if needed

            logger.info(f"--- Agent Turn Complete (Thread: {thread_id}) ---")
            # Get the final state from the checkpointer for inspection
            final_state_persisted = self.graph.get_state(config)
            if final_state_persisted:
                logger.info(f"Final Persisted Messages Count: {len(final_state_persisted.values.get('messages', []))}")
                # Safely access last_tool_outputs
                last_outputs = final_state_persisted.values.get('last_tool_outputs', 'N/A') 
                logger.info(f"Last Tool Outputs: {last_outputs}")
                return final_state_persisted.values  # Return the dict representation
            else:
                logger.error(f"Could not retrieve final state for thread {thread_id}")
                return {"error": "Failed to retrieve final state"}

        except Exception as e:
             logger.error(f"Error during agent run (Thread: {thread_id}): {e}", exc_info=True)
             try:
                 # Try to get state even if run failed mid-way
                 current_state = self.graph.get_state(config)
                 return current_state.values if current_state else {"error": str(e)}
             except Exception:
                  logger.error(f"Could not retrieve state for thread {thread_id} after error.")
                  return {"error": str(e)}  # Return error dict

    def get_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the current state for a given thread_id."""
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = self.graph.get_state(config)
            return state.values if state else None
        except Exception as e:
            logger.error(f"Error retrieving state for thread {thread_id}: {e}", exc_info=True)
            return None

    def close(self):
        """Closes resources like the database connection."""
        if self.conn:
            try:
                self.conn.close()
                logger.info("SQLite connection closed explicitly.")
                self.conn = None
            except Exception as e:
                 logger.error(f"Error closing SQLite connection: {e}", exc_info=True)

    def __del__(self):
        # Fallback cleanup if close() isn't called
        self.close()

# --- Compile graphs at module level for different use cases ---
# Build graph without persistence for direct use (e.g., LangGraph server 'graph' export)
try:
    # Create the basic graph
    _internal_graph = build_agent_graph(checkpointer=None)
    logger.info("Internal agent graph compiled (In-Memory Persistence).")
    
    # Create a dedicated handler function for LangGraph dev/Studio
    async def langgraph_handle_message(message_input):
        """Handles incoming messages, converts to state, invokes graph, and returns result."""
        thread_id = message_input.get("thread_id")
        
        # --- Check for Empty/Initialization Messages --- 
        # Ignore empty messages often sent on initial connection/reload by dev server/studio
        if not message_input or message_input == {}:
            logger.warning("Received empty message input, likely from server init/reload. Ignoring.")
            # Returning None might be suitable, or adjust based on how LangServe handles it.
            # Consider returning an empty dict or a specific status if None causes issues upstream.
            return None 
        
        logger.info(f"LangGraph Studio received message: {type(message_input)}")
        
        # --- State Creation/Update --- 
        # Create a new state with the message
        if isinstance(message_input, str):
            # Convert string to HumanMessage
            human_msg = HumanMessage(content=message_input)
            state = AgentState(messages=[human_msg])
        elif isinstance(message_input, dict) and "content" in message_input:
            # Handle dict format (common in API requests)
            human_msg = HumanMessage(content=message_input["content"])
            state = AgentState(messages=[human_msg])
        elif isinstance(message_input, BaseMessage):
            # Direct message object
            state = AgentState(messages=[message_input])
        else:
            # Unknown format, log warning and try to handle gracefully
            logger.warning(f"Received unknown message format: {message_input}")
            human_msg = HumanMessage(content=str(message_input))
            state = AgentState(messages=[human_msg])
            
        # Process through the graph
        try:
            # Set environment variable to isolate blocking operations
            # This is equivalent to using --allow-blocking flag 
            import os
            os.environ["BG_JOB_ISOLATED_LOOPS"] = "true"
            
            # Using _internal_graph for direct invocation
            result = await _internal_graph.ainvoke(state)
            logger.info(f"Message processed successfully, result: {result}")
            # Extract final messages for return if needed, or return the full state
            # For Studio, returning the final state might be more informative
            return result
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            # Return an error state or raise?
            # For Studio, maybe return a state with the error message
            error_state = AgentState(messages=[AIMessage(content=f"Error: {e}")])
            return error_state
    
    # Export the COMPILED GRAPH object as the main entry point for LangGraph dev/Studio visualization
    graph = _internal_graph 
    
except Exception as e:
    logger.error(f"Failed to compile graph or define handler: {e}", exc_info=True)
    graph = None # Assign None or handle error appropriately