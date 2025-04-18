Okay, let's embark on building the agentic PPA quoting system step-by-step, starting from scratch and progressively implementing the v1.2 design. We will reuse concepts and logic from your initial codebase where applicable, focusing on the agentic refactoring.

**Project Setup**

1.  **Create Project Directory:**
    ```bash
    mkdir mercury-ppa-agent
    cd mercury-ppa-agent
    ```
2.  **Initialize Poetry (Optional but Recommended):**
    ```bash
    poetry init --name mercury-ppa-agent --dependency python:^3.11 --dev-dependency pytest -n
    poetry add langgraph langchain-core langchain-openai langchain-google-genai pydantic python-dotenv sqlite
    # Add other dependencies as needed later (e.g., requests for API calls)
    ```
3.  **Create Directory Structure:**
    ```
    mercury-ppa-agent/
    ├── src/
    │   └── ppa_agent/
    │       ├── __init__.py
    │       ├── state.py
    │       ├── config.py
    │       ├── tools.py
    │       ├── llm.py
    │       ├── agent.py        # Will contain graph definition and main agent class
    │       └── prompts.py      # For planner prompts
    ├── scripts/
    │   └── run_agent.py        # For testing milestones
    ├── tests/
    ├── .env.example
    ├── .env
    ├── pyproject.toml
    └── README.md
    ```
4.  **Create `.env.example`:**
    ```dotenv
    # src/ppa_agent/.env.example
    # Choose ONE provider or provide keys for both if needed
    # OPENAI_API_KEY="your_openai_api_key_here"
    GOOGLE_API_KEY="your_google_api_key_here" # Preferred (uses google-genai)

    # Optional: Specify models if different from defaults
    # OPENAI_MODEL_NAME="gpt-4-turbo-preview"
    # GOOGLE_MODEL_NAME="gemini-1.5-pro-latest"
    ```
5.  **Create `.env`:** Copy `.env.example` to `.env` and add your actual API keys.
6.  **Basic Logging Setup (Optional - can add later):** Configure standard Python logging.

---

**Milestone 1: Foundation - Configuration and State Definition**

This milestone sets up the basic configuration and defines the agent's state using Pydantic.

*   **`src/ppa_agent/config.py`**:
    ```python
    # src/ppa_agent/config.py
    import os
    from dotenv import load_dotenv
    import logging

    # Load environment variables from .env file
    load_dotenv()

    # --- API Keys ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Preferred for google-genai

    # --- Model Selection ---
    # Prioritize Google API key if present
    DEFAULT_LLM_PROVIDER = "google" if GOOGLE_API_KEY else "openai"
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo-preview")
    # Use a capable Gemini model - 1.5 Pro recommended for planning
    GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-pro-latest")

    # --- Logging ---
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    if not GOOGLE_API_KEY and not OPENAI_API_KEY:
        logger.warning("Neither GOOGLE_API_KEY nor OPENAI_API_KEY found in .env. LLM functionality will be limited.")
    elif DEFAULT_LLM_PROVIDER == "google":
         logger.info(f"Using Google LLM Provider (Model: {GOOGLE_MODEL_NAME})")
    else:
         logger.info(f"Using OpenAI LLM Provider (Model: {OPENAI_MODEL_NAME})")

    # --- PPA Quote Requirements (From original code) ---
    # Define the fields needed for a complete quote for reference
    PPA_QUOTE_REQUIREMENTS = [
        "driver_name", "driver_age", "driver_dob", "driver_license_number",
        "address_line1", "city", "state_code", "zip_code",
        "vehicle_year", "vehicle_make", "vehicle_model", "vehicle_vin",
        "coverage_limits", "deductibles",
        # Add any other essential fields identified by Mercury APIs
    ]

    # --- Database for Persistence ---
    SQLITE_DB_NAME = "ppa_agent_state.sqlite"

    ```

*   **`src/ppa_agent/state.py`**:
    ```python
    # src/ppa_agent/state.py
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
        goal: str = Field(default="Generate accurate PPA quote for the customer", description="The overall objective of the agent.")
        # Use Annotated List[BaseMessage] for automatic message appending by LangGraph
        messages: Annotated[List[BaseMessage], operator.add] = Field(default_factory=list, description="Full history of messages (Human, AI, Tool).")

        # == Extracted & API Data ==
        customer_info: Dict[str, Any] = Field(default_factory=dict, description="Accumulated structured data about customer, drivers, vehicles etc.")
        mercury_session: Optional[Dict[str, Any]] = Field(default=None, description="Stores context from Mercury APIs (e.g., Quote ID, session tokens).")

        # == Agent Planning & Execution State ==
        planned_tool_inputs: Optional[Dict[str, Any]] = Field(default=None, description="The tool name and arguments planned by the Planner for the *next* action. Reviewed during Agency Review.")
        last_tool_outputs: Optional[Dict[str, Any]] = Field(default=None, description="The dictionary result {'status': 'success'|'error', ...} from the *last executed* tool.")
        agent_scratchpad: Optional[str] = Field(default=None, description="Internal reasoning or notes from the Planner.")

        # == HITL Control Flags & Feedback ==
        requires_agency_review: bool = Field(default=False, description="Set by Planner if the planned action in 'planned_tool_inputs' needs human review before execution.")
        awaiting_customer_reply: bool = Field(default=False, description="Set by Planner *before* triggering the 'Wait for Customer' interrupt.")
        human_feedback: Optional[HumanFeedback] = Field(default=None, description="Feedback received from the Agency Review interrupt. Processed and cleared by the Planner.")

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
    ```

---

**Milestone 2: Basic Agent Loop & Mocked Tools**

We'll create the basic Planner -> Executor loop. The Planner will use a *mocked* LLM initially to control its output, and the Executor will use *mocked* tools. No persistence or HITL yet.

*   **`src/ppa_agent/llm.py`**:
    ```python
    # src/ppa_agent/llm.py
    import logging
    from typing import Optional, Dict, Any
    from .config import (
        DEFAULT_LLM_PROVIDER, GOOGLE_API_KEY, GOOGLE_MODEL_NAME,
        OPENAI_API_KEY, OPENAI_MODEL_NAME
    )

    # Conditional imports based on available keys
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        ChatGoogleGenerativeAI = None # type: ignore

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        ChatOpenAI = None # type: ignore

    logger = logging.getLogger(__name__)

    def get_llm_client(provider: Optional[str] = None, model: Optional[str] = None):
        """Initializes and returns the LLM client based on config."""
        provider = provider or DEFAULT_LLM_PROVIDER

        if provider == "google" and ChatGoogleGenerativeAI and GOOGLE_API_KEY:
            model_name = model or GOOGLE_MODEL_NAME
            logger.info(f"Initializing Google LLM: {model_name}")
            try:
                # Configure for JSON mode where needed by Planner
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.1, # Lower temp for planning
                    convert_system_message_to_human=True # Often needed for Gemini
                    # Add generation_config={'response_mime_type': 'application/json'} if needed globally
                    # Or handle JSON output format in the prompt itself
                )
            except Exception as e:
                logger.error(f"Failed to initialize Google LLM: {e}", exc_info=True)

        if provider == "openai" and ChatOpenAI and OPENAI_API_KEY:
            model_name = model or OPENAI_MODEL_NAME
            logger.info(f"Initializing OpenAI LLM: {model_name}")
            try:
                # Configure for JSON mode
                return ChatOpenAI(
                    model=model_name,
                    openai_api_key=OPENAI_API_KEY,
                    temperature=0.1, # Lower temp for planning
                    model_kwargs={"response_format": {"type": "json_object"}} # Request JSON
                )
            except Exception as e:
                 logger.error(f"Failed to initialize OpenAI LLM: {e}", exc_info=True)

        logger.error(f"Could not initialize LLM provider '{provider}'. Check API keys and dependencies.")
        raise ValueError(f"Failed to initialize LLM provider: {provider}")

    # --- MOCK LLM for early testing ---
    class MockPlannerLLM:
        """Simulates the planner LLM's output for testing the graph structure."""
        def __init__(self, plan: list):
            self.plan_iterator = iter(plan)
            self.last_output = None

        def invoke(self, prompt: Any, config: Optional[Dict] = None) -> Any:
            try:
                output = next(self.plan_iterator)
                logger.info(f"[MOCK LLM] Planner outputting: {output}")
                self.last_output = output
                # Simulate LangChain message structure if needed by parser
                from langchain_core.messages import AIMessage
                import json
                return AIMessage(content=json.dumps(output))
            except StopIteration:
                logger.warning("[MOCK LLM] Plan exhausted. Returning last output or error.")
                if self.last_output:
                     return AIMessage(content=json.dumps(self.last_output))
                # Simulate ending the process if plan runs out
                return AIMessage(content=json.dumps({"action": "complete"}))

    # Example usage later:
    # mock_plan = [
    #     {"tool_name": "mock_tool_1", "args": {"param": "value1"}},
    #     {"tool_name": "mock_tool_2", "args": {"param": "value2"}},
    #     {"action": "complete"}
    # ]
    # mock_llm = MockPlannerLLM(mock_plan)
    ```

*   **`src/ppa_agent/tools.py`**:
    ```python
    # src/ppa_agent/tools.py
    from langchain.tools import tool
    from pydantic import BaseModel, Field
    from typing import List, Dict, Any
    import logging
    import time
    import random

    logger = logging.getLogger(__name__)

    # --- Mock Tools for Milestone 2 ---

    class MockToolInput(BaseModel):
        detail: str = Field(description="Some detail needed by the mock tool")

    @tool("mock_quote_initiate", args_schema=MockToolInput)
    def mock_quote_initiate_tool(detail: str) -> Dict[str, Any]:
        """MOCK: Simulates initiating a quote. Returns a mock quote ID."""
        logger.info(f"Executing mock_quote_initiate_tool with detail: {detail}")
        time.sleep(0.5) # Simulate work
        if random.random() < 0.1: # Simulate occasional failure
             logger.error("MOCK Failure: quote_initiate failed randomly.")
             return {"status": "error", "message": "Random API failure during quote initiation."}
        mock_id = f"MOCK_Q_{random.randint(1000, 9999)}"
        logger.info(f"MOCK Success: quote_initiate succeeded. Quote ID: {mock_id}")
        return {"status": "success", "quote_id": mock_id, "session_detail": f"Session for {detail}"}

    class MockAskCustomerInput(BaseModel):
        missing_fields: List[str] = Field(description="List of fields to ask the customer about.")

    @tool("mock_ask_customer", args_schema=MockAskCustomerInput)
    def mock_ask_customer_tool(missing_fields: List[str]) -> Dict[str, Any]:
        """MOCK: Simulates generating a message asking the customer for info."""
        logger.info(f"Executing mock_ask_customer_tool for fields: {missing_fields}")
        if not missing_fields:
            logger.error("MOCK Failure: ask_customer called with no missing fields.")
            return {"status": "error", "message": "No fields specified to ask the customer."}
        message = f"MOCK MESSAGE: Please provide: {', '.join(missing_fields)}"
        logger.info(f"MOCK Success: Generated message: '{message}'")
        return {"status": "success", "message_content": message, "message_type": "info_request"}

    @tool("mock_api_step_2")
    def mock_api_step_2_tool() -> Dict[str, Any]:
        """MOCK: Simulates a generic second API step."""
        logger.info("Executing mock_api_step_2_tool")
        time.sleep(0.3)
        logger.info("MOCK Success: API Step 2 completed.")
        return {"status": "success", "step_2_data": "Some data from step 2"}

    # --- Tool Registry ---
    # Collect all defined tools
    all_tools = [
        mock_quote_initiate_tool,
        mock_ask_customer_tool,
        mock_api_step_2_tool,
    ]
    ```

*   **`src/ppa_agent/prompts.py`**: (Define Planner Prompt - Initial Version)
    ```python
    # src/ppa_agent/prompts.py

    # Initial, simple planner prompt for Milestone 2 (using mocked tools)
    # This will become much more complex later.
    PLANNER_PROMPT_TEMPLATE_V1 = """
You are an AI assistant helping process PPA insurance quote requests. Your goal is: {goal}.

Conversation History:
{messages_str}

Current Extracted Information:
{customer_info_str}

Last Action Result:
{last_tool_outputs_str}

Available Tools:
{tools_str}

Based on the goal and the conversation history, decide the next logical action.
You MUST choose one of the available tools or decide the process is complete.
Respond ONLY with a JSON object in the following format:

{{
  "thought": "Your reasoning for choosing the next step.",
  "tool_name": "Name of the tool to use (e.g., 'mock_quote_initiate', 'mock_ask_customer') OR null if complete.",
  "args": {{ "arg_name": "value", ... }} # Arguments for the chosen tool, matching its schema. Empty {} if tool takes no args. OR null if complete.
}}

Example Response:
{{
  "thought": "The customer just started the request. I need to initiate the quote process.",
  "tool_name": "mock_quote_initiate",
  "args": {{ "detail": "Initial request" }}
}}

Current Task: Decide the next action. Respond with JSON only.
"""
    ```

*   **`src/ppa_agent/agent.py`**: (Basic Graph Structure)
    ```python
    # src/ppa_agent/agent.py
    import logging
    import json
    from typing import List, Dict, Any, Optional
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
    from langchain_core.pydantic_v1 import ValidationError # Use BaseException for broader catch if needed

    from .state import AgentState
    from .tools import all_tools # Import the list of tools
    from .prompts import PLANNER_PROMPT_TEMPLATE_V1
    from .llm import get_llm_client, MockPlannerLLM # Import mock LLM for now

    logger = logging.getLogger(__name__)

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
            update["planned_tool_inputs"] = {
                "tool_name": tool_name,
                "args": llm_decision.get("args", {})
            }
            logger.info(f"Planner setting planned_tool_inputs: {update['planned_tool_inputs']}")
        else:
            # Planner decided to complete or errored
            update["planned_tool_inputs"] = None
            logger.info("Planner decided no tool to execute (complete/error).")

        # Add thought process to scratchpad if available
        if "thought" in llm_decision:
             update["agent_scratchpad"] = llm_decision["thought"]

        return update

    # Use prebuilt ToolNode for execution
    execute_tool_node = ToolNode(all_tools)

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

    def build_agent_graph():
        """Builds the LangGraph StateGraph."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("planner", planner_node)
        workflow.add_node("executor", execute_tool_node)

        workflow.set_entry_point("planner")

        # Define edges for the basic loop
        workflow.add_conditional_edges(
            "planner",
            should_execute, # Decide whether to execute or end
            {
                "execute": "executor",
                END: END
            }
        )
        # After execution, always go back to planner
        workflow.add_edge("executor", "planner")

        # Compile the graph (NO checkpointer or interrupts yet)
        app = workflow.compile()
        logger.info("Agent graph compiled (Milestone 2 - No Persistence/HITL).")
        return app

    # --- Main Agent Class (Simple Runner for Now) ---
    class PPAAgentRunner:
        def __init__(self):
            self.graph = build_agent_graph()
            # Reset mock step counter on initialization
            if hasattr(planner_node, "mock_step"):
                 del planner_node.mock_step


        def run(self, initial_message: str):
            logger.info(f"--- Starting Agent Run (Input: '{initial_message}') ---")
            # Initial state for a new conversation
            initial_state = AgentState(messages=[HumanMessage(content=initial_message)])
            # No config needed yet as no persistence/thread_id

            final_state_dict = {}
            try:
                # Use stream or invoke - invoke returns final state
                for step in self.graph.stream(initial_state):
                    step_name = list(step.keys())[0]
                    step_data = list(step.values())[0]
                    logger.info(f"Step: {step_name}")
                    # logger.debug(f"Step Data: {step_data}")
                    final_state_dict = step_data # Keep track of the latest full state

                logger.info("--- Agent Run Complete ---")
                final_state = AgentState(**final_state_dict) # Convert back for type hints
                logger.info(f"Final Messages: {final_state.messages}")
                logger.info(f"Final Customer Info: {final_state.customer_info}")
                logger.info(f"Last Tool Outputs: {final_state.last_tool_outputs}")
                return final_state

            except Exception as e:
                 logger.error(f"Error during agent run: {e}", exc_info=True)
                 return AgentState(**final_state_dict) # Return state up to the error point
    ```

*   **`scripts/run_agent.py`**: (Script for Milestone 2)
    ```python
    # scripts/run_agent.py
    import sys
    import os
    import logging

    # Ensure the src directory is in the Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from src.ppa_agent.agent import PPAAgentRunner
    from src.ppa_agent.config import logger # Use configured logger

    if __name__ == "__main__":
        logger.info("--- Milestone 2: Basic Agent Loop Test ---")
        initial_email = "Hello, I need a PPA quote."

        agent_runner = PPAAgentRunner()
        final_state = agent_runner.run(initial_email)

        print("\n--- Final State ---")
        # Use model_dump for clean Pydantic output
        print(final_state.model_dump_json(indent=2))
        print("------")

        # Example of running again (will restart the mock plan)
        # print("\n--- Running again (Restarting Plan) ---")
        # agent_runner_2 = PPAAgentRunner()
        # final_state_2 = agent_runner_2.run("Second Request")
        # print(final_state_2.model_dump_json(indent=2))
    ```

**To Run Milestone 2:**

1.  Ensure dependencies are installed (`poetry install`).
2.  Make sure `.env` is set up (though keys aren't strictly needed for mocks).
3.  Run the script: `python scripts/run_agent.py`
4.  Observe the logs. You should see the planner making decisions based on the `mock_plan`, the executor running the mock tools, and the loop progressing until the mock plan signals completion. The final state will be printed.

This completes Milestone 2. The next step is to add persistence using `SqliteSaver`.

---

**Milestone 3: Adding Persistence with `SqliteSaver`**

Modify the agent compilation and the run script to use persistence.

*   **`src/ppa_agent/agent.py`**: (Modify `build_agent_graph` and `PPAAgentRunner`)
    ```python
    # src/ppa_agent/agent.py
    # ... (imports remain the same) ...
    from langgraph.checkpoint.sqlite import SqliteSaver # <-- Import Saver
    from .config import SQLITE_DB_NAME # <-- Import DB name

    # ... (planner_node, execute_tool_node, should_execute remain the same) ...

    # --- Graph Definition ---

    def build_agent_graph():
        """Builds the LangGraph StateGraph with persistence."""
        workflow = StateGraph(AgentState)

        # Add nodes (same as before)
        workflow.add_node("planner", planner_node)
        workflow.add_node("executor", execute_tool_node)

        workflow.set_entry_point("planner")

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

        # Compile the graph WITH checkpointer
        memory = SqliteSaver.sqlite(db_name=SQLITE_DB_NAME) # <-- Use saver
        app = workflow.compile(checkpointer=memory) # <-- Compile with it
        logger.info("Agent graph compiled (Milestone 3 - WITH Persistence).")
        return app

    # --- Main Agent Class (Modified for Persistence) ---
    class PPAAgentRunner:
        def __init__(self):
            # Graph is now compiled with persistence
            self.app = build_agent_graph()
            # Reset mock step counter on initialization
            if hasattr(planner_node, "mock_step"):
                 del planner_node.mock_step

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
                for step in self.app.stream({"messages": input_messages}, config=config):
                    step_name = list(step.keys())[0]
                    step_data = list(step.values())[0]
                    logger.info(f"Step: {step_name}")
                    # logger.debug(f"Step Data: {step_data}")
                    final_state_dict = step_data

                logger.info(f"--- Agent Turn Complete (Thread: {thread_id}) ---")
                # Get the final state from the checkpointer for inspection
                final_state_persisted = self.app.get_state(config)
                logger.info(f"Final Persisted Messages: {final_state_persisted.values['messages']}")
                logger.info(f"Last Tool Outputs: {final_state_persisted.values['last_tool_outputs']}")
                return final_state_persisted.values # Return the dict representation

            except Exception as e:
                 logger.error(f"Error during agent run (Thread: {thread_id}): {e}", exc_info=True)
                 try:
                     # Try to get state even if run failed mid-way
                     current_state = self.app.get_state(config)
                     return current_state.values
                 except Exception:
                      logger.error(f"Could not retrieve state for thread {thread_id} after error.")
                      return {"error": str(e)} # Return error dict
    ```

*   **`scripts/run_agent.py`**: (Modify for multi-turn testing)
    ```python
    # scripts/run_agent.py
    import sys
    import os
    import logging
    import json

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from src.ppa_agent.agent import PPAAgentRunner
    from src.ppa_agent.config import logger, SQLITE_DB_NAME

    if __name__ == "__main__":
        logger.info("--- Milestone 3: Persistence Test ---")

        # Ensure clean slate for test if db exists
        if os.path.exists(SQLITE_DB_NAME):
            logger.warning(f"Deleting existing database: {SQLITE_DB_NAME}")
            os.remove(SQLITE_DB_NAME)

        agent_runner = PPAAgentRunner()

        # --- Simulate Conversation ---
        thread_id = "test_thread_123" # Use a consistent ID

        print(f"\n--- Turn 1 (Thread: {thread_id}) ---")
        turn1_input = "Hello, I need a PPA quote."
        state1 = agent_runner.run_turn(thread_id, turn1_input)
        print("\nState after Turn 1:")
        print(json.dumps(state1, indent=2, default=str)) # Use default=str for BaseMessage

        print(f"\n--- Turn 2 (Thread: {thread_id}) ---")
        turn2_input = "My name is Bob." # Simulate user providing more info (though mock agent ignores for now)
        state2 = agent_runner.run_turn(thread_id, turn2_input)
        print("\nState after Turn 2:")
        print(json.dumps(state2, indent=2, default=str))

        print(f"\n--- Turn 3 (Thread: {thread_id}) ---")
        # No new input, agent should continue its mock plan from where it left off
        state3 = agent_runner.run_turn(thread_id)
        print("\nState after Turn 3:")
        print(json.dumps(state3, indent=2, default=str))

        print(f"\n--- Turn 4 (Thread: {thread_id}) ---")
        state4 = agent_runner.run_turn(thread_id)
        print("\nState after Turn 4:")
        print(json.dumps(state4, indent=2, default=str)) # Should reach end of mock plan

        print("\nPersistence test complete. Check logs and state outputs.")
        print(f"Database file created at: {SQLITE_DB_NAME}")

    ```

**To Run Milestone 3:**

1.  Run `python scripts/run_agent.py`.
2.  Observe the logs. You should see multiple turns being run for the *same* `thread_id`.
3.  Notice how the mock planner progresses through its predefined steps across turns (`mock_step` counter is reset per runner instance, but state is loaded). The `messages` list in the state output should grow with each turn that had input.
4.  A `ppa_agent_state.sqlite` file will be created.

This confirms persistence is working. Next, we'll replace mock tools and the mock planner LLM with real implementations.

---

**Milestone 4: Develop Core Tools & Real Planner**

Replace mocks with real logic, focusing on `quote_initiate_tool` and `ask_customer_tool`, and enable the real LLM planner.

*   **`src/ppa_agent/tools.py`**: (Implement real tools, keep mocks for others)
    ```python
    # src/ppa_agent/tools.py
    from langchain.tools import tool
    from pydantic import BaseModel, Field
    from typing import List, Dict, Any
    import logging
    import time
    import random
    import uuid
    import requests # <-- Add dependency if needed: poetry add requests

    # from .llm import get_llm_client # If tool needs LLM for generation
    from .prompts import GENERATE_INFO_REQUEST_PROMPT_TEMPLATE # Reuse prompt

    logger = logging.getLogger(__name__)

    # --- Real Tools (Implementations based on original node logic) ---

    class QuoteInitiateInput(BaseModel):
        # Define based on ACTUAL Mercury API requirements
        driver_name: str = Field(..., description="Primary driver's full name")
        driver_dob: str = Field(..., description="Primary driver's date of birth (YYYY-MM-DD)")
        address_line1: str = Field(..., description="Street address line 1")
        city: str = Field(...)
        state_code: str = Field(..., description="2-letter state code")
        zip_code: str = Field(..., description="5-digit zip code")
        # Add other required fields for Quote - Initiate API

    @tool("quote_initiate_tool", args_schema=QuoteInitiateInput)
    def quote_initiate_tool(**kwargs) -> Dict[str, Any]:
        """
        Initiates a new PPA quote session with the Mercury Insurance API.
        Requires essential customer and address details (name, dob, address).
        Returns the new quote ID and session context on success, or an error message on failure.
        Use this as the first step when starting a new quote after gathering initial info.
        """
        api_endpoint = "https://your-mercury-api-domain.com/api/v1/quote/initiate" # Replace with actual URL
        headers = {"Authorization": "Bearer YOUR_API_TOKEN", "Content-Type": "application/json"} # Replace with actual auth

        try:
            logger.info(f"Calling Mercury Quote Initiate API with: {kwargs}")
            # response = requests.post(api_endpoint, headers=headers, json=kwargs, timeout=30)
            # response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            # data = response.json()

            # --- MOCK API CALL for development ---
            if random.random() < 0.1: raise ConnectionError("Mock network error")
            time.sleep(1) # Simulate latency
            mock_quote_id = f"Q-{uuid.uuid4().hex[:8]}"
            mock_session = {"sessionId": f"S-{uuid.uuid4().hex[:12]}", "context": "abc"}
            data = {"quoteId": mock_quote_id, "session": mock_session}
            logger.info(f"Mock API Success: Quote Initiate. Response: {data}")
            # --- END MOCK API CALL ---

            # Validate response structure if needed
            quote_id = data.get("quoteId")
            session = data.get("session")
            if not quote_id or not session:
                 raise ValueError("API response missing 'quoteId' or 'session'")

            return {
                "status": "success",
                "quote_id": quote_id,
                "session": session
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"API Network/HTTP Error calling Quote Initiate: {e}", exc_info=True)
            # Try to extract error details from response if possible
            error_detail = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json().get('message', e.response.text)
                except json.JSONDecodeError:
                    error_detail = e.response.text
            return {"status": "error", "message": f"API Error: {error_detail}"}
        except (ValueError, KeyError) as e:
             logger.error(f"API Response Error (Quote Initiate): {e}", exc_info=True)
             return {"status": "error", "message": f"Invalid API response format: {e}"}
        except Exception as e:
            logger.error(f"Unexpected Error in quote_initiate_tool: {e}", exc_info=True)
            return {"status": "error", "message": f"Unexpected error: {e}"}

    class AskCustomerInput(BaseModel):
        missing_fields: List[str] = Field(description="A list of specific information field names the customer needs to provide.")

    @tool("ask_customer_tool", args_schema=AskCustomerInput)
    def ask_customer_tool(missing_fields: List[str]) -> Dict[str, Any]:
        """
        Use this tool ONLY when specific information is missing and needs to be requested FROM THE CUSTOMER.
        Generates the text for a message asking the customer for the specified missing information fields.
        The message should be reviewed by a human agent before sending.
        Input must be a list of strings naming the missing fields (e.g., ['driver_age', 'vehicle_vin']).
        Output indicates success and contains the generated message content.
        """
        if not missing_fields:
            return {"status": "error", "message": "No missing fields specified for asking the customer."}

        # Reuse the prompt structure from the original node
        # Adapt the prompt if needed for better generation
        missing_info_str = ", ".join(missing_fields)
        prompt = GENERATE_INFO_REQUEST_PROMPT_TEMPLATE.format(missing_info_list=missing_info_str)
        logger.info(f"Generating customer request for fields: {missing_fields} using prompt.")
        logger.debug(f"Ask Customer Prompt:\n{prompt}")

        try:
            # Use a simple, dedicated LLM call for this generation, maybe not the main planner LLM
            # Or reuse the main one if configured appropriately.
            # llm_generator = get_llm_client(model="gemini-1.5-flash-latest") # Example: Use flash for generation
            # message_content = llm_generator.invoke(prompt).content

            # --- MOCK LLM Generation ---
            time.sleep(0.5)
            message_content = f"To help me provide an accurate quote, could you please share the following: {missing_info_str}?"
            logger.info(f"Mock LLM Generated message: '{message_content}'")
            # --- END MOCK ---

            if not message_content:
                 raise ValueError("LLM returned empty content for customer message.")

            return {
                "status": "success",
                "message_content": message_content.strip(),
                "message_type": "info_request" # Consistent type
            }
        except Exception as e:
            logger.error(f"Error generating customer info request message: {e}", exc_info=True)
            # Fallback message
            message_content = f"I need a bit more information to proceed. Could you please provide details on: {missing_info_str}?"
            return {
                "status": "success", # Still generated *a* message
                "message_content": message_content,
                "message_type": "info_request",
                "error_in_generation": str(e)
            }

    # --- Add other REAL tool implementations similarly ---
    # e.g., add_vehicle_tool, update_driver_tool, rate_quote_tool...

    # --- Keep some Mocks for now ---
    @tool("mock_api_step_2")
    def mock_api_step_2_tool() -> Dict[str, Any]:
        """MOCK: Simulates a generic second API step."""
        logger.info("Executing mock_api_step_2_tool")
        time.sleep(0.3)
        logger.info("MOCK Success: API Step 2 completed.")
        return {"status": "success", "step_2_data": "Some data from step 2"}


    # --- Tool Registry ---
    # Update with real tools, keep mocks for unimplemented ones
    all_tools = [
        quote_initiate_tool,
        ask_customer_tool,
        mock_api_step_2_tool,
        # Add other real/mock tools as implemented
    ]
    ```

*   **`src/ppa_agent/prompts.py`**: (Refine Planner Prompt)
    ```python
    # src/ppa_agent/prompts.py
    import json

    # --- Planner Prompt V2 (More Detailed for Real Tools) ---
    # This needs significant iteration based on testing!
    PLANNER_PROMPT_TEMPLATE_V2 = """
You are a PPA Insurance Quoting Agent assistant. Your primary goal is: {goal}.
You interact with the customer via messages and use internal tools to call Mercury Insurance APIs.

**Current Conversation State:**

*   **Conversation History (Newest Last):**
{messages_str}

*   **Current Customer Information Extracted:**
{customer_info_str}

*   **Mercury API Session Context (e.g., Quote ID):**
{mercury_session_str}

*   **Result of the VERY LAST Action Taken:**
{last_tool_outputs_str}

**Your Task:**

Based on the current state and conversation, decide the single next best action to progress towards the goal. Consider the following:

1.  **Goal Progress:** Are we closer to a full quote? What's the next logical API step (e.g., Initiate -> Add Driver -> Add Vehicle -> Rate)?
2.  **Last Action Result:** Did the last tool succeed or fail? If it failed, analyze the error message and decide how to recover (retry? ask customer? request human help?). If it succeeded, use its output (e.g., quote_id) for the next step.
3.  **Information Needs:** Does the next logical API step require information not present in 'Current Customer Information' or 'Mercury API Session Context'?
4.  **Customer Input:** Is there new information in the latest HumanMessage? Does it fulfill a previous request?

**Available Tools:**

{tools_str}

**Decision Options:**

*   **Invoke a Tool:** If ready to call an API or perform another action. Choose ONE tool and provide ALL required arguments based on its description and the current state.
*   **Ask Customer:** If specific information is needed *from the customer* for the next step. Use the 'ask_customer_tool'.
*   **Request Human Review:** If you are stuck, encounter an unrecoverable error, or reach a critical decision point needing confirmation. Use the 'request_human_review_tool' (IMPLEMENT THIS TOOL LATER).
*   **Complete:** If the goal (getting a quote) is fully achieved or cannot be completed.

**Output Format:**

You MUST respond with ONLY a JSON object in the following format:

```json
{{
  "thought": "Your detailed step-by-step reasoning for the decision. Explain why you chose this action based on the current state and goal.",
  "tool_name": "Name of the ONE tool to use (e.g., 'quote_initiate_tool', 'ask_customer_tool') OR null if completing.",
  "args": {{ "arg_name": "value", ... }} // Arguments for the chosen tool, matching its schema exactly. Empty {{}} if tool takes no args. Null if completing.
}}
```

**Example 1 (Initiating):**
```json
{{
  "thought": "The conversation just started. The goal is to get a quote. The first step is to initiate the quote using the Mercury API. I have the basic info (name, dob, address) from the first message.",
  "tool_name": "quote_initiate_tool",
  "args": {{ "driver_name": "John Smith", "driver_dob": "1990-01-15", "address_line1": "123 Main St", "city": "Anytown", "state_code": "CA", "zip_code": "90210" }}
}}
```

**Example 2 (Asking Customer):**
```json
{{
  "thought": "Quote initiated successfully (Quote ID: Q-1234). The next step is to add the vehicle. I need the vehicle's VIN, make, model, and year, which are missing from customer_info. I need to ask the customer for this.",
  "tool_name": "ask_customer_tool",
  "args": {{ "missing_fields": ["vehicle_vin", "vehicle_make", "vehicle_model", "vehicle_year"] }}
}}
```

**Example 3 (Handling API Error):**
```json
{{
  "thought": "The last action attempted to call 'add_vehicle_tool' but failed with an error 'Invalid VIN format'. I need to ask the customer to provide the correct VIN.",
  "tool_name": "ask_customer_tool",
  "args": {{ "missing_fields": ["vehicle_vin"] }}
}}
```

Now, analyze the current state and provide your decision in the specified JSON format ONLY.
"""

    def format_planner_prompt(state: 'AgentState', tools: List[Any]) -> str:
        """Formats the prompt for the planner LLM."""
        messages = state.messages or []
        # Simple formatting, consider adding role labels
        messages_str = "\n".join([f"{type(m).__name__}: {m.content}" for m in messages])
        if not messages_str: messages_str = "No messages yet."

        customer_info_str = json.dumps(state.customer_info or {}, indent=2)
        mercury_session_str = json.dumps(state.mercury_session or {}, indent=2)
        last_tool_outputs_str = json.dumps(state.last_tool_outputs or {"status": "N/A"}, indent=2)

        # Format tools for prompt (name and description)
        tools_str = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])

        prompt = PLANNER_PROMPT_TEMPLATE_V2.format(
            goal=state.goal,
            messages_str=messages_str,
            customer_info_str=customer_info_str,
            mercury_session_str=mercury_session_str,
            last_tool_outputs_str=last_tool_outputs_str,
            tools_str=tools_str
        )
        return prompt
    ```

*   **`src/ppa_agent/agent.py`**: (Enable Real LLM Planner)
    ```python
    # src/ppa_agent/agent.py
    # ... (imports) ...
    from .prompts import format_planner_prompt # <-- Import prompt formatter
    from .llm import get_llm_client # <-- Import real LLM getter
    from langchain_core.messages import AIMessage # For parsing LLM response

    # ... (AgentState, ToolNode, all_tools) ...

    logger = logging.getLogger(__name__)

    def planner_node(state: AgentState) -> Dict[str, Any]:
        """
        The core planning node. Uses LLM to decide the next action.
        """
        logger.info("--- Entering Planner Node ---")
        # logger.debug(f"Current State: {state}")

        # --- Clear previous planning artifacts ---
        # Note: This assumes state updates are merged correctly. If nodes return
        # full state, this might be handled differently. Best practice is often
        # for nodes to return only the *changes* to the state.
        update = {
            "planned_tool_inputs": None,
            "last_tool_outputs": state.last_tool_outputs, # Keep last output for planning
            "requires_agency_review": False,
            "awaiting_customer_reply": False,
            "human_feedback": None, # Clear feedback after processing (handled in prompt)
            "agent_scratchpad": None
         }


        # --- Prepare Prompt ---
        # TODO: Handle human_feedback processing within the prompt/logic
        if state.human_feedback:
             logger.warning("Human feedback processing not fully implemented in prompt yet!")
             # Add feedback to prompt context here

        prompt = format_planner_prompt(state, all_tools)
        # logger.debug(f"Planner Prompt:\n------\n{prompt}\n------") # Log prompt for debugging

        # --- Call REAL LLM ---
        llm = get_llm_client() # Get configured LLM
        logger.info("Invoking Planner LLM...")
        llm_decision = None
        try:
            # Ensure the LLM is configured for JSON output (either via client or prompt)
            ai_msg : AIMessage = llm.invoke(prompt)
            response_content = ai_msg.content
            logger.debug(f"Planner LLM Raw Response Content: {response_content}")

            # Attempt to parse JSON, handling potential markdown fences ```json ... ```
            if isinstance(response_content, str):
                content_to_parse = response_content.strip()
                if content_to_parse.startswith("```json"):
                    content_to_parse = content_to_parse[7:]
                if content_to_parse.endswith("```"):
                    content_to_parse = content_to_parse[:-3]
                content_to_parse = content_to_parse.strip()

                llm_decision = json.loads(content_to_parse)
                logger.info(f"Planner LLM Decided Action: {llm_decision}")
            else:
                 # Should not happen if LLM configured correctly, but handle defensively
                 logger.error(f"Planner LLM response content was not a string: {type(response_content)}")
                 raise TypeError("LLM response content is not a string")

        except json.JSONDecodeError as e:
            logger.error(f"Planner LLM did not return valid JSON: {response_content} - Error: {e}", exc_info=True)
            update["agent_scratchpad"] = f"ERROR: LLM output parse error.\nRaw Response:\n{response_content}"
            # Decide error handling - For now, just end the flow by not planning a tool
            llm_decision = {"tool_name": None, "args": None} # Ensure structure exists
        except Exception as e:
            logger.error(f"Error invoking planner LLM: {e}", exc_info=True)
            update["agent_scratchpad"] = f"ERROR: LLM invocation error: {e}"
            llm_decision = {"tool_name": None, "args": None} # Ensure structure exists
        # --- End REAL LLM CALL ---


        # --- Prepare State Update ---
        tool_name = llm_decision.get("tool_name") if llm_decision else None

        if tool_name:
            # Validate if the tool exists? Optional.
            update["planned_tool_inputs"] = {
                "tool_name": tool_name,
                "args": llm_decision.get("args", {})
            }
            logger.info(f"Planner setting planned_tool_inputs: {update['planned_tool_inputs']}")
        else:
            update["planned_tool_inputs"] = None
            logger.info("Planner decided no tool to execute (complete/error).")

        if llm_decision and "thought" in llm_decision:
            current_scratch = update["agent_scratchpad"] or ""
            update["agent_scratchpad"] = f"{current_scratch}\nThought: {llm_decision['thought']}".strip()

        # Update state fields based on the decision
        # We only return the *changes* for LangGraph to merge
        logger.info("--- Exiting Planner Node ---")
        return update


    # --- Graph Definition & Runner Class (remain largely the same structure as M3) ---
    # Use ToolNode(all_tools) which now includes real tools
    execute_tool_node = ToolNode(all_tools)

    def build_agent_graph():
        # ... (same graph structure as M3: planner -> conditional -> executor -> planner) ...
        workflow = StateGraph(AgentState)
        workflow.add_node("planner", planner_node)
        workflow.add_node("executor", execute_tool_node)
        workflow.set_entry_point("planner")
        workflow.add_conditional_edges(
            "planner", should_execute, {"execute": "executor", END: END}
        )
        workflow.add_edge("executor", "planner")
        memory = SqliteSaver.sqlite(db_name=SQLITE_DB_NAME)
        app = workflow.compile(checkpointer=memory)
        logger.info("Agent graph compiled (Milestone 4 - Real Planner/Tools).")
        return app

    class PPAAgentRunner:
         # __init__ and run_turn methods are the same as Milestone 3
         def __init__(self):
            self.app = build_agent_graph()
            # Reset mock step counter not needed anymore

         def run_turn(self, thread_id: str, user_input: Optional[str] = None):
            # ... (same implementation as M3) ...
            logger.info(f"--- Running Agent Turn (Thread: {thread_id}) ---")
            if user_input: logger.info(f"Input: '{user_input}'")
            input_messages = []
            if user_input: input_messages.append(HumanMessage(content=user_input))
            config = {"configurable": {"thread_id": thread_id}}
            final_state_dict = {}
            try:
                for step in self.app.stream({"messages": input_messages}, config=config, stream_mode="values"): # Use values stream mode
                    final_state_dict = step # Keep latest state dict
                    step_name = list(step.keys())[0] # Get node name
                    logger.info(f"Step: {step_name} completed.")
                    # logger.debug(f"State after {step_name}: {step}") # Log intermediate state if needed

                logger.info(f"--- Agent Turn Complete (Thread: {thread_id}) ---")
                final_state_persisted = self.app.get_state(config)
                if final_state_persisted:
                     logger.info(f"Final Persisted Messages Count: {len(final_state_persisted.values['messages'])}")
                     logger.info(f"Last Tool Outputs: {final_state_persisted.values['last_tool_outputs']}")
                     return final_state_persisted.values
                else:
                    logger.error(f"Could not retrieve final state for thread {thread_id}")
                    return {"error": "Failed to retrieve final state"}

            except Exception as e:
                 logger.error(f"Error during agent run (Thread: {thread_id}): {e}", exc_info=True)
                 try:
                     current_state = self.app.get_state(config)
                     return current_state.values if current_state else {"error": str(e)}
                 except Exception:
                      logger.error(f"Could not retrieve state for thread {thread_id} after error.")
                      return {"error": str(e)}

    ```

*   **`scripts/run_agent.py`**: (Test scenarios invoking real tools)
    ```python
    # scripts/run_agent.py
    # ... (imports remain the same) ...
    import time

    if __name__ == "__main__":
        logger.info("--- Milestone 4: Real Planner & Core Tools Test ---")

        if os.path.exists(SQLITE_DB_NAME):
            logger.warning(f"Deleting existing database: {SQLITE_DB_NAME}")
            os.remove(SQLITE_DB_NAME)

        agent_runner = PPAAgentRunner()
        thread_id = "real_tool_test_456"

        print(f"\n--- Turn 1 (Thread: {thread_id}) ---")
        # Provide enough info to potentially trigger quote_initiate
        turn1_input = "Hi, I need car insurance. I'm Jane Doe, DOB 1985-03-10, live at 456 Oak Ln, Maplewood, NJ 07040."
        state1 = agent_runner.run_turn(thread_id, turn1_input)
        print("\nState after Turn 1:")
        print(json.dumps(state1, indent=2, default=str))
        # Expect: Planner likely called quote_initiate_tool. Check last_tool_outputs.

        print(f"\n--- Turn 2 (Thread: {thread_id}) ---")
        # No new input, let agent continue. It should realize info is missing.
        state2 = agent_runner.run_turn(thread_id)
        print("\nState after Turn 2:")
        print(json.dumps(state2, indent=2, default=str))
        # Expect: Planner likely called ask_customer_tool. Check last_tool_outputs.

        print(f"\n--- Turn 3 (Thread: {thread_id}) ---")
        # Provide some of the missing info
        turn3_input = "My car is a 2021 Toyota RAV4."
        state3 = agent_runner.run_turn(thread_id, turn3_input)
        print("\nState after Turn 3:")
        print(json.dumps(state3, indent=2, default=str))
        # Expect: Planner processes reply, updates customer_info. Might ask for more or plan next API.

        print("\nReal planner/tool test complete. Check logs and state outputs.")

    ```

**To Run Milestone 4:**

1.  Ensure API keys are correctly set in `.env`.
2.  Install any new dependencies (`requests`).
3.  Run `python scripts/run_agent.py`.
4.  **Carefully** examine the logs. See the planner's reasoning (if logging prompt/thoughts), the planned tool, the executor output, and how the state evolves. Debug the planner prompt (`prompts.py`) iteratively based on whether the LLM makes logical decisions and correctly formats tool inputs. This milestone requires the most debugging.

---

**Milestone 5: Implement Agency Review HITL**

Introduce the first interrupt for agency review before tool execution.

*   **`src/ppa_agent/state.py`**: (No changes needed, fields `requires_agency_review` and `human_feedback` already exist).
*   **`src/ppa_agent/prompts.py`**: (Update Planner Prompt V3)
    ```python
    # src/ppa_agent/prompts.py
    import json

    # --- Planner Prompt V3 (Adds Human Feedback Handling & Review Decision) ---
    PLANNER_PROMPT_TEMPLATE_V3 = """
You are a PPA Insurance Quoting Agent assistant. Your primary goal is: {goal}.
You interact with the customer via messages and use internal tools to call Mercury Insurance APIs.

**Current Conversation State:**

*   **Conversation History (Newest Last):**
{messages_str}

*   **Current Customer Information Extracted:**
{customer_info_str}

*   **Mercury API Session Context (e.g., Quote ID):**
{mercury_session_str}

*   **Result of the VERY LAST Action Taken:**
{last_tool_outputs_str}

*   **Feedback from Last Human Review (if any):**
{human_feedback_str}

**Your Task:**

Based on the current state, conversation, AND HUMAN FEEDBACK (if provided), decide the single next best action.

1.  **Process Human Feedback:** If feedback exists, **you MUST address it**. Re-evaluate your previous plan based on the comment or use the edited inputs if provided. Do NOT repeat the rejected action without changes.
2.  **Analyze State & Goal:** Review history, current data, goal, and last action result.
3.  **Determine Next Step:** Decide the most logical action (Call API Tool? Ask Customer? Handle Error? Complete?).
4.  **Formulate Plan:** If invoking a tool, generate the exact `tool_name` and `args`.
5.  **Decide on Agency Review:** Determine if this *planned action* needs human review before execution. Set `requires_review` to `true` if:
    *   You plan to use `ask_customer_tool`.
    *   You plan to use `rate_quote_tool`.
    *   You are uncertain about the plan or handling an error.
    *   Human feedback explicitly requested review.
    Otherwise, set `requires_review` to `false`.

**Available Tools:**

{tools_str}

**Output Format:**

You MUST respond with ONLY a JSON object in the following format:

```json
{{
  "thought": "Detailed reasoning, including how you processed human feedback (if any) and why you chose this action and review requirement.",
  "requires_review": true | false, // Does THIS planned action need review?
  "tool_name": "Name of the tool to use OR null.",
  "args": {{ ... }} // Arguments for the tool OR null.
}}
```

**Example (Processing Rejection):**
```json
{{
  "thought": "The human agent rejected the previous plan to call 'add_vehicle_tool' with VIN '123', commenting 'VIN is incorrect'. I must ask the customer for the correct VIN instead.",
  "requires_review": true, // Asking the customer always requires review
  "tool_name": "ask_customer_tool",
  "args": {{ "missing_fields": ["vehicle_vin"] }}
}}
```

Analyze the current state and provide your decision in the specified JSON format ONLY.
"""

    def format_planner_prompt(state: 'AgentState', tools: List[Any]) -> str: # Updated function signature
        """Formats the prompt for the planner LLM (V3 - includes feedback)."""
        # ... (message, customer_info, session formatting same as V2) ...
        messages = state.messages or []
        messages_str = "\n".join([f"{type(m).__name__}: {m.content}" for m in messages])
        if not messages_str: messages_str = "No messages yet."

        customer_info_str = json.dumps(state.customer_info or {}, indent=2)
        mercury_session_str = json.dumps(state.mercury_session or {}, indent=2)
        last_tool_outputs_str = json.dumps(state.last_tool_outputs or {"status": "N/A"}, indent=2)
        # Format human feedback
        human_feedback_str = "None"
        if state.human_feedback:
            human_feedback_str = json.dumps(state.human_feedback, indent=2)

        tools_str = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])

        prompt = PLANNER_PROMPT_TEMPLATE_V3.format(
            goal=state.goal,
            messages_str=messages_str,
            customer_info_str=customer_info_str,
            mercury_session_str=mercury_session_str,
            last_tool_outputs_str=last_tool_outputs_str,
            human_feedback_str=human_feedback_str, # <-- Add feedback
            tools_str=tools_str
        )
        return prompt
    ```

*   **`src/ppa_agent/agent.py`**: (Update Planner Node, Add Graph Logic for Review)
    ```python
    # src/ppa_agent/agent.py
    # ... (imports) ...
    from langgraph.interrupt import InterruptForHumanApproval # <-- Import interrupt
    from .prompts import format_planner_prompt # <-- Use V3 prompt formatter

    # ... (AgentState, tools, execute_tool_node) ...
    logger = logging.getLogger(__name__)

    def planner_node(state: AgentState) -> Dict[str, Any]:
        """
        Planner node (V3). Considers human feedback, decides action & review need.
        """
        logger.info("--- Entering Planner Node (V3) ---")

        # --- Prepare State Update Dictionary ---
        # Start with clearing previous cycle's transient state, keep necessary context
        update = {
            "planned_tool_inputs": None,
            "last_tool_outputs": state.last_tool_outputs, # Planner needs last result
            "requires_agency_review": False, # Default to no review needed
            "awaiting_customer_reply": False, # Default to not waiting
            "human_feedback": None, # ALWAYS clear feedback after processing
            "agent_scratchpad": state.agent_scratchpad # Persist scratchpad unless overwritten
        }

        # --- Prepare & Call LLM ---
        prompt = format_planner_prompt(state, all_tools) # Uses V3 template
        llm = get_llm_client()
        logger.info("Invoking Planner LLM...")
        llm_decision = None
        try:
            ai_msg = llm.invoke(prompt)
            response_content = ai_msg.content
            logger.debug(f"Planner LLM Raw Response Content: {response_content}")
            # ... (JSON parsing logic as in M4) ...
            content_to_parse = response_content.strip()
            if content_to_parse.startswith("```json"): content_to_parse = content_to_parse[7:]
            if content_to_parse.endswith("```"): content_to_parse = content_to_parse[:-3]
            llm_decision = json.loads(content_to_parse.strip())
            logger.info(f"Planner LLM Decided Action: {llm_decision}")

        except Exception as e:
             # ... (Error handling as in M4, ensure llm_decision is dict) ...
             logger.error(f"Planner LLM Error: {e}", exc_info=True)
             llm_decision = {"tool_name": None, "args": None, "requires_review": False, "thought": f"Error processing LLM response: {e}"}


        # --- Update State Based on LLM Decision ---
        tool_name = llm_decision.get("tool_name") if llm_decision else None
        if tool_name:
            update["planned_tool_inputs"] = {
                "tool_name": tool_name,
                "args": llm_decision.get("args", {})
            }
            logger.info(f"Planner setting planned_tool_inputs: {update['planned_tool_inputs']}")
        else:
            update["planned_tool_inputs"] = None # Ensure cleared if no tool planned
            logger.info("Planner decided no tool to execute.")

        # Set review flag based on LLM output
        update["requires_agency_review"] = llm_decision.get("requires_review", False) if llm_decision else False
        logger.info(f"Planner setting requires_agency_review: {update['requires_agency_review']}")

        # Update scratchpad
        if llm_decision and "thought" in llm_decision:
            update["agent_scratchpad"] = llm_decision['thought']


        logger.info("--- Exiting Planner Node ---")
        # Return ONLY the changes to be merged into the state
        return update

    # --- Conditional Edges (Updated for Agency Review) ---

    def check_agency_review(state: AgentState) -> str:
        """Routes based on whether agency review is needed."""
        # This function is called *after* the planner sets the flag
        if state.requires_agency_review:
            logger.info("Routing to: agency_review (Review Required)")
            return "review_needed"
        else:
            logger.info("Routing to: executor (No Review Required)")
            return "execute_tool" # Skip review, go straight to execution

    # --- Graph Definition (Updated) ---
    def build_agent_graph():
        """Builds the LangGraph StateGraph with Agency Review HITL."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("planner", planner_node)
        workflow.add_node("executor", execute_tool_node)
        # Interrupt node doesn't need explicit add_node if using InterruptForHumanApproval directly

        workflow.set_entry_point("planner")

        # Planner decides, then check if review is needed
        workflow.add_conditional_edges(
            "planner",
            check_agency_review, # Check the flag set by the planner
            {
                "review_needed": InterruptForHumanApproval(node_name="agency_review"), # Use built-in interrupt
                "execute_tool": "executor",
            }
        )

        # After execution, always loop back to planner (error or success)
        # The check_execution_result logic for CUSTOMER_INTERRUPT will be added later
        workflow.add_edge("executor", "planner")

        # After interrupt, if human approves (external resume), implicitly loops.
        # LangGraph's default InterruptForHumanApproval resumes execution *after* the interrupt point.
        # So, on approval, it needs to go to the executor. We add an edge from the conceptual interrupt name.
        # If rejected, external system must update state and resume, planner handles feedback.
        workflow.add_edge("agency_review", "executor") # On approval, execute the plan


        # Compile with checkpointer and NEW interrupt config
        memory = SqliteSaver.sqlite(db_name=SQLITE_DB_NAME)
        # Interrupt *before* the named node 'agency_review' is hit
        app = workflow.compile(checkpointer=memory, interrupt_before=["agency_review"])
        logger.info("Agent graph compiled (Milestone 5 - Agency Review HITL Added).")
        return app

    # --- Main Agent Class (No changes needed from M3 runner logic) ---
    class PPAAgentRunner:
        # ... (Same __init__ and run_turn as M3) ...
         def __init__(self):
            self.app = build_agent_graph() # Now compiles graph with interrupts

         def run_turn(self, thread_id: str, user_input: Optional[str] = None):
            # ... (same implementation as M3) ...
             # The runner logic doesn't change, but the *behavior* of app.stream
             # will now pause if it hits the interrupt. The simulation script
             # needs to handle this pause.
            logger.info(f"--- Running Agent Turn (Thread: {thread_id}) ---")
            if user_input: logger.info(f"Input: '{user_input}'")
            input_messages = []
            if user_input: input_messages.append(HumanMessage(content=user_input))
            config = {"configurable": {"thread_id": thread_id}}
            final_state_dict = {}
            try:
                # Stream can now pause
                for step in self.app.stream({"messages": input_messages}, config=config, stream_mode="values"):
                    final_state_dict = step
                    step_name = list(step.keys())[0]
                    logger.info(f"Step: {step_name} completed.")

                    # Check if paused - LangGraph state might indicate next node
                    current_state = self.app.get_state(config)
                    if current_state and current_state.next == ("agency_review",): # Check if next step is the interrupt
                         logger.warning(f"--- Agent Paused for Agency Review (Thread: {thread_id}) ---")
                         # In a real system, return control here. Script will simulate.
                         # Return the state dict so the script can inspect it.
                         return final_state_dict

                logger.info(f"--- Agent Turn Complete (Thread: {thread_id}) ---")
                final_state_persisted = self.app.get_state(config)
                return final_state_persisted.values if final_state_persisted else {"error": "Final state not found"}

            except Exception as e:
                 logger.error(f"Error during agent run (Thread: {thread_id}): {e}", exc_info=True)
                 # ... (Error state retrieval as in M4) ...
                 try:
                     current_state = self.app.get_state(config)
                     return current_state.values if current_state else {"error": str(e)}
                 except Exception: return {"error": str(e)}

         # --- Add methods for HITL interaction (used by script) ---
         def get_current_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
              """Gets the current state of a thread."""
              config = {"configurable": {"thread_id": thread_id}}
              try:
                  state = self.app.get_state(config)
                  return state.values if state else None
              except Exception as e:
                   logger.error(f"Error getting state for thread {thread_id}: {e}")
                   return None

         def update_state_and_resume(self, thread_id: str, update_dict: Dict[str, Any]):
             """Updates the state of a paused thread and resumes execution."""
             config = {"configurable": {"thread_id": thread_id}}
             logger.info(f"--- Resuming Agent (Thread: {thread_id}) with Update ---")
             # logger.debug(f"Update Dict: {update_dict}")

             # Update the state first
             try:
                  self.app.update_state(config, update_dict)
                  logger.info("State updated successfully.")
             except Exception as e:
                   logger.error(f"Failed to update state for thread {thread_id}: {e}")
                   return {"error": f"Failed to update state: {e}"} # Indicate update failure

             # Resume execution by invoking with None input
             final_state_dict = {}
             try:
                # Use stream again to capture subsequent steps until next pause/end
                for step in self.app.stream(None, config=config, stream_mode="values"):
                    final_state_dict = step
                    step_name = list(step.keys())[0]
                    logger.info(f"Step after resume: {step_name} completed.")

                    # Check again if paused
                    current_state = self.app.get_state(config)
                    if current_state and current_state.next and current_state.next[0].startswith("agency_review"): # Check if paused *again*
                         logger.warning(f"--- Agent Paused AGAIN for Agency Review (Thread: {thread_id}) ---")
                         return final_state_dict # Return paused state

                logger.info(f"--- Agent Resume Complete (Thread: {thread_id}) ---")
                final_state_persisted = self.app.get_state(config)
                return final_state_persisted.values if final_state_persisted else {"error": "Final state not found post-resume"}

             except Exception as e:
                 logger.error(f"Error during agent resume (Thread: {thread_id}): {e}", exc_info=True)
                 try: # Try to get state even after resume error
                     current_state = self.app.get_state(config)
                     return current_state.values if current_state else {"error": str(e)}
                 except Exception: return {"error": str(e)}
    ```

*   **`scripts/run_agent.py`**: (Updated for HITL Simulation)
    ```python
    # scripts/run_agent.py
    import sys
    import os
    import logging
    import json
    import time

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from src.ppa_agent.agent import PPAAgentRunner
    from src.ppa_agent.state import AgentState # For type hint
    from src.ppa_agent.config import logger, SQLITE_DB_NAME

    def print_state_summary(state_dict: dict, title: str = "State Summary"):
        """Prints key parts of the state."""
        print(f"\n--- {title} ---")
        if not state_dict or isinstance(state_dict.get("error"), str):
            print(f"Error or empty state: {state_dict}")
            return

        print(f"Thread ID: {state_dict.get('thread_id')}")
        print(f"Messages Count: {len(state_dict.get('messages', []))}")
        # print(f"Last Message: {state_dict.get('messages', [])[-1] if state_dict.get('messages') else 'N/A'}")
        print(f"Customer Info: {json.dumps(state_dict.get('customer_info'), indent=2)}")
        print(f"Planned Tool: {state_dict.get('planned_tool_inputs')}")
        print(f"Last Output: {state_dict.get('last_tool_outputs')}")
        print(f"Requires Review: {state_dict.get('requires_agency_review')}")
        print(f"Human Feedback: {state_dict.get('human_feedback')}")
        print(f"Awaiting Customer: {state_dict.get('awaiting_customer_reply')}")
        # print(f"Scratchpad: {state_dict.get('agent_scratchpad')}")
        print("------")

    def simulate_agency_review(agent_runner: PPAAgentRunner, thread_id: str, current_state: dict) -> bool:
        """Simulates the human review process. Returns True if approved."""
        print("\n--- !!! AGENCY REVIEW REQUIRED !!! ---")
        print(f"Reviewing plan for Thread: {thread_id}")
        planned_action = current_state.get('planned_tool_inputs')
        print(f"Planned Action: {planned_action}")
        # print(f"Planner Thought: {current_state.get('agent_scratchpad')}") # Show thought process

        # Simple simulation: Auto-approve 'ask_customer', reject others once
        if not hasattr(simulate_agency_review, "rejected_once"):
             simulate_agency_review.rejected_once = False

        tool_name = planned_action.get("tool_name") if planned_action else None

        if tool_name == "ask_customer_tool":
            print("Decision: Auto-approving 'ask_customer_tool'.")
            feedback = {"approved": True}
            update = {"human_feedback": feedback} # Provide feedback object
            agent_runner.update_state_and_resume(thread_id, update)
            return True
        elif not simulate_agency_review.rejected_once:
            print("Decision: Simulating REJECTION (first time).")
            feedback = {"approved": False, "comment": "Please double-check the inputs before executing this API."}
            update = {
                "human_feedback": feedback,
                "planned_tool_inputs": None # Clear the rejected plan
            }
            simulate_agency_review.rejected_once = True # Only reject once per run
            agent_runner.update_state_and_resume(thread_id, update)
            return False # Indicate rejection
        else:
            print("Decision: Auto-approving other tool (second time).")
            feedback = {"approved": True}
            update = {"human_feedback": feedback}
            agent_runner.update_state_and_resume(thread_id, update)
            return True


    if __name__ == "__main__":
        logger.info("--- Milestone 5: Agency Review HITL Test ---")

        if os.path.exists(SQLITE_DB_NAME):
            logger.warning(f"Deleting existing database: {SQLITE_DB_NAME}")
            os.remove(SQLITE_DB_NAME)

        agent_runner = PPAAgentRunner()
        thread_id = "hitl_review_test_789"

        # --- Turn 1: Initial email, should trigger quote_initiate plan & review ---
        print(f"\n--- Turn 1 (Thread: {thread_id}) ---")
        turn1_input = "Hi, I need car insurance. I'm Jane Doe, DOB 1985-03-10, live at 456 Oak Ln, Maplewood, NJ 07040."
        current_state = agent_runner.run_turn(thread_id, turn1_input)
        print_state_summary(current_state, "State After Turn 1 Run")

        # Check if paused for review
        if current_state and agent_runner.get_current_state(thread_id).get('requires_agency_review'):
            # Simulate review - this will reject the first non-ask_customer plan
            approved = simulate_agency_review(agent_runner, thread_id, current_state)
            if not approved:
                print("\n--- Turn 1 Continued (After Rejection & Replanning) ---")
                # The agent resumed and replanned automatically inside simulate_agency_review
                # We need to get the state *after* that resume finished or paused again
                current_state = agent_runner.get_current_state(thread_id)
                print_state_summary(current_state, "State After Rejection & Replanning")

                # It might pause *again* for review of the *new* plan
                if current_state and agent_runner.get_current_state(thread_id).get('requires_agency_review'):
                    print("\n--- Simulating Second Review ---")
                    # This time it should approve (based on simulation logic)
                    approved_2 = simulate_agency_review(agent_runner, thread_id, current_state)
                    if approved_2:
                         print("\n--- Turn 1 Continued (After Second Approval) ---")
                         # Agent resumed and likely executed, get final state for the turn
                         current_state = agent_runner.get_current_state(thread_id)
                         print_state_summary(current_state, "State After Final Approval & Execution")
                    else:
                         print("Error: Second review was unexpectedly rejected.")
            else:
                 # It was approved the first time (likely ask_customer)
                 print("\n--- Turn 1 Continued (After First Approval) ---")
                 current_state = agent_runner.get_current_state(thread_id)
                 print_state_summary(current_state, "State After First Approval & Execution")
        else:
            print("Agent did not pause for review as expected.")


        print("\n--- Turn 2: No input, agent continues ---")
        # Agent should now plan based on the outcome of Turn 1 (e.g., plan ask_customer)
        current_state_t2 = agent_runner.run_turn(thread_id)
        print_state_summary(current_state_t2, "State After Turn 2 Run")

        if current_state_t2 and agent_runner.get_current_state(thread_id).get('requires_agency_review'):
             print("\n--- Simulating Turn 2 Review ---")
             # This should approve ask_customer
             approved_t2 = simulate_agency_review(agent_runner, thread_id, current_state_t2)
             if approved_t2:
                  print("\n--- Turn 2 Continued (After Approval) ---")
                  current_state_t2_final = agent_runner.get_current_state(thread_id)
                  print_state_summary(current_state_t2_final, "State After T2 Approval & Execution")
                  # Now check if it's paused waiting for customer (Milestone 6 check)
                  # if current_state_t2_final and agent_runner.get_current_state(thread_id).get('awaiting_customer_reply'):
                  #    print("\n*** Agent is now paused waiting for customer reply. ***")
             else:
                  print("Error: Turn 2 review was unexpectedly rejected.")
        else:
             print("Agent did not pause for review in Turn 2 (or finished).")


        print("\nMilestone 5 Test Complete. Check logs for review simulation.")
    ```

**To Run Milestone 5:**

1.  Run `python scripts/run_agent.py`.
2.  Observe the logs carefully. You should see:
    *   The planner setting `requires_agency_review: True`.
    *   The `check_agency_review` function routing to the interrupt.
    *   The `run_turn` method detecting the pause.
    *   The `simulate_agency_review` function logging its decision (reject first, then approve).
    *   The `update_state_and_resume` call being made.
    *   The planner node running again after rejection, processing the `human_feedback`.
    *   The planner making a *new* plan.
    *   The second review potentially happening and being approved.
    *   The executor running the *approved* plan.

This confirms the Agency Review HITL loop is functional.

---

**(Continue to Milestone 6 in the next response due to length constraints)**