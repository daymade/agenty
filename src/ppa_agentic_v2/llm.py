# src/ppa_agentic_v2/llm.py
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