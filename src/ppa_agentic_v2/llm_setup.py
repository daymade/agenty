# src/ppa_agentic_v2/llm_setup.py

import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool
from .config import (
    GOOGLE_API_KEY, OPENAI_API_KEY,
    DEFAULT_LLM_PROVIDER, GOOGLE_MODEL_NAME, OPENAI_MODEL_NAME, logger
)
from .tools import all_tools

# --- Instantiate LLM --- #

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
