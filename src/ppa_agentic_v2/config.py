# src/ppa_agentic_v2/config.py
import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Preferred for google-genai

# --- Model Selection ---
DEFAULT_LLM_PROVIDER = "openai"
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME", "gemini-2.5-pro-preview-03-25")

# --- Logging ---
# Change default to DEBUG for more verbose logging during development
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
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