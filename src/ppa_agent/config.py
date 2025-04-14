"""Configuration constants for the PPA Agent."""

import os

# --- LLM Model Configuration ---

# Use the preview model as suggested by API errors for better quota
# gemini-2.5-pro-exp-03-25
# gemini-2.5-pro-preview-03-25
# https://ai.google.dev/gemini-api/docs/models#gemini-2.5-pro-preview-03-25
GEMINI_MODEL_NAME = "gemini-2.5-pro-preview-03-25"

# Default OpenAI model (can be overridden by environment variable)
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")

# --- PPA Requirements ---
# (Moved here from state.py for central config)
PPA_REQUIREMENTS = [
    "driver_name",
    "driver_age",
    "vehicle_make",
    "vehicle_model",
    "vehicle_year",
    "address",
]

# Add other configurations here as needed, e.g.:
# LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
