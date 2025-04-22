# PPA New Business AI Agent (Agentic v2)

An AI Agent using LangGraph to dynamically manage and automate the initial stages of the Personal Private Auto (PPA) insurance new business quoting process.

## Overview

This project implements an agentic workflow that:
- Processes multi-turn customer conversations (e.g., from emails) for new PPA quotes.
- Uses a central "Planner" agent to dynamically decide the next action.
- Leverages a set of "Tools" for tasks like interacting with external APIs (e.g., Mercury Insurance) and communicating with the customer.
- Incorporates Human-in-the-Loop (HITL) review points for critical steps.
- Persists conversation state across turns and interruptions using `SqliteSaver`.

This v2 represents a shift from a predefined workflow to a more flexible, planner-driven approach.

## Architecture

The agent follows a planner-centric design:

1.  **State (`AgentState`):** Defined using Pydantic, storing conversation history (`messages`), extracted `customer_info`, API session data (`mercury_session`), planner decisions (`planned_tool_inputs`), tool results (`last_tool_outputs`), and HITL flags/feedback. Persisted using `langgraph.checkpoint.sqlite.SqliteSaver`.
2.  **Planner Node:** The core LLM-powered node that analyzes the current `AgentState`, including history, feedback, and tool results, to decide the next action (which tool to call, whether to request review, or wait for the customer).
3.  **Tools:** Functions decorated with `@tool` that perform specific actions (e.g., `quote_initiate_tool`, `add_driver_tool`, `ask_customer_tool`).
4.  **Executor Node:** Executes the tool chosen by the Planner.
5.  **Conditional Edges:** Control the flow based on planner decisions (e.g., routing to HITL or executor).
6.  **HITL Interrupts:**
    *   `Agency Review`: Pauses *before* tool execution for optional human approval/rejection of the planned action.
    *   `Wait for Customer`: Pauses *after* deciding to ask the customer for information, awaiting their reply.

For a detailed diagram and explanation, see the [System Design Document](docs/agentic_v2/System%20Design:%20Agentic%20PPA%20Insurance%20Quoting%20Workflow%20(v2.0%20-%20Final).md).

## Key Features

- Dynamic workflow orchestration via LangGraph Planner.
- Modular tools for interacting with APIs and customers.
- Multi-turn conversation memory and state persistence (`SqliteSaver`).
- Two distinct Human-in-the-Loop integration points.
- Pydantic-based state management for clarity and validation.

## Project Structure

```
.
├── docs/
│   ├── agentic_v2/             # Agentic v2 specific design/milestones
│   │   ├── System Design: Agentic PPA Insurance Quoting Workflow (v2.0 - Final).md
│   │   ├── Milestones.md
│   │   └── ...
│   ├── langserve_testing.md    # Guide for testing the API server
│   ├── google_genai_migration.md # Migration guide for google-genai SDK
│   └── ...
├── src/
│   └── ppa_agentic_v2/         # Main agentic v2 implementation
│       ├── __init__.py
│       ├── agent.py            # Graph definition, nodes, runner
│       ├── server.py           # FastAPI/LangServe server
│       ├── state.py            # Pydantic AgentState definition
│       ├── tools.py            # Tool definitions
│       ├── prompts.py          # Planner prompt templates
│       ├── llm.py              # LLM client setup
│       ├── config.py           # Configuration loading
│       └── run_agent.py        # Script for local testing
├── tests/
│   └── ...
├── .env                          # Environment variables (local, not in git)
├── .gitignore
├── pyproject.toml                # Poetry dependencies
├── poetry.lock
└── README.md
```

## Setup

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/) (For Python environment management)
- [Poetry](https://python-poetry.org/) (Used *within* the Conda environment for Python package management)
- Python 3.11+
- Google Gemini API key or OpenAI API key

### Environment Setup

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  Create and activate a Conda environment (optional but recommended):
    ```bash
    conda create -n ppa-agent python=3.11 -y
    conda activate ppa-agent
    ```

3.  Install dependencies using Poetry (within the activated Conda environment):
    ```bash
    poetry install
    ```
    *(This installs dependencies like `langgraph`, `langchain-core`, `langchain-google-genai`, `pydantic`, `fastapi`, `uvicorn`, `aiosqlite`, etc., as defined in `pyproject.toml`)*

4.  Create a `.env` file in the project root (copy `.env.example` if it exists):
    ```dotenv
    # .env
    # Provide API keys for your chosen LLM provider(s)
    # GOOGLE_API_KEY="your_google_api_key_here" # Preferred
    # OPENAI_API_KEY="your_openai_api_key_here"

    # Optional: Specify models if different from defaults
    # GOOGLE_MODEL_NAME="gemini-1.5-pro-latest"
    # OPENAI_MODEL_NAME="gpt-4-turbo-preview"
    ```

## Running the Agent

There are two main ways to run the agent:

### 1. Local Testing Script

Use the script provided for simulating conversations and testing the agent logic locally, including HITL steps.

```bash
# Ensure conda env is active
poetry run python src/ppa_agentic_v2/run_agent.py
```
*(Note: You might need to modify `run_agent.py` to test specific scenarios.)*

### 2. Web Server (FastAPI / LangServe)

Run the agent as a web service using FastAPI and LangServe, exposing API endpoints.

```bash
# Ensure conda env is active and you are in the project root
# Use the full path to the conda env python
/path/to/conda/envs/ppa-agent/bin/python -m uvicorn src.ppa_agentic_v2.server:app_fastapi --host 127.0.0.1 --port 8000 --reload
```
Replace `/path/to/conda/envs/ppa-agent/bin/python` with the actual path to the python executable in your activated conda environment.

Refer to [docs/langserve_testing.md](docs/langserve_testing.md) for details on how to interact with the server endpoints using `curl`.

## Testing

Run the automated test suite using pytest:

```bash
# Ensure conda env is active
poetry run pytest
```

## Documentation

- **System Design (v2):** [docs/agentic_v2/System Design: Agentic PPA Insurance Quoting Workflow (v2.0 - Final).md](docs/agentic_v2/System%20Design:%20Agentic%20PPA%20Insurance%20Quoting%20Workflow%20(v2.0%20-%20Final).md)
- **Development Milestones (v2):** [docs/agentic_v2/Milestones.md](docs/agentic_v2/Milestones.md)
- **API Server Testing:** [docs/langserve_testing.md](docs/langserve_testing.md)
- **Google Genai SDK Migration:** [docs/google_genai_migration.md](docs/google_genai_migration.md)

## Model Support

This project supports multiple LLM providers:

- **Gemini** (Primary): Using Google's `google-genai` SDK (e.g., Gemini 1.5 Pro).
- **OpenAI** (Alternative): Using GPT models via the OpenAI API.

> **Note**: The project uses the newer `google-genai` SDK.
> See the [Google Genai Migration Guide](docs/google_genai_migration.md).

## Contributing

1.  Create a feature branch.
2.  Make your changes.
3.  Ensure tests pass.
4.  Submit a pull request.

## License

(To be determined)
