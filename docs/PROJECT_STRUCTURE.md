# Project Structure

```
.
├── .env.example                    # Example environment variables file (contains Gemini API key setup)
├── .gitignore                      # Git ignore rules for the project
├── .pre-commit-config.yaml         # Pre-commit hook configurations for code quality
├── docs/                           # Documentation directory
│   ├── system_design.md           # System architecture and design documentation
│   └── todo.md                    # Project implementation roadmap
├── environment.yml                 # Conda environment specification
├── pyproject.toml                  # Python project configuration and dependencies
├── README.md                       # Project overview and setup instructions
├── src/                           # Source code directory
│   └── ppa_agent/                # Main package directory
│       ├── __init__.py           # Package initialization
│       ├── agent.py              # Core agent implementation
│       └── llm_providers.py      # LLM provider integrations
└── tests/                         # Test directory
    ├── __init__.py               # Test package initialization
    ├── test_agent.py             # Unit tests for agent functionality
    └── test_integration.py       # Integration tests

```

## Directory Details

### Root Directory
- Contains configuration files and primary project documentation
- `.env.example`: Template for environment variables including API keys
- `environment.yml`: Conda environment specification for dependency management
- `pyproject.toml`: Project metadata and build system requirements

### docs/
- Contains project documentation
- `system_design.md`: Detailed technical architecture and design decisions
- `todo.md`: Implementation tasks and project roadmap

### src/ppa_agent/
- Main source code package
- `agent.py`: Core implementation of the PPA insurance quoting agent
- `llm_providers.py`: Integrations with language models and AI providers

### tests/
- Test suite for the project
- Includes both unit tests and integration tests
- Follows standard Python testing practices
