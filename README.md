# PPA New Business AI Agent

An AI Agent using LangGraph to automate the initial stages of Personal Private Auto (PPA) insurance new business quoting process within Mercury Insurance.

## Overview

This project implements a Proof of Concept (PoC) AI Agent that:
- Processes customer email inquiries for new PPA insurance quotes
- Extracts relevant information
- Manages the quote request workflow
- Integrates with human agents for review and approval

For detailed system design, see [System Design Document](docs/system_design.md).

## Setup

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/) for environment management
- Python 3.11+
- OpenAI API key

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ppa-agent
```

2. Create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate ppa-agent
```

3. Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

```
.
├── docs/
│   ├── system_design.md   # Detailed system design documentation
│   └── todo.md           # Implementation plan
├── src/
│   ├── __init__.py
│   └── ppa_agent.py      # Main agent implementation
├── tests/
│   └── __init__.py
├── .env                  # Environment variables (not in git)
├── .gitignore
├── environment.yml       # Conda environment specification
└── README.md
```

## Development

Follow the implementation plan in `docs/todo.md` for step-by-step development guidance.

## Testing

(To be implemented)

## Contributing

1. Create a feature branch
2. Make your changes
3. Submit a pull request

## License

(To be determined) 