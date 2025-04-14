# Makefile for PPA Agent Development

.PHONY: help setup install test test_one visualize clean

# Variables
PYTHON := poetry run python
POETRY := poetry
PYTEST := poetry run pytest
VISUALIZE_SCRIPT := src/visualize_agent.py
TEST_OUTPUT_DIR := test_outputs

# Find the latest state JSON file in the test output directory
# Use shell commands: list by time, take first, handle case where dir is empty
LATEST_STATE_JSON := $(shell test -d $(TEST_OUTPUT_DIR) && ls -t $(TEST_OUTPUT_DIR)/*_state.json 2>/dev/null | head -n 1 || echo "test_outputs/default_state.json")
# Derive default HTML output name from the JSON name
DEFAULT_OUTPUT_HTML := $(patsubst %_state.json,%_result.html,$(LATEST_STATE_JSON))


help:
	@echo "Available commands:"
	@echo "  make setup          - Install dependencies using Poetry."
	@echo "  make install        - Alias for setup."
	@echo "  make test           - Run all tests using pytest."
	@echo "  make test_one       - Run a specific test function/module (e.g., make test_one TEST_NAME=test_complete_workflow)."
	@echo "  make visualize      - Generate visualization HTML from a state JSON file."
	@echo "                      - Args: INPUT_JSON (default: latest in $(TEST_OUTPUT_DIR)), OUTPUT_HTML (default: derived from INPUT_JSON)."
	@echo "                      - Example: make visualize INPUT_JSON=test_outputs/my_state.json"
	@echo "  make clean          - Remove generated files (__pycache__, test_outputs)."

# Environment Setup
setup: install
install:
	@echo ">>> Installing dependencies using Poetry..."
	@$(POETRY) install
	@echo ">>> Setup complete."
	@if [ ! -f .env ]; then \
		echo ">>> NOTE: .env file not found. Create one with OPENAI_API_KEY and/or GEMINI_API_KEY if needed for integration tests."; \
	fi

# Testing
test:
	@echo ">>> Running all tests..."
	@$(PYTEST) -v

test_one:
ifndef TEST_NAME
	@echo "Usage: make test_one TEST_NAME=<test_name_pattern>"
	@exit 1
endif
	@echo ">>> Running tests matching pattern: $(TEST_NAME)..."
	@$(PYTEST) -k "$(TEST_NAME)" -v -s --log-cli-level=INFO

# Visualization
# Allow overriding defaults via command line arguments
INPUT_JSON ?= $(LATEST_STATE_JSON)
OUTPUT_HTML ?= $(DEFAULT_OUTPUT_HTML)
visualize:
	@echo ">>> Generating visualization..."
	@echo "    Input JSON: $(INPUT_JSON)"
	@echo "    Output HTML: $(OUTPUT_HTML)"
	@if [ ! -f $(INPUT_JSON) ]; then \
		echo ">>> ERROR: Input JSON file '$(INPUT_JSON)' not found. Run tests first or specify INPUT_JSON."; \
		exit 1; \
	fi
	@$(PYTHON) $(VISUALIZE_SCRIPT) --input "$(INPUT_JSON)" --output "$(OUTPUT_HTML)"
	@echo ">>> Visualization saved to $(OUTPUT_HTML)"

# Cleanup
clean:
	@echo ">>> Cleaning generated files..."
	@rm -rf $(TEST_OUTPUT_DIR)
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.py[co]" -delete
	@echo ">>> Cleanup complete." 