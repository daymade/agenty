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