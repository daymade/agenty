# src/ppa_agentic_v2/run_agent.py
import sys
import os
import logging
import uuid
import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Ensure the src directory is in the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.ppa_agentic_v2.agent import PPAAgentRunner
from src.ppa_agentic_v2.config import logger # Use configured logger

# Custom JSON encoder to handle LangChain message objects
class MessageEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (HumanMessage, AIMessage, ToolMessage)):
            return {
                "type": obj.__class__.__name__,
                "content": obj.content,
                "id": getattr(obj, "id", None)
            }
        return super().default(obj)

def print_state(state_dict, title):
    """Helper function to print state in a readable format"""
    print(f"\n--- {title} ---")
    
    # First try to convert messages to a readable format
    if "messages" in state_dict:
        formatted_messages = []
        for msg in state_dict["messages"]:
            if isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
                formatted_messages.append({
                    "type": msg.__class__.__name__,
                    "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                })
            else:
                formatted_messages.append(str(msg))
        
        # Create a copy of the state dict with formatted messages
        printable_state = state_dict.copy()
        printable_state["messages"] = formatted_messages
        
        # Use custom encoder for any remaining message objects
        print(json.dumps(printable_state, indent=2, cls=MessageEncoder))
    else:
        # Fall back to regular printing with custom encoder
        print(json.dumps(state_dict, indent=2, cls=MessageEncoder))
    
    print("------")

if __name__ == "__main__":
    logger.info("--- Milestone 3: Persistent Agent Turn Test ---")
    initial_email = "Hello, I need a PPA quote. My name is Alice."
    thread_id = f"test-thread-{uuid.uuid4()}" # Create a unique thread ID for the test

    agent_runner = PPAAgentRunner()

    logger.info(f"Running first turn for thread {thread_id}...")
    final_state_dict = agent_runner.run_turn(thread_id=thread_id, user_input=initial_email)

    print_state(final_state_dict, "Final State (Dictionary) After Turn 1")

    # Example of running a second turn in the same thread
    print(f"\n--- Running second turn for thread {thread_id} (no new input) ---")
    final_state_dict_2 = agent_runner.run_turn(thread_id=thread_id)
    print_state(final_state_dict_2, "Final State (Dictionary) After Turn 2")