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

def print_state(state_dict, label):
    """Helper function to print state in a readable format"""
    print(f"\n--- {label} ---")
    
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

def main():
    logger.info("--- Milestone 4: Real Planner with Tools Test ---")
    
    # Initialize Agent Runner (ensure it's defined here)
    agent_runner = PPAAgentRunner(use_sqlite_persistence=True)
    thread_id = f"test-thread-{uuid.uuid4()}" # Create a unique thread ID for the test
    logger.info(f"Initialized PPAAgentRunner for thread: {thread_id} (Persistence: True)")
    
    # Test customer initial query with driver and vehicle information
    initial_email = """
    Hello,
    I need a quote for my car insurance.
    My name is Alice Johnson, I'm 35 years old and live at 123 Main St, San Francisco, CA 94105. My DOB is 1990-01-15.
    My car is a 2022 Toyota Camry, VIN: 12345ABCDEF67890.
    Thanks,
    Alice
    """
    
    logger.info(f"Running first turn with customer information for thread {thread_id}...")
    final_state_dict_1 = agent_runner.run_turn(thread_id=thread_id, user_input=initial_email)
    print_state(final_state_dict_1, "Final State After Turn 1 (Initial Information)")
    
    # Test follow-up question
    follow_up_question = "What deductible options do you offer?"
    logger.info(f"Running second turn with follow-up question for thread {thread_id}...")
    final_state_dict_2 = agent_runner.run_turn(thread_id=thread_id, user_input=follow_up_question)
    print_state(final_state_dict_2, "Final State After Turn 2 (Follow-up Question)")

    # Test providing additional information
    additional_info = "I also have another driver, my husband Bob, who is 38 years old."
    logger.info(f"Running third turn with additional information for thread {thread_id}...")
    final_state_dict_3 = agent_runner.run_turn(thread_id=thread_id, user_input=additional_info)
    print_state(final_state_dict_3, "Final State After Turn 3 (Additional Driver Info)")
    logger.info("\n------")
    
    # Return the runner so it can be closed
    return agent_runner 

if __name__ == "__main__":
    runner_instance = None # Initialize to None
    try:
        # Execute the main logic and get the runner instance
        runner_instance = main()
    except Exception as e:
        logger.error(f"An error occurred during the test run: {e}", exc_info=True)
    finally:
        # Ensure resources are cleaned up even if errors occur
        # Check if runner_instance was successfully created before closing
        if runner_instance and hasattr(runner_instance, 'close'):
            logger.info("Closing agent runner resources...")
            runner_instance.close() 
        else:
            logger.info("Agent runner not initialized or already closed.")