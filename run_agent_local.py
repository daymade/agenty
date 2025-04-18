# run_agent_local.py
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Load environment variables from .env file
load_dotenv()

# Import the agent graph builder
from src.ppa_agentic_v2.agent import build_agent_graph

async def main():
    """Builds the agent graph and invokes it with sample input."""
    print("Building agent graph...")
    app = build_agent_graph()
    print("Agent graph built.")

    # Define sample input (replace with your desired test message)
    sample_input = {
        "messages": [
            HumanMessage(content="Hi, I need an auto insurance quote. My name is Bob, I live in California, and I drive a 2020 Toyota Camry.")
        ]
    }

    print(f"\nInvoking agent with input:\n{sample_input}\n")

    # Invoke the graph asynchronously using astream_log
    print("\n--- Streaming Agent Log ---")
    async for log_patch in app.astream_log(sample_input, include_types=["llm", "tool", "chain"]):
        print(log_patch, flush=True) # Print each log patch as it arrives

    # Note: astream_log consumes the stream. If you need the final state,
    # you might need to run ainvoke separately or reconstruct it from the log.
    # For simple dev viewing, just printing the stream is often enough.
    print("\n--- Agent Log Stream Complete ---")

    # Example: Accessing the last message more robustly
    # if isinstance(final_state, dict) and 'messages' in final_state and final_state['messages']:
    #     print("\nLast message from agent:")
    #     print(repr(final_state['messages'][-1])) # Use repr for detailed view
    # elif isinstance(final_state, AgentState) and final_state.messages: # If state is returned as AgentState object
    #      print("\nLast message from agent:")
    #      print(repr(final_state.messages[-1])) # Use repr for detailed view

if __name__ == "__main__":
    # Setup and run the async main function
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}")
