"""
Visualization script for the PPA Agent workflow using LangGraph UI.

This script creates a visualization of the agent's workflow, showing the states
and transitions to help understand and debug the agent.

Usage:
    python -m src.visualize_agent
"""

import os
from pathlib import Path
import asyncio
import logging
from dotenv import load_dotenv

# Import the agent
from ppa_agent.agent import (
    build_agent,
    AgentState,
    init_state,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


async def main():
    """Run the visualization for the agent."""
    logger.info("Building agent for visualization...")
    
    # Build the agent
    agent = build_agent()
    
    # Export to LangGraph UI
    logger.info("Exporting agent to LangGraph UI...")
    
    # Make sure a visualizations directory exists
    vis_dir = Path("visualizations")
    vis_dir.mkdir(exist_ok=True)
    
    # Create the UI visualization
    await agent.aget_graph().aexport_html(
        vis_dir / "agent_workflow.html",
        title="PPA Agent Workflow",
        include_weights=True,
        numbered_steps=True,
        show_prompt=True,
    )
    
    # Display a trace of a sample workflow
    # Create a simple state 
    state = init_state()
    
    # Add a sample email for visualization
    state["customer_email"] = """
    Hello, I need car insurance for my vehicle.
    I'm John Smith, 35 years old, living at 123 Main St, Seattle, WA.
    
    I drive a 2020 Toyota Camry. I've been driving for 15 years with no accidents.
    
    Looking forward to your quote.
    Best regards,
    John
    """
    
    # Run the agent with tracing
    logger.info("Running agent with tracing...")
    config = {"recursion_limit": 25}
    result = await agent.ainvoke(state, config=config)
    
    # Export the trace to HTML
    logger.info("Exporting execution trace...")
    await agent.atrace(state, config=config).aexport_html(
        vis_dir / "agent_trace.html", 
        title="PPA Agent Execution Trace"
    )
    
    logger.info(f"Visualization files created in {vis_dir.absolute()}")
    logger.info("Open the following files in your browser:")
    logger.info(f"  - {vis_dir.absolute() / 'agent_workflow.html'} (Agent Workflow)")
    logger.info(f"  - {vis_dir.absolute() / 'agent_trace.html'} (Sample Execution Trace)")


if __name__ == "__main__":
    asyncio.run(main()) 