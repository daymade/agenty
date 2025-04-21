# visualize_graph.py
import os
import logging
from src.ppa_agentic_v2.graph import build_agent_graph

# Configure basic logging if needed by graph build
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_graph_visualization(output_filename="graph_structure.png"):
    """Builds the graph and saves its structure as a PNG image."""
    try:
        # Build the graph application
        # Note: This assumes build_agent_graph doesn't need specific state/config to build
        app = build_agent_graph()
        logger.info("Graph built successfully.")

        # Get the underlying graph object
        graph = app.get_graph()

        # Generate the image
        logger.info(f"Generating graph image: {output_filename}")
        # Ensure the directory exists (save in the project root)
        output_path = os.path.abspath(output_filename)
        image_bytes = graph.draw_mermaid_png()

        if image_bytes:
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            logger.info(f"Graph visualization saved to: {output_path}")
        else:
            logger.error("Failed to generate graph image (draw_mermaid_png returned None).")

    except ImportError as ie:
        logger.error(f"Import Error: {ie}. Make sure all dependencies, including pygraphviz and its system libraries (graphviz), are installed correctly.")
    except Exception as e:
        logger.error(f"An error occurred during graph visualization: {e}", exc_info=True)

if __name__ == "__main__":
    generate_graph_visualization()
