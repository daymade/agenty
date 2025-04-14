"""Integration tests for the PPA Agent using real APIs."""

import os

import pytest
from dotenv import load_dotenv

from src.ppa_agent.agent import LLMProvider, PPAAgent

# Load environment variables from .env file
load_dotenv()


@pytest.fixture
def openai_agent():
    """Create an OpenAI agent instance with real API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in environment")
    return PPAAgent(provider=LLMProvider.OPENAI, model="gpt-3.5-turbo", api_key=api_key)


@pytest.fixture
def gemini_agent():
    """Create a Gemini agent instance with real API key."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not found in environment")
    return PPAAgent(provider=LLMProvider.GEMINI, model="gemini-2.5-pro-exp-03-25", api_key=api_key)


def test_complete_workflow_openai(openai_agent):
    """Test complete workflow with OpenAI."""
    email = """
    Hi, I'm interested in getting auto insurance for my new car.
    My name is John Smith, I'm 22 years old, and I'm a full-time student
    with a 3.8 GPA. I live at 123 Main St, San Francisco, CA.

    I'm looking to insure my 2022 Tesla Model 3, which has all the latest
    safety features. I also have a clean driving record.

    I'm also interested in bundling this with renters insurance if possible.

    Thanks,
    John
    """

    state = openai_agent.process_email(email)

    # Verify basic state
    assert state["customer_email"] == email
    assert state["customer_info"]["driver_name"] == "John Smith"
    assert state["customer_info"]["driver_age"] == "22"
    assert state["customer_info"]["vehicle_make"] == "Tesla"
    assert state["customer_info"]["vehicle_model"] == "Model 3"
    assert state["customer_info"]["vehicle_year"] == "2022"
    assert state["customer_info"]["address"] == "123 Main St, San Francisco, CA"

    # Verify discount proof request
    discount_request = next(
        (msg for msg in state["messages"] if msg["type"] == "discount_proof_request"), None
    )
    assert discount_request is not None
    assert "student" in discount_request["content"].lower()

    # Verify quote generation
    quote_msg = next((msg for msg in state["messages"] if msg["type"] == "quote"), None)
    assert quote_msg is not None
    assert "Quote ID:" in quote_msg["content"]
    assert state["quote_data"] is not None
    assert state["quote_ready"] is True

    # Verify agency review
    review_msg = next((msg for msg in state["messages"] if msg["type"] == "review_summary"), None)
    assert review_msg is not None
    assert state["review_summary"] is not None
    assert "priority" in state["review_summary"]


def test_complete_workflow_gemini(gemini_agent):
    """Test complete workflow with Gemini."""
    email = """
    Hello, I need car insurance for my vehicle.
    I'm Sarah Johnson, 35 years old, living at 456 Oak Ave, Seattle, WA.

    I drive a 2021 Honda CR-V with advanced safety features and parking
    sensors. I've been driving for 15 years with no accidents.

    Looking forward to your quote.
    Best regards,
    Sarah
    """

    state = gemini_agent.process_email(email)

    # Verify basic state
    assert state["customer_email"] == email
    assert state["customer_info"]["driver_name"] == "Sarah Johnson"
    assert state["customer_info"]["driver_age"] == "35"
    assert state["customer_info"]["vehicle_make"] == "Honda"
    assert state["customer_info"]["vehicle_model"] == "CR-V"
    assert state["customer_info"]["vehicle_year"] == "2021"
    assert state["customer_info"]["address"] == "456 Oak Ave, Seattle, WA"

    # Verify quote generation
    quote_msg = next((msg for msg in state["messages"] if msg["type"] == "quote"), None)
    assert quote_msg is not None
    assert "Quote ID:" in quote_msg["content"]
    assert state["quote_data"] is not None
    assert state["quote_ready"] is True

    # Verify agency review
    review_msg = next((msg for msg in state["messages"] if msg["type"] == "review_summary"), None)
    assert review_msg is not None
    assert state["review_summary"] is not None
    assert "priority" in state["review_summary"]


def test_incomplete_info_workflow(openai_agent):
    """Test workflow with incomplete information."""
    email = "Hi, I want car insurance. My name is Mike Brown."

    state = openai_agent.process_email(email)

    assert state["customer_email"] == email
    assert state["customer_info"]["driver_name"] == "Mike Brown"
    assert len(state["missing_info"]) > 0
    assert "driver_age" in state["missing_info"]
    assert "vehicle_make" in state["missing_info"]
    assert state["requires_review"] is True


def test_non_ppa_workflow(gemini_agent):
    """Test workflow with non-PPA inquiry."""
    email = """
    Hi, I'm interested in getting homeowners insurance for my new house.
    Can you help me with that?
    """

    state = gemini_agent.process_email(email)

    assert state["customer_email"] == email
    assert len(state["messages"]) == 0  # No messages for non-PPA inquiries
