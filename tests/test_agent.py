"""Tests for the PPA Agent implementation."""

import json
import os
from unittest.mock import MagicMock

import pytest

from src.ppa_agent.agent import PPAAgent
from src.ppa_agent.llm_providers import LLMProvider, BaseLLMProvider


def create_mock_llm(responses):
    """Create a mock LLM with specified responses for generate_sync."""
    mock = MagicMock()
    # Mock generate_sync to return JSON strings
    mock.generate_sync.side_effect = [json.dumps(resp) for resp in responses]
    return mock


@pytest.fixture
def mock_llm():
    """Create a mock LLM with different responses for different method calls."""
    # These are now the JSON objects expected from the prompts
    responses = [
        # First call - identify intention
        {"intent": "new_business"},
        # Second call - identify line of business
        {"lob": "PPA"},
        # Third call - analyze information
        {
            "customer_info": {
                "driver_name": "John Smith",
                "driver_age": None,
                "vehicle_make": None,
                "vehicle_model": None,
                "vehicle_year": None,
                "address": None,
            },
            "missing_info": [
                "driver_age",
                "vehicle_make",
                "vehicle_model",
                "vehicle_year",
                "address",
            ],
        },
        # Fourth call (if info incomplete) - generate_info_request (returns TEXT)
        "To provide a quote, please tell us your age, vehicle details (make, model, year), and address.",
        # Fifth call (if info incomplete) - prepare_agency_review (returns TEXT)
        "Customer John Smith requested PPA. Missing info: driver_age, vehicle_make, vehicle_model, vehicle_year, address. Sent info request.",
    ]

    mock = MagicMock()
    # Side effect needs both JSON strings and plain text strings
    side_effects = []
    # Identify intent (JSON)
    side_effects.append(json.dumps(responses[0]))
    # Identify LOB (JSON)
    side_effects.append(json.dumps(responses[1]))
    # Analyze info (JSON)
    side_effects.append(json.dumps(responses[2]))
    # Generate info request (TEXT)
    side_effects.append(responses[3])
    # Prepare agency review (TEXT)
    side_effects.append(responses[4])

    mock.generate_sync.side_effect = side_effects
    return mock


@pytest.fixture
def openai_agent():
    """Create an OpenAI agent instance with a mocked provider."""
    # Create the mock provider FIRST
    mock_provider = MagicMock(spec=BaseLLMProvider)
    # The tests will set the side_effect on generate_sync later
    # Pass the mock provider directly to the agent
    agent = PPAAgent(llm_provider=mock_provider)
    # Set provider attribute for tests that check it (optional)
    agent.provider_type = LLMProvider.OPENAI
    return agent


@pytest.fixture
def gemini_agent():
    """Create a Gemini agent instance with a mocked provider."""
    # Create the mock provider FIRST
    mock_provider = MagicMock(spec=BaseLLMProvider)
    # The tests will set the side_effect on generate_sync later
    # Pass the mock provider directly to the agent
    agent = PPAAgent(llm_provider=mock_provider)
    # Set provider attribute for tests that check it (optional)
    agent.provider_type = LLMProvider.GEMINI
    return agent


def test_process_new_business_email_openai(openai_agent):
    """Test processing a new business inquiry email with mocked OpenAI."""
    email = """
    Hi, I'm interested in getting auto insurance for my 2020 Toyota Camry.
    My name is John Smith, I'm 30 years old, and I live at 123 Main St.
    """

    # Set up mock responses (JSON objects or text)
    responses = [
        {"intent": "new_business"},
        {"lob": "PPA"},
        { # analyze_information (complete)
            "customer_info": {
                "driver_name": "John Smith", "driver_age": "30",
                "vehicle_make": "Toyota", "vehicle_model": "Camry",
                "vehicle_year": "2020", "address": "123 Main St",
            },
            "missing_info": [],
        },
        {"status": "no_proof_needed", "message_to_customer": None}, # check_discounts
    ]

    openai_agent.llm.generate_sync.side_effect = [
        json.dumps(responses[0]), # intent
        json.dumps(responses[1]), # lob
        json.dumps(responses[2]), # analyze
        json.dumps(responses[3]), # discount check
    ]

    state = openai_agent.process_email(email)

    assert state["customer_email"] == email
    assert state["customer_info"]["driver_name"] == "John Smith"
    assert state["status"] == "quote_generated", f"Expected quote_generated, got {state.get('status')}"
    assert state.get("quote_ready") is True
    assert state["quote_data"] is not None


def test_process_new_business_email_gemini(gemini_agent):
    """Test processing a new business inquiry email with mocked Gemini."""
    email = """
    Hi, I'm interested in getting auto insurance for my 2020 Toyota Camry.
    My name is John Smith, I'm 30 years old, and I live at 123 Main St.
    """

    # Set up mock responses (JSON objects or text)
    responses = [
        {"intent": "new_business"},
        {"lob": "PPA"},
        { # analyze_information (complete)
            "customer_info": {
                "driver_name": "John Smith", "driver_age": "30",
                "vehicle_make": "Toyota", "vehicle_model": "Camry",
                "vehicle_year": "2020", "address": "123 Main St",
            },
            "missing_info": [],
        },
        {"status": "no_proof_needed", "message_to_customer": None}, # check_discounts
    ]

    gemini_agent.llm.generate_sync.side_effect = [
        json.dumps(responses[0]),
        json.dumps(responses[1]),
        json.dumps(responses[2]),
        json.dumps(responses[3]),
    ]

    state = gemini_agent.process_email(email)

    assert state["customer_email"] == email
    assert state["customer_info"]["driver_name"] == "John Smith"
    assert state["status"] == "quote_generated", f"Expected quote_generated, got {state.get('status')}"
    assert state.get("quote_ready") is True
    assert state["quote_data"] is not None


def test_process_non_business_email(openai_agent):
    """Test processing an email that is not a new business inquiry."""
    email = "What's the status of my claim from last week?"

    # Set up mock responses
    responses = [{"intent": "other"}]

    # Set the side effect on the MOCKED provider
    openai_agent.llm.generate_sync.side_effect = [json.dumps(resp) for resp in responses]
    state = openai_agent.process_email(email)

    assert state["customer_email"] == email
    assert state["intent"] == "other"
    assert state["status"] == "new"
    assert not state["messages"]


def test_process_incomplete_info_email(openai_agent):
    """Test processing an email with incomplete information."""
    email = "Hi, I want car insurance. My name is John Smith."

    # Set up mock responses
    responses = [
        {"intent": "new_business"},
        {"lob": "PPA"},
        { # analyze (incomplete)
            "customer_info": {"driver_name": "John Smith"},
            "missing_info": ["driver_age", "vehicle_make", "vehicle_model", "vehicle_year", "address"],
        },
        # generate_info_request LLM call (TEXT)
        "To complete your quote, please provide: driver_age, ...",
        # prepare_agency_review node runs next (NO LLM CALL)
        "Mocked review summary for incomplete info test.",
    ]

    openai_agent.llm.generate_sync.side_effect = [
        json.dumps(responses[0]),
        json.dumps(responses[1]),
        json.dumps(responses[2]),
        responses[3], # generate_info_request call
        responses[4],
    ]
    state = openai_agent.process_email(email)

    assert state["customer_email"] == email
    assert state["customer_info"].get("driver_name") == "John Smith"
    assert len(state["missing_info"]) == 5
    assert "driver_age" in state["missing_info"]
    assert "vehicle_make" in state["missing_info"]
    assert state["requires_review"] is True
    assert state["status"] == "ready_for_review"
    assert any(msg["type"] == "info_request" for msg in state["messages"])
    assert any(msg["type"] == "quote_summary_for_review" for msg in state["messages"])


def test_error_handling(openai_agent):
    """Test agent's error handling capabilities when LLM call fails."""
    email = "Hi there"

    # Set up mock provider to raise an exception on the first generate_sync call
    openai_agent.llm.generate_sync.side_effect = ValueError("LLM API Error")

    state = openai_agent.process_email(email)

    assert state["customer_email"] == email
    assert state["intent"] == "error"
    # Status should remain 'new' as error happens before status update
    assert state.get("status") == "new", "Status should remain 'new' on early LLM error"
    assert any(msg["type"] == "final_error" for msg in state["messages"])
