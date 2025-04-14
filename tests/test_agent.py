"""Tests for the PPA Agent implementation."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.ppa_agent.agent import LLMProvider, PPAAgent


def create_mock_response(content: str) -> ChatResult:
    """Create a mock LLM response."""
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def create_mock_llm(responses):
    """Create a mock LLM with specified responses."""
    mock = MagicMock()
    mock.invoke.side_effect = [MagicMock(content=resp) for resp in responses]
    return mock


@pytest.fixture
def mock_llm():
    """Create a mock LLM with different responses for different method calls."""
    mock = MagicMock()

    # Create a side effect function to return different responses
    responses = [
        # First call - identify intention
        MagicMock(content={"intent": "new_business"}),
        # Second call - identify line of business
        MagicMock(content={"lob": "PPA"}),
        # Third call - analyze information
        MagicMock(
            content={
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
                "status": "info_incomplete",
            }
        ),
    ]

    mock.invoke.side_effect = responses
    return mock


@pytest.fixture
def openai_agent():
    """Create an OpenAI agent instance."""
    return PPAAgent(provider=LLMProvider.OPENAI, model="gpt-3.5-turbo", api_key="test-key")


@pytest.fixture
def gemini_agent():
    """Create a Gemini agent instance."""
    return PPAAgent(provider=LLMProvider.GEMINI, model="gemini-pro", api_key="test-key")


def test_agent_initialization_openai():
    """Test OpenAI agent initialization."""
    agent = PPAAgent(provider=LLMProvider.OPENAI, model="gpt-3.5-turbo", api_key="test-key")
    assert agent.provider == LLMProvider.OPENAI
    assert agent.model == "gpt-3.5-turbo"


def test_agent_initialization_gemini():
    """Test Gemini agent initialization."""
    agent = PPAAgent(provider=LLMProvider.GEMINI, model="gemini-pro", api_key="test-key")
    assert agent.provider == LLMProvider.GEMINI
    assert agent.model == "gemini-pro"


def test_process_new_business_email_openai(openai_agent):
    """Test processing a new business inquiry email with OpenAI."""
    email = """
    Hi, I'm interested in getting auto insurance for my 2020 Toyota Camry.
    My name is John Smith, I'm 30 years old, and I live at 123 Main St.
    """

    # Set up mock responses
    responses = [
        {"intent": "new_business"},
        {"lob": "PPA"},
        {
            "customer_info": {
                "driver_name": "John Smith",
                "driver_age": "30",
                "vehicle_make": "Toyota",
                "vehicle_model": "Camry",
                "vehicle_year": "2020",
                "address": "123 Main St",
            },
            "missing_info": [],
            "status": "info_complete",
        },
    ]

    openai_agent.llm = create_mock_llm(responses)
    state = openai_agent.process_email(email)

    assert state["customer_email"] == email
    assert state["customer_info"]["driver_name"] == "John Smith"
    assert state["customer_info"]["driver_age"] == "30"
    assert len(state["messages"]) == 0


def test_process_new_business_email_gemini(gemini_agent):
    """Test processing a new business inquiry email with Gemini."""
    email = """
    Hi, I'm interested in getting auto insurance for my 2020 Toyota Camry.
    My name is John Smith, I'm 30 years old, and I live at 123 Main St.
    """

    # Set up mock responses
    responses = [
        {"intent": "new_business"},
        {"lob": "PPA"},
        {
            "customer_info": {
                "driver_name": "John Smith",
                "driver_age": "30",
                "vehicle_make": "Toyota",
                "vehicle_model": "Camry",
                "vehicle_year": "2020",
                "address": "123 Main St",
            },
            "missing_info": [],
            "status": "info_complete",
        },
    ]

    gemini_agent.llm = create_mock_llm(responses)
    state = gemini_agent.process_email(email)

    assert state["customer_email"] == email
    assert state["customer_info"]["driver_name"] == "John Smith"
    assert state["customer_info"]["driver_age"] == "30"
    assert len(state["messages"]) == 0


def test_process_non_business_email(openai_agent):
    """Test processing an email that is not a new business inquiry."""
    email = "What's the status of my claim from last week?"

    # Set up mock responses
    responses = [{"intent": "other"}]

    openai_agent.llm = create_mock_llm(responses)
    state = openai_agent.process_email(email)

    assert state["customer_email"] == email
    assert len(state["messages"]) == 0


def test_process_incomplete_info_email(openai_agent):
    """Test processing an email with incomplete information."""
    email = "Hi, I want car insurance. My name is John Smith."

    # Set up mock responses
    responses = [
        {"intent": "new_business"},
        {"lob": "PPA"},
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
            "status": "info_incomplete",
        },
    ]

    openai_agent.llm = create_mock_llm(responses)
    state = openai_agent.process_email(email)

    assert state["customer_email"] == email
    assert len(state["missing_info"]) == 5
    assert "driver_age" in state["missing_info"]
    assert "vehicle_make" in state["missing_info"]
    assert state["requires_review"] is True
    assert any(msg["type"] == "info_request" for msg in state["messages"])


def test_error_handling(openai_agent):
    """Test agent's error handling capabilities."""
    email = ""  # Empty email to trigger error

    # Set up mock responses to raise an exception
    mock = MagicMock()
    mock.invoke.side_effect = Exception("API Error")
    openai_agent.llm = mock

    state = openai_agent.process_email(email)

    assert state["customer_email"] == email
    assert state["requires_review"] is True
    assert any(msg["type"] == "error" for msg in state["messages"])
