"""
PPA New Business AI Agent implementation.

This module implements a LangGraph-based agent for handling new PPA insurance business inquiries.
"""

import json
import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .llm_providers import GeminiChatAdapter

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Configure console handler if not already configured
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# Constants
PPA_REQUIREMENTS = [
    "driver_name",
    "driver_age",
    "vehicle_make",
    "vehicle_model",
    "vehicle_year",
    "address",
]


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    GEMINI = "gemini"


class AgentState(TypedDict):
    """State maintained throughout the agent's execution."""

    customer_email: str  # Raw content of the customer's email(s)
    customer_info: Dict[str, Any]  # Information extracted about the customer/vehicle
    missing_info: List[str]  # List of required fields still missing
    ppa_requirements: List[str]  # List of fields needed for a PPA quote
    quote_ready: bool  # Flag indicating if a quote can be generated
    quote_data: Optional[Dict[str, Any]]  # Holds the generated (mock) quote data
    proof_of_discount: Optional[bool]  # Status of discount proof (True/False/None if unknown)
    messages: List[Dict[str, Any]]  # History of messages (internal and drafts for customer)
    requires_review: bool  # Flag indicating if the current state requires human review


class PPAAgent:
    """Agent for handling PPA insurance new business inquiries."""

    def __init__(
        self,
        provider: Union[str, LLMProvider] = LLMProvider.OPENAI,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the PPA Agent.

        Args:
            provider: LLM provider to use (default: openai)
            model: Model name (provider-specific)
            api_key: API key for the provider
        """
        self.provider = LLMProvider(provider)

        if self.provider == LLMProvider.OPENAI:
            self.model = model or "gpt-4-turbo-preview"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key not provided")
            self.llm = ChatOpenAI(model=self.model, api_key=self.api_key)

        elif self.provider == LLMProvider.GEMINI:
            self.model = model or "gemini-2.5-pro-exp-03-25"
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("Gemini API key not provided")
            self.llm = GeminiChatAdapter(api_key=self.api_key, model=self.model)

        self.workflow = self._build_workflow()

    def _identify_intention(self, state: dict) -> dict:
        """Identify the intention of the email."""
        prompt = f"""
        Identify the intention of this email:

        {state["customer_email"]}

        You MUST respond with ONLY a JSON object in this exact format:
        {{
            "intent": "new_business" | "policy_change" | "other"
        }}

        Do not include any other text, markdown, or formatting.
        """

        try:
            response = self.llm.invoke(prompt)
            logger.info(f"Raw response from LLM: {response}")
            if isinstance(response, str):
                logger.info("Response is string, parsing as JSON")
                result = json.loads(response)
            elif hasattr(response, "content"):
                logger.info(f"Response has content attribute: {response.content}")
                if isinstance(response.content, str):
                    try:
                        result = json.loads(response.content)
                        logger.info(f"Parsed JSON result: {result}")
                    except json.JSONDecodeError:
                        # If content is not JSON, try to extract intent from text
                        content = response.content.lower()
                        logger.info(f"Failed to parse JSON, extracting from text: {content}")
                        if "new business" in content:
                            result = {"intent": "new_business"}
                        elif "policy change" in content:
                            result = {"intent": "policy_change"}
                        else:
                            result = {"intent": "other"}
                else:
                    logger.info(f"Content is not string: {type(response.content)}")
                    result = response.content
            else:
                logger.info(f"Response is something else: {type(response)}")
                result = response

            return {"intent": result.get("intent", "other")}
        except Exception as e:
            logger.error(f"Error identifying intention: {str(e)}")
            return {"intent": "error"}

    def _identify_line_of_business(self, state: dict) -> dict:
        """Identify the line of business from the email."""
        prompt = f"""
        Identify the line of business from this email:

        {state["customer_email"]}

        You MUST respond with ONLY a JSON object in this exact format:
        {{
            "lob": "PPA" | "HOME" | "OTHER"
        }}

        Do not include any other text, markdown, or formatting.
        """

        try:
            response = self.llm.invoke(prompt)
            if isinstance(response, str):
                result = json.loads(response)
            elif hasattr(response, "content"):
                if isinstance(response.content, str):
                    try:
                        result = json.loads(response.content)
                    except json.JSONDecodeError:
                        # If content is not JSON, try to extract LOB from text
                        content = response.content.lower()
                        if "ppa" in content or "auto" in content:
                            result = {"lob": "PPA"}
                        elif "home" in content:
                            result = {"lob": "HOME"}
                        else:
                            result = {"lob": "OTHER"}
                else:
                    result = response.content
            else:
                result = response

            return {"lob": result.get("lob", "OTHER")}
        except Exception as e:
            logger.error(f"Error identifying line of business: {str(e)}")
            return {"lob": "error"}

    def _analyze_information(self, state: dict) -> dict:
        """Analyze the customer information from the email."""
        state["ppa_requirements"] = PPA_REQUIREMENTS

        # Create a prompt for the LLM to extract required information
        prompt = f"""
        Extract the following required information from the customer's email:
        {json.dumps(PPA_REQUIREMENTS, indent=2)}

        Customer's email:
        {state["customer_email"]}

        You MUST respond with ONLY a JSON object in this exact format:
        {{
            "customer_info": {{
                "driver_name": string,
                "driver_age": string,
                "vehicle_make": string,
                "vehicle_model": string,
                "vehicle_year": string,
                "address": string
            }},
            "missing_info": [string],
            "status": "info_complete" | "info_incomplete"
        }}

        Do not include any other text, markdown, or formatting.
        """

        try:
            response = self.llm.invoke(prompt)
            if isinstance(response.content, str):
                try:
                    result = json.loads(response.content)
                except json.JSONDecodeError:
                    # If not JSON, try to extract structured data
                    lines = response.content.split("\n")
                    customer_info = {}
                    missing_info = []
                    for line in lines:
                        if ":" in line:
                            key, value = line.split(":", 1)
                            key = key.strip().lower().replace(" ", "_")
                            value = value.strip()
                            if value and value != "None":
                                customer_info[key] = value
                            else:
                                missing_info.append(key)
                    result = {
                        "customer_info": customer_info,
                        "missing_info": missing_info,
                        "status": "info_incomplete" if missing_info else "info_complete",
                    }
            else:
                result = response.content

            # Update state with customer info
            if "customer_info" in result:
                state["customer_info"].update(result["customer_info"])

            # Update missing info and add message if needed
            if "missing_info" in result:
                state["missing_info"] = result["missing_info"]
                if len(state["missing_info"]) > 0:
                    missing_fields = ", ".join(state["missing_info"])
                    state["messages"].append(
                        {
                            "role": "agent",
                            "content": f"Please provide the following information: {missing_fields}",
                            "type": "info_request",
                            "requires_review": True,
                        }
                    )
                    state["requires_review"] = True

            return {"status": result.get("status", "info_incomplete")}

        except Exception as e:
            logger.error(f"Error analyzing information: {str(e)}")
            return {"status": "error"}

    def _generate_info_request(self, state: AgentState) -> AgentState:
        """Generate an email requesting missing information.

        Args:
            state: Current agent state.

        Returns:
            Updated agent state.
        """
        logger.info("Generating information request email...")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a professional insurance agent. Write a polite email "
                    "requesting the following missing information from a potential "
                    "customer: {missing_info}. Use any existing information we have "
                    "as context: {customer_info}",
                ),
            ]
        )

        try:
            email_content = prompt.invoke(
                {
                    "missing_info": ", ".join(state["missing_info"]),
                    "customer_info": json.dumps(state["customer_info"]),
                }
            ).content

            message = {
                "role": "agent",
                "content": email_content,
                "type": "info_request",
                "requires_review": True,
            }
            state["messages"].append(message)
            state["requires_review"] = True

            logger.info("Generated information request email")
            return state
        except Exception as e:
            logger.error(f"Error generating information request: {e}")
            return state

    def _ask_for_discount_proof(self, state: dict) -> dict:
        """Ask for proof of discount eligibility."""
        prompt = f"""
        Based on the customer information, determine if they might be eligible for any discounts
        and ask for proof. Consider:
        - Good student discount (if driver age < 25)
        - Safe driver discount (clean record)
        - Multi-car discount
        - Bundling discount

        Customer info:
        {json.dumps(state["customer_info"], indent=2)}

        You MUST respond with ONLY a JSON object in this exact format:
        {{
            "discounts": [string],
            "proof_needed": boolean,
            "message": string
        }}

        Do not include any other text, markdown, or formatting.
        """

        try:
            response = self.llm.invoke(prompt)
            if isinstance(response, str):
                result = json.loads(response)
            elif hasattr(response, "content"):
                if isinstance(response.content, str):
                    try:
                        result = json.loads(response.content)
                    except json.JSONDecodeError:
                        # If not JSON, try to extract from text
                        content = response.content.lower()
                        if "discount" in content and "proof" in content:
                            result = {
                                "discounts": ["safe driver"],
                                "proof_needed": True,
                                "message": response.content,
                            }
                        else:
                            result = {"discounts": [], "proof_needed": False, "message": ""}
                else:
                    result = response.content
            else:
                result = response

            if result.get("proof_needed"):
                state["messages"].append(
                    {
                        "role": "agent",
                        "content": result["message"],
                        "type": "discount_proof_request",
                        "requires_review": True,
                    }
                )
                state["requires_review"] = True
                # Continue with quote generation even if proof is needed
                return {"status": "no_proof_needed"}

            return {"status": "no_proof_needed"}

        except Exception as e:
            logger.error(f"Error asking for discount proof: {str(e)}")
            return {"status": "error"}

    def _generate_quote(self, state: dict) -> dict:
        """Generate a mock quote based on customer information."""
        prompt = f"""
        Generate a mock insurance quote based on the following customer information:
        {json.dumps(state["customer_info"], indent=2)}

        Consider:
        - Base premium calculation
        - Age-based adjustments
        - Vehicle-based factors
        - Available discounts

        You MUST respond with ONLY a JSON object in this exact format:
        {{
            "quote_id": string,
            "base_premium": number,
            "discounts": [string],
            "final_premium": number,
            "coverage": {{
                "liability": {{
                    "bodily_injury": string,
                    "property_damage": string
                }},
                "collision": {{
                    "deductible": number
                }},
                "comprehensive": {{
                    "deductible": number
                }}
            }}
        }}

        Do not include any other text, markdown, or formatting.
        """

        try:
            response = self.llm.invoke(prompt)
            if isinstance(response, str):
                result = json.loads(response)
            elif hasattr(response, "content"):
                if isinstance(response.content, str):
                    try:
                        result = json.loads(response.content)
                    except json.JSONDecodeError:
                        # If not JSON, try to extract structured data
                        lines = response.content.split("\n")
                        result = {
                            "quote_id": "Q" + str(hash(state["customer_email"]))[:8],
                            "base_premium": 1000.0,
                            "discounts": [],
                            "final_premium": 1000.0,
                            "coverage": {
                                "liability": {
                                    "bodily_injury": "100k/300k",
                                    "property_damage": "50k",
                                },
                                "collision": {"deductible": 500},
                                "comprehensive": {"deductible": 500},
                            },
                        }
                else:
                    result = response.content
            else:
                result = response

            state["quote_data"] = result
            state["quote_ready"] = True

            # Generate quote summary message
            summary = f"""
            Here's your estimated quote:
            - Base Premium: ${result['base_premium']:.2f}
            - Applied Discounts: {', '.join(result['discounts'])}
            - Final Premium: ${result['final_premium']:.2f}

            Coverage:
            - Liability: {result['coverage']['liability']['bodily_injury']} bodily injury / {result['coverage']['liability']['property_damage']} property damage
            - Collision Deductible: ${result['coverage']['collision']['deductible']}
            - Comprehensive Deductible: ${result['coverage']['comprehensive']['deductible']}

            Quote ID: {result['quote_id']}
            """

            state["messages"].append(
                {"role": "agent", "content": summary, "type": "quote", "requires_review": True}
            )

            return {"status": "quote_generated"}

        except Exception as e:
            logger.error(f"Error generating quote: {str(e)}")
            return {"status": "error"}

    def _agency_review(self, state: dict) -> dict:
        """Prepare case for agency review."""
        prompt = f"""
        Review the customer case and generate a summary for agency review.
        Include any red flags, special considerations, or follow-up items.

        Customer Information:
        {json.dumps(state["customer_info"], indent=2)}

        Quote Data:
        {json.dumps(state.get("quote_data", {}), indent=2)}

        You MUST respond with ONLY a JSON object in this exact format:
        {{
            "summary": string,
            "red_flags": [string],
            "follow_up_items": [string],
            "priority": "high" | "medium" | "low"
        }}

        Do not include any other text, markdown, or formatting.
        """

        try:
            response = self.llm.invoke(prompt)
            if isinstance(response, str):
                result = json.loads(response)
            elif hasattr(response, "content"):
                if isinstance(response.content, str):
                    try:
                        result = json.loads(response.content)
                    except json.JSONDecodeError:
                        # If not JSON, try to extract structured data
                        lines = response.content.split("\n")
                        result = {
                            "summary": response.content,
                            "red_flags": [],
                            "follow_up_items": [],
                            "priority": "medium",
                        }
                else:
                    result = response.content
            else:
                result = response

            # Add review summary to state
            state["review_summary"] = result
            state["requires_review"] = True

            # Add internal note for review
            state["messages"].append(
                {
                    "role": "agent",
                    "content": f"""
                Priority: {result['priority'].upper()}

                Summary:
                {result['summary']}

                Red Flags:
                {chr(10).join('- ' + flag for flag in result['red_flags'])}

                Follow-up Items:
                {chr(10).join('- ' + item for item in result['follow_up_items'])}
                """,
                    "type": "review_summary",
                    "requires_review": True,
                }
            )

            return {"status": "ready_for_review"}

        except Exception as e:
            logger.error(f"Error preparing agency review: {str(e)}")
            return {"status": "error"}

    def _build_workflow(self) -> StateGraph:
        """Build the agent workflow graph.

        Returns:
            Configured StateGraph instance.
        """
        # Create the workflow
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("identify_intention", self._identify_intention)
        workflow.add_node("identify_line_of_business", self._identify_line_of_business)
        workflow.add_node("analyze_information", self._analyze_information)
        workflow.add_node("ask_for_discount_proof", self._ask_for_discount_proof)
        workflow.add_node("generate_quote", self._generate_quote)
        workflow.add_node("agency_review", self._agency_review)

        # Set entry point
        workflow.set_entry_point("identify_intention")

        # Add edges
        workflow.add_conditional_edges(
            "identify_intention",
            lambda x: x["intent"],
            {
                "new_business": "identify_line_of_business",
                "other": END,
                "error": END,
            },
        )

        workflow.add_conditional_edges(
            "identify_line_of_business",
            lambda x: x["lob"],
            {
                "PPA": "analyze_information",
                "other": END,
                "error": END,
            },
        )

        workflow.add_conditional_edges(
            "analyze_information",
            lambda x: x["status"],
            {
                "info_incomplete": END,
                "info_complete": "ask_for_discount_proof",
                "error": END,
            },
        )

        workflow.add_conditional_edges(
            "ask_for_discount_proof",
            lambda x: x["status"],
            {
                "proof_needed": END,
                "no_proof_needed": "generate_quote",
                "error": END,
            },
        )

        workflow.add_conditional_edges(
            "generate_quote",
            lambda x: x["status"],
            {
                "quote_generated": "agency_review",
                "error": END,
            },
        )

        # Add edge from agency_review to END
        workflow.add_edge("agency_review", END)

        # Compile the workflow
        return workflow.compile()

    def process_email(self, email: str) -> dict:
        """Process a customer email and return the updated state."""
        # Initialize state
        state = {
            "customer_email": email,
            "customer_info": {},
            "missing_info": [],
            "messages": [],
            "requires_review": False,
            "ppa_requirements": [],
        }

        try:
            # Identify intention
            logger.info("Identifying intention...")
            intention_result = self._identify_intention(state)
            logger.info(f"Intention result: {intention_result}")
            if intention_result["intent"] == "error":
                raise Exception("Failed to identify intention")

            # If not a new business inquiry, return early
            if intention_result["intent"] != "new_business":
                return state

            # Identify line of business
            logger.info("Identifying line of business...")
            lob_result = self._identify_line_of_business(state)
            logger.info(f"LOB result: {lob_result}")
            if lob_result["lob"] == "error":
                raise Exception("Failed to identify line of business")

            # If not PPA, return early
            if lob_result["lob"] != "PPA":
                return state

            # Analyze information
            logger.info("Analyzing information...")
            info_result = self._analyze_information(state)
            logger.info(f"Info result: {info_result}")
            if info_result["status"] == "error":
                raise Exception("Failed to analyze information")

            # If information is complete, check for discounts
            if info_result["status"] == "info_complete":
                logger.info("Checking for discounts...")
                discount_result = self._ask_for_discount_proof(state)
                logger.info(f"Discount result: {discount_result}")

                # If no proof needed, generate quote
                if discount_result["status"] == "no_proof_needed":
                    logger.info("Generating quote...")
                    quote_result = self._generate_quote(state)
                    logger.info(f"Quote result: {quote_result}")

                    # If quote generated, prepare agency review
                    if quote_result["status"] == "quote_generated":
                        logger.info("Preparing agency review...")
                        review_result = self._agency_review(state)
                        logger.info(f"Review result: {review_result}")

        except Exception as e:
            logger.error(f"Error processing email: {str(e)}")
            state["requires_review"] = True
            state["messages"].append(
                {
                    "role": "agent",
                    "content": "I apologize, but I encountered an error while processing your request. A human agent will review your email and get back to you shortly.",
                    "type": "error",
                    "requires_review": True,
                }
            )

        return state
