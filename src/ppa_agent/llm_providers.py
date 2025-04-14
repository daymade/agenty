"""LLM providers for the PPA Agent."""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import google.genai as genai
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate text from prompt asynchronously."""
        pass

    @abstractmethod
    def generate_sync(self, prompt: str) -> str:
        """Generate text from prompt synchronously."""
        pass


class GeminiProvider(BaseLLMProvider):
    """Provider for Google Gemini models using google-genai SDK."""

    api_key: str
    model: str
    _client: genai.Client = PrivateAttr()

    def __init__(self, api_key: str, model: str = "gemini-2.5-pro-exp-03-25") -> None:
        """Initialize the Gemini provider using the Google Gemini SDK."""
        self.api_key = api_key
        self.model = model
        # Initialize the client using the API key
        # The SDK will automatically pick up GOOGLE_API_KEY if api_key is None/empty
        self._client = genai.Client(api_key=self.api_key if self.api_key else None)

    def generate_sync(self, prompt: str) -> str:
        """Generate text from prompt synchronously using the Gemini SDK."""
        try:
            # Use client.models.generate_content
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                # No config parameter needed for basic usage
            )
            return response.text
        except genai.errors.APIError as e:
            # Check if this is a safety error (blocked prompt)
            if "blocked" in str(e).lower() or "safety" in str(e).lower():
                logger.warning(f"Gemini prompt likely blocked: {e}")
                return "Response blocked due to safety concerns."
            # Re-raise other API errors
            logger.error(f"Gemini API error: {e}")
            raise
        # Add specific error handling for API errors
        except Exception as e:
            # Log the specific error type and message
            logger.error(f"Gemini API error ({type(e).__name__}): {e}")
            # Consider raising a more specific custom exception or re-raising
            raise

    async def generate(self, prompt: str) -> str:
        """Generate text from prompt asynchronously."""
        # Using the async method from the Gemini SDK
        try:
            response = await self._client.models.generate_content_async(
                model=self.model,
                contents=prompt,
                # No config parameter needed for basic usage
            )
            return response.text
        except genai.errors.APIError as e:
            # Check if this is a safety error (blocked prompt)
            if "blocked" in str(e).lower() or "safety" in str(e).lower():
                logger.warning(f"Gemini prompt likely blocked: {e}")
                return "Response blocked due to safety concerns."
            # Re-raise other API errors
            logger.error(f"Gemini API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Gemini API error ({type(e).__name__}): {e}")
            raise


class GeminiChatAdapter(BaseChatModel):
    """Adapter to make Gemini provider compatible with LangChain's chat interface."""

    model: str = Field(default="gemini-2.5-pro-exp-03-25", description="The model to use")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    _client: genai.Client = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-pro-exp-03-25",
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter using the Google Gemini SDK."""
        super().__init__(model=model, temperature=temperature, **kwargs)
        self.model = model
        # Initialize the genai client directly within the adapter
        self._client = genai.Client(api_key=api_key if api_key else None)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate completions for the given messages using google-genai."""
        # Convert LangChain messages to a single prompt string
        # TODO: Improve this to pass structured history if possible/needed
        prompt = "\n".join(f"{type(msg).__name__}: {msg.content}" for msg in messages)

        # Create the config parameter with direct temperature and responseMimeType fields
        config = genai.types.GenerateContentConfig(
            temperature=self.temperature,
            responseMimeType="application/json",
        )

        try:
            # Use the internal client with the correct API
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )

            # Handle potential errors or blocked responses
            if not response.candidates:
                content = "Response blocked or empty."
                finish_reason = "blocked"
                # Check for prompt feedback details
                if (
                    hasattr(response, "prompt_feedback")
                    and response.prompt_feedback
                    and response.prompt_feedback.block_reason
                ):
                    content = (
                        f"Response blocked due to: {response.prompt_feedback.block_reason.name}"
                    )
                    logger.warning(
                        f"Gemini prompt blocked. Reason: {response.prompt_feedback.block_reason.name}. Feedback: {response.prompt_feedback}"
                    )
                else:
                    logger.warning("Gemini response empty.")
                generation = ChatGeneration(
                    message=AIMessage(content=content),
                    generation_info={"finish_reason": finish_reason},
                )
            else:
                # Process successful response - expecting JSON structured output
                response_text = response.text
                # The response.text should contain the JSON string when response_mime_type="application/json" is used.
                # We still need to parse it, but it avoids the need for ```json``` stripping.
                try:
                    # Directly parse the response text as JSON
                    json_response = json.loads(response_text)
                    # Ensure content is string for AIMessage (Langchain requires string content)
                    generation = ChatGeneration(
                        message=AIMessage(content=json.dumps(json_response)),
                        generation_info={"finish_reason": "stop"},
                    )
                except json.JSONDecodeError as json_err:
                    logger.warning(
                        f"Failed to decode JSON response despite requesting JSON mime type: {json_err}. Response text: {response_text}"
                    )
                    # Fallback to using the raw text if JSON parsing fails
                    generation = ChatGeneration(
                        message=AIMessage(content=response_text),
                        generation_info={
                            "finish_reason": "stop",
                            "json_decode_error": str(json_err),
                        },
                    )

            return ChatResult(generations=[generation])

        # Specific handling for blocked prompts during the API call itself
        #        except genai.errors.BlockedPromptError as e:
        #             logger.warning(f"Gemini prompt blocked during API call: {e}")
        #             generation = ChatGeneration(
        #                 message=AIMessage(content="Prompt blocked due to safety concerns."),
        #                 generation_info={"finish_reason": "blocked"}
        #             )
        #             return ChatResult(generations=[generation])
        # General exception handling
        except Exception as e:
            # Log the full traceback for better debugging
            logger.error(f"Generation error ({type(e).__name__}): {e}", exc_info=True)
            generation = ChatGeneration(
                message=AIMessage(content=f"Error during generation: {type(e).__name__}"),
                generation_info={"finish_reason": "error"},
            )
            return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "gemini"


class GeminiConfig(BaseModel):
    """Configuration for Gemini provider."""

    api_key: str = Field(description="The API key to use")
    model: str = Field(default="gemini-2.5-pro-exp-03-25", description="The model to use")

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        """Create a config from environment variables."""
        return cls(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-pro-exp-03-25"),
        )
