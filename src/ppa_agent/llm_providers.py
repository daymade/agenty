"""LLM providers for the PPA Agent."""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional

import google.genai as genai
from pydantic import BaseModel, Field, PrivateAttr, confloat

# Add openai import
import openai

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Enum for supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt asynchronously."""
        pass

    @abstractmethod
    def generate_sync(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt synchronously."""
        pass


class GeminiProvider(BaseLLMProvider, BaseModel):
    """Provider for Google Gemini models using the NEW google-genai SDK."""

    api_key: str = Field(description="The API key to use")
    model: str = Field(default="gemini-2.5-pro-exp-03-25", description="The model to use")
    temperature: confloat(ge=0.0, le=1.0) = Field(default=0.7, description="Sampling temperature")
    _client: genai.Client = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-pro-exp-03-25",
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        """Initialize the Gemini provider using the NEW Google Genai SDK Client."""
        super().__init__(api_key=api_key, model=model, temperature=temperature, **kwargs)
        # Initialize the client using the API key
        # SDK automatically picks up GOOGLE_API_KEY if api_key is None/empty
        self._client = genai.Client(api_key=self.api_key if self.api_key else None)

    def generate_sync(self, prompt: str, structured: bool = False, **kwargs: Any) -> str:
        """Generate text from prompt synchronously using the NEW Gemini SDK."""
        logger.debug(f"Sending prompt to Gemini: {prompt}")
        # Use GenerateContentConfig with camelCase based on migration guide
        config = genai.types.GenerateContentConfig(
            temperature=self.temperature,
            responseMimeType="application/json" if structured else None,
        )
        try:
            response: genai.types.GenerateContentResponse = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,  # Use config instead of generation_config
                **kwargs
            )

            if not response.candidates:
                content = "Response blocked or empty."
                if hasattr(response, "prompt_feedback") and response.prompt_feedback and response.prompt_feedback.block_reason:
                    content = f"Response blocked due to: {response.prompt_feedback.block_reason.name}"
                    logger.warning(f"Gemini prompt blocked. Reason: {response.prompt_feedback.block_reason.name}. Feedback: {response.prompt_feedback}")
                else:
                    logger.warning("Gemini response empty.")
                raise ValueError(content)
            
            if not response.candidates[0].content.parts:
                raise ValueError("Received empty response parts from Gemini API")

            response_text = response.candidates[0].content.parts[0].text
            logger.debug(f"Raw response from Gemini: {response_text}")

            if structured:
                # Attempt to extract JSON robustly
                json_str = response_text.strip() # Strip leading/trailing whitespace
                # Check if response is wrapped in markdown code block
                if json_str.startswith("```json\n") and json_str.endswith("\n```"):
                    json_str = json_str[len("```json\n"):-len("\n```")]
                elif json_str.startswith("```") and json_str.endswith("```"):
                     # Handle potential block without language specifier
                    json_str = json_str[3:-3]
                
                json_str = json_str.strip() # Strip again after potential extraction

                try:
                    # Parse the cleaned JSON string
                    parsed_json = json.loads(json_str)
                    # Return the validated JSON string representation
                    return json.dumps(parsed_json) 
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON response from Gemini after extraction attempt. Original text: '{response_text}', Parsed string: '{json_str}', Error: {e}")
                    raise ValueError("Response could not be parsed as valid JSON") from e
            else:
                return response_text # Return raw text if not structured

        except Exception as e:
            # Catch other potential API errors (network, authentication, etc.)
            logger.error(f"Gemini generic error ({type(e).__name__}): {e}", exc_info=False) # Avoid duplicate traceback for simple ValueErrors
            # Re-raise specific errors or a generic one
            if isinstance(e, ValueError): # Propagate our JSON parsing error
                raise e
            raise RuntimeError(f"Error interacting with Gemini API: {e}") from e

    async def generate(self, prompt: str, structured: bool = False, **kwargs: Any) -> str:
        """Generate text from prompt asynchronously using the NEW Gemini SDK."""
        config = genai.types.GenerateContentConfig(
            temperature=self.temperature,
            responseMimeType="application/json" if structured else None,
        )
        try:
            response: genai.types.GenerateContentResponse = await self._client.models.generate_content_async(
                model=self.model,
                contents=prompt,
                config=config,  # Use config instead of generation_config
                **kwargs
            )

            if not response.candidates:
                content = "Response blocked or empty (async)."
                if hasattr(response, "prompt_feedback") and response.prompt_feedback and response.prompt_feedback.block_reason:
                    content = f"Response blocked due to: {response.prompt_feedback.block_reason.name} (async)"
                    logger.warning(f"Gemini prompt blocked (async). Reason: {response.prompt_feedback.block_reason.name}. Feedback: {response.prompt_feedback}")
                else:
                    logger.warning("Gemini response empty (async).")
                raise ValueError(content)
            
            # For structured output, validate JSON
            if structured:
                try:
                    json_str = response.text
                    json.loads(json_str)  # Validate JSON
                    return json_str
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON response from Gemini: {response.text}")
                    raise ValueError("Response was not valid JSON")
            
            # For unstructured output, return text directly
            return response.text

        except (genai.types.BlockedPromptException, genai.types.StopCandidateException) as e:
             logger.warning(f"Gemini prompt/content blocked during async API call: {e}")
             raise ValueError(f"Response blocked due to safety concerns (async): {e}")
        except Exception as e:
            logger.error(f"Gemini async API error ({type(e).__name__}): {e}", exc_info=True)
            raise


class OpenAIProvider(BaseLLMProvider, BaseModel):
    """Provider for OpenAI models using the openai SDK."""
    api_key: str = Field(description="The API key to use")
    model: str = Field(default="gpt-3.5-turbo", description="The model to use")
    temperature: confloat(ge=0.0, le=1.0) = Field(default=0.7, description="Sampling temperature")
    _client: openai.OpenAI = PrivateAttr()
    _async_client: openai.AsyncOpenAI = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI provider using the OpenAI SDK."""
        super().__init__(api_key=api_key, model=model, temperature=temperature, **kwargs)
        self._client = openai.OpenAI(api_key=self.api_key)
        self._async_client = openai.AsyncOpenAI(api_key=self.api_key)

    def generate_sync(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt synchronously using the OpenAI SDK."""
        try:
            # Use chat completions endpoint
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    # Assuming simple prompt, convert to chat message format
                    # TODO: Potentially enhance to handle chat history if needed later
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}, # Request JSON output
                **kwargs,
            )
            content = response.choices[0].message.content
            if content is None:
                 raise ValueError("Received null content from OpenAI.")
            return content
        except Exception as e:
            logger.error(f"OpenAI API error ({type(e).__name__}): {e}", exc_info=True)
            raise

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt asynchronously using the OpenAI SDK."""
        try:
            response = await self._async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}, # Request JSON output
                **kwargs,
            )
            content = response.choices[0].message.content
            if content is None:
                 raise ValueError("Received null content from OpenAI (async).")
            return content
        except Exception as e:
            logger.error(f"OpenAI async API error ({type(e).__name__}): {e}", exc_info=True)
            raise


class GeminiConfig(BaseModel):
    """Configuration for Gemini provider."""

    api_key: str = Field(description="The API key to use")
    model: str = Field(default="gemini-2.5-pro-preview-03-25", description="The model to use")

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        """Create a config from environment variables."""
        return cls(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-pro-preview-03-25"),
        )
