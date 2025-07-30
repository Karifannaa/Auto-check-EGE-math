"""
OpenRouter API Client Module

This module provides a client for interacting with the OpenRouter API,
which gives access to various AI models like o3, o4-mini, Deepseek R1,
and distilled qwen-2.5 models.
"""

import json
import logging
from typing import Dict, List, Optional, Union, Any
import httpx
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define models for request and response
class Message(BaseModel):
    """Message model for chat completions."""
    role: str
    content: str

class ImageContent(BaseModel):
    """Image content model for multimodal messages."""
    type: str = "image_url"
    image_url: Dict[str, str]

class MultiModalContent(BaseModel):
    """Content model that can be text or image."""
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

class MultiModalMessage(BaseModel):
    """Message model that supports multimodal content."""
    role: str
    content: Union[str, List[Union[str, ImageContent, Dict[str, Any]]]]

class ChatCompletionRequest(BaseModel):
    """Request model for chat completions."""
    model: str
    messages: List[Union[Message, MultiModalMessage]]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 10000
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

class OpenRouterClient:
    """Client for interacting with the OpenRouter API."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, site_url: Optional[str] = None, site_name: Optional[str] = None):
        """
        Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key
            site_url: Optional site URL for rankings on openrouter.ai
            site_name: Optional site name for rankings on openrouter.ai
        """
        self.api_key = api_key
        self.site_url = site_url
        self.site_name = site_name
        self.client = httpx.AsyncClient(timeout=120.0)  # Longer timeout for model inference

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for API requests.

        Returns:
            Dict with authorization and optional site information headers
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self.site_url:
            headers["HTTP-Referer"] = self.site_url

        if self.site_name:
            headers["X-Title"] = self.site_name

        return headers

    async def chat_completion(
        self,
        model: str,
        messages: List[Union[Dict[str, Any], Message, MultiModalMessage]],
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: Optional[int] = 10000,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion using the OpenRouter API.

        Args:
            model: Model identifier (e.g., "openai/gpt-4o", "anthropic/claude-3-opus")
            messages: List of messages in the conversation
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            stop: List of strings that will stop generation when encountered
            **kwargs: Additional parameters to pass to the API

        Returns:
            API response as a dictionary
        """
        url = f"{self.BASE_URL}/chat/completions"

        # Convert Message objects to dictionaries if needed
        processed_messages = []
        for msg in messages:
            if isinstance(msg, (Message, MultiModalMessage)):
                processed_messages.append(msg.model_dump())
            else:
                processed_messages.append(msg)

        payload = {
            "model": model,
            "messages": processed_messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if stop is not None:
            payload["stop"] = stop

        # Add any additional parameters
        payload.update(kwargs)

        logger.info(f"Sending request to {model}")

        try:
            response = await self.client.post(
                url,
                headers=self._get_headers(),
                json=payload
            )
            response.raise_for_status()

            if stream:
                return response  # Return the response object for streaming
            else:
                return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise

    async def process_chat_stream(self, response: httpx.Response):
        """
        Process a streaming chat completion response.

        Args:
            response: Streaming response from the API

        Yields:
            Parsed chunks from the stream
        """
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                line = line[6:]  # Remove "data: " prefix

                if line.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(line)
                    yield chunk
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON from chunk: {line}")

    async def list_models(self) -> Dict[str, Any]:
        """
        List available models from OpenRouter.

        Returns:
            Dictionary containing available models information
        """
        url = f"{self.BASE_URL}/models"

        try:
            response = await self.client.get(
                url,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise

    async def get_credits(self) -> Dict[str, Any]:
        """
        Get account credits information.

        Returns:
            Dictionary containing credit information
        """
        url = f"{self.BASE_URL}/auth/credits"

        try:
            response = await self.client.get(
                url,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
