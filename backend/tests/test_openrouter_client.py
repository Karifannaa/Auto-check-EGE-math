"""
Tests for the OpenRouter API client.
"""

import os
import pytest
import httpx
from unittest.mock import patch, MagicMock

from app.api.openrouter_client import OpenRouterClient, Message


@pytest.mark.asyncio
async def test_client_initialization():
    """Test that the client initializes correctly."""
    client = OpenRouterClient(api_key="test_key", site_url="http://test.com", site_name="Test App")
    assert client.api_key == "test_key"
    assert client.site_url == "http://test.com"
    assert client.site_name == "Test App"
    await client.close()


@pytest.mark.asyncio
async def test_get_headers():
    """Test that headers are correctly generated."""
    client = OpenRouterClient(api_key="test_key", site_url="http://test.com", site_name="Test App")
    headers = client._get_headers()
    
    assert headers["Authorization"] == "Bearer test_key"
    assert headers["Content-Type"] == "application/json"
    assert headers["HTTP-Referer"] == "http://test.com"
    assert headers["X-Title"] == "Test App"
    
    await client.close()


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_chat_completion(mock_post):
    """Test that chat completion works correctly."""
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": "test_id",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "openai/gpt-4o",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "This is a test response."
                },
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response
    
    # Create client and call chat_completion
    client = OpenRouterClient(api_key="test_key")
    messages = [
        Message(role="user", content="This is a test message.")
    ]
    
    response = await client.chat_completion(
        model="openai/gpt-4o",
        messages=messages,
        temperature=0.7
    )
    
    # Check that the response is correct
    assert response["choices"][0]["message"]["content"] == "This is a test response."
    assert response["usage"]["total_tokens"] == 30
    
    # Check that the request was made correctly
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[1]["headers"]["Authorization"] == "Bearer test_key"
    assert call_args[1]["json"]["model"] == "openai/gpt-4o"
    assert call_args[1]["json"]["temperature"] == 0.7
    
    await client.close()


@pytest.mark.asyncio
@patch("httpx.AsyncClient.get")
async def test_list_models(mock_get):
    """Test that list_models works correctly."""
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {
                "id": "openai/gpt-4o",
                "name": "GPT-4o",
                "description": "OpenAI's GPT-4o model"
            },
            {
                "id": "anthropic/claude-3-opus",
                "name": "Claude 3 Opus",
                "description": "Anthropic's Claude 3 Opus model"
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response
    
    # Create client and call list_models
    client = OpenRouterClient(api_key="test_key")
    models = await client.list_models()
    
    # Check that the response is correct
    assert len(models["data"]) == 2
    assert models["data"][0]["id"] == "openai/gpt-4o"
    assert models["data"][1]["name"] == "Claude 3 Opus"
    
    # Check that the request was made correctly
    mock_get.assert_called_once()
    call_args = mock_get.call_args
    assert call_args[1]["headers"]["Authorization"] == "Bearer test_key"
    
    await client.close()
