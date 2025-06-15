#!/usr/bin/env python3
"""
Test script to verify OpenRouter API connection and Qwen model availability.
"""

import os
import sys
import asyncio
import httpx

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

async def test_api_connection():
    """Test the OpenRouter API connection."""
    
    # Set the API key
    api_key = "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"
    os.environ["OPENROUTER_API_KEY"] = api_key
    
    print("=== Testing OpenRouter API Connection ===")
    print(f"API Key: {api_key[:20]}...")
    
    try:
        # Test 1: Check if we can import the client
        print("\n1. Testing imports...")
        from app.api.openrouter_client import OpenRouterClient
        print("✓ OpenRouterClient imported successfully")
        
        # Test 2: Create client instance
        print("\n2. Creating client instance...")
        client = OpenRouterClient(api_key=api_key)
        print("✓ Client created successfully")
        
        # Test 3: Test simple API call with Qwen model
        print("\n3. Testing API call with Qwen model...")
        model_id = "qwen/qwen2.5-vl-32b-instruct"
        
        messages = [
            {
                "role": "user",
                "content": "Hello! Can you solve this simple math problem: 2 + 2 = ?"
            }
        ]
        
        print(f"Using model: {model_id}")
        print("Sending test message...")
        
        response = await client.chat_completion(
            model=model_id,
            messages=messages,
            temperature=0.1,
            max_tokens=100
        )
        
        print("✓ API call successful!")
        print(f"Response: {response.get('choices', [{}])[0].get('message', {}).get('content', 'No content')[:100]}...")
        
        # Test 4: Check usage and cost
        usage = response.get('usage', {})
        print(f"\nUsage stats:")
        print(f"  Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
        print(f"  Completion tokens: {usage.get('completion_tokens', 'N/A')}")
        print(f"  Total tokens: {usage.get('total_tokens', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'client' in locals():
            await client.close()

async def test_direct_api():
    """Test direct API call to OpenRouter."""
    
    api_key = "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"
    
    print("\n=== Testing Direct API Call ===")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "EGE Math Solution Checker"
    }
    
    data = {
        "model": "qwen/qwen2.5-vl-32b-instruct",
        "messages": [
            {
                "role": "user",
                "content": "Hello! Test message. What is 5 + 3?"
            }
        ],
        "temperature": 0.1,
        "max_tokens": 50
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            print("Sending direct API request...")
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✓ Direct API call successful!")
                print(f"Response: {result.get('choices', [{}])[0].get('message', {}).get('content', 'No content')}")
                
                usage = result.get('usage', {})
                print(f"Usage: {usage}")
                return True
            else:
                print(f"✗ API call failed with status {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"✗ Direct API error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("Testing OpenRouter API connection and Qwen model...")
    
    # Run tests
    try:
        # Test 1: Direct API call
        success1 = asyncio.run(test_direct_api())
        
        # Test 2: Through our client
        success2 = asyncio.run(test_api_connection())
        
        if success1 and success2:
            print("\n" + "="*50)
            print("✓ ALL TESTS PASSED!")
            print("API connection is working correctly.")
            print("Ready to run full evaluation.")
        else:
            print("\n" + "="*50)
            print("✗ SOME TESTS FAILED!")
            print("Please check the errors above.")
            
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
