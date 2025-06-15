#!/usr/bin/env python3
"""
Test script to verify the new OpenRouter API key works.
"""

import os
import sys
import asyncio
import httpx

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

async def test_new_api_key():
    """Test the new OpenRouter API key."""
    
    # Set the NEW API key
    api_key = "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"
    os.environ["OPENROUTER_API_KEY"] = api_key
    
    print("=== Testing New OpenRouter API Key ===")
    print(f"API Key: {api_key[:20]}...")
    
    try:
        # Test 1: Direct API call
        print("\n1. Testing direct API call...")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "EGE Math Solution Checker"
        }
        
        data = {
            "model": "qwen/qwen2.5-vl-72b-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello! Test message. What is 5 + 3?"
                }
            ],
            "temperature": 0.1,
            "max_tokens": 50
        }
        
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
        
        # Test 2: Through our client
        print("\n2. Testing through our client...")
        from app.api.openrouter_client import OpenRouterClient
        
        client = OpenRouterClient(api_key=api_key)
        
        messages = [
            {
                "role": "user",
                "content": "Hello! Can you solve this simple math problem: 2 + 2 = ?"
            }
        ]
        
        print("Sending test message through client...")
        
        response = await client.chat_completion(
            model="qwen/qwen2.5-vl-72b-instruct",
            messages=messages,
            temperature=0.1,
            max_tokens=100
        )
        
        print("✓ Client API call successful!")
        print(f"Response: {response.get('choices', [{}])[0].get('message', {}).get('content', 'No content')[:100]}...")
        
        # Check usage and cost
        usage = response.get('usage', {})
        print(f"\nUsage stats:")
        print(f"  Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
        print(f"  Completion tokens: {usage.get('completion_tokens', 'N/A')}")
        print(f"  Total tokens: {usage.get('total_tokens', 'N/A')}")
        
        await client.close()
        return True
                
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("Testing new OpenRouter API key with Qwen 2.5 VL 72B model...")
    
    # Run tests
    try:
        success = asyncio.run(test_new_api_key())
        
        if success:
            print("\n" + "="*50)
            print("✓ ALL TESTS PASSED!")
            print("New API key is working correctly.")
            print("Ready to run full evaluation.")
        else:
            print("\n" + "="*50)
            print("✗ TESTS FAILED!")
            print("Please check the API key and try again.")
            
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
