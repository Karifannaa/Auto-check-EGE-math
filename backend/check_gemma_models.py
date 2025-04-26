"""
Script to check available Gemma 3 models on OpenRouter.
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from app.api.openrouter_client import OpenRouterClient

async def check_gemma_models():
    """Check available Gemma 3 models on OpenRouter."""
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        print("Error: OpenRouter API key not configured")
        return
    
    # Create client
    client = OpenRouterClient(
        api_key=api_key,
        site_url="http://localhost:3000",
        site_name="EGE Math Solution Checker"
    )
    
    try:
        # Get models
        models = await client.list_models()
        
        # Filter for Gemma 3 models
        gemma_models = [model for model in models.get("data", []) if "gemma-3" in model.get("id", "").lower()]
        
        # Print Gemma 3 models
        print("\nFound Gemma 3 models:")
        print("=====================")
        for model in gemma_models:
            print(f"ID: {model.get('id')}")
            print(f"Name: {model.get('name')}")
            print(f"Description: {model.get('description', 'No description')}")
            print(f"Context Length: {model.get('context_length', 'Unknown')}")
            print(f"Pricing: {json.dumps(model.get('pricing', {}), indent=2)}")
            print("---------------------")
        
        # Print all models for reference
        print("\nAll available models:")
        print("====================")
        for model in models.get("data", []):
            print(f"- {model.get('id')}: {model.get('name')}")
    
    finally:
        # Close client
        await client.close()

if __name__ == "__main__":
    asyncio.run(check_gemma_models())
