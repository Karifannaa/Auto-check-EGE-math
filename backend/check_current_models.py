"""
Script to check all currently available models on OpenRouter.
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from app.api.openrouter_client import OpenRouterClient

async def check_current_models():
    """Check all available models on OpenRouter."""
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
        
        # Print all models with their full IDs
        print("\nAll available models:")
        print("====================")
        for model in models.get("data", []):
            model_id = model.get("id", "")
            model_name = model.get("name", "")
            context_length = model.get("context_length", "Unknown")
            pricing = model.get("pricing", {})
            
            # Check if there's a free version
            free_version = ":free" in model_id
            
            print(f"ID: {model_id}")
            print(f"Name: {model_name}")
            print(f"Context Length: {context_length}")
            print(f"Free Version: {free_version}")
            print(f"Pricing: {json.dumps(pricing, indent=2)}")
            print("---------------------")
        
        # Save the full model list to a file for reference
        with open("openrouter_models.json", "w") as f:
            json.dump(models, f, indent=2)
        print(f"Full model list saved to openrouter_models.json")
    
    finally:
        # Close client
        await client.close()

if __name__ == "__main__":
    asyncio.run(check_current_models())
