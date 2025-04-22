"""
Script to test the task 13 prompt with a real example.
"""

import os
import asyncio
import json
import logging
from PIL import Image
import io

from app.utils.prompt_utils import PromptGenerator
from app.utils.image_utils import prepare_image_for_api
from app.api.openrouter_client import OpenRouterClient
from app.core.config import settings

# Configure logging with timestamp
from datetime import datetime
import os

# Create logs directory if it doesn't exist
logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)

# Create log file with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(logs_dir, f"task13_evaluation_{timestamp}.log")

# Create a formatter that includes more details
detailed_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create file handler for detailed logging
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(detailed_formatter)

# Create console handler with a simpler format
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Get a logger for this module
logger = logging.getLogger(__name__)

async def test_task13_evaluation():
    """Test the evaluation of a Task 13 solution using the OpenRouter API."""
    # Check if API key is configured
    if not settings.OPENROUTER_API_KEY:
        logger.error("OpenRouter API key not configured")
        return

    # Create an OpenRouter client
    client = OpenRouterClient(
        api_key=settings.OPENROUTER_API_KEY,
        site_url=settings.SITE_URL,
        site_name=settings.SITE_NAME
    )

    try:
        # Create a prompt generator
        prompt_generator = PromptGenerator()

        # Load the example image for testing
        examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "13")
        evaluation_dir = os.path.join(examples_dir, "evaluation")

        if not os.path.exists(evaluation_dir):
            logger.error("Evaluation directory not found")
            return

        # We're not using the correct solution image in this test
        # correct_solution_file = os.path.join(evaluation_dir, "correct_solution_13.2.png")

        # Try different possible student solution filenames
        possible_filenames = [
            "example_1point_without_correct_answer.png",
            "example_1point.png"
        ]

        student_solution_file = None
        for filename in possible_filenames:
            file_path = os.path.join(evaluation_dir, filename)
            if os.path.exists(file_path):
                student_solution_file = file_path
                logger.info(f"Using student solution file: {filename}")
                break

        if student_solution_file is None:
            logger.error(f"No student solution file found in {evaluation_dir}")
            logger.error(f"Tried: {', '.join(possible_filenames)}")
            return

        # We're running without the correct solution image
        correct_solution_image_data = None
        logger.info("Running test without correct solution image")

        # Load the student solution image
        # (We've already checked that the file exists)
        with open(student_solution_file, "rb") as f:
            student_img_data = f.read()

        student_img = Image.open(io.BytesIO(student_img_data))

        # Log image details before processing
        logger.info(f"Student solution image details:")
        logger.info(f"Format: {student_img.format}, Size: {student_img.size}, Mode: {student_img.mode}")
        logger.info(f"Original file size: {len(student_img_data)} bytes")

        # Prepare image for API with detailed logging and higher quality
        # Use a balanced max_size to maintain quality while keeping token count reasonable
        student_image_data = prepare_image_for_api(
            student_img,
            max_size=2048,  # Increased from 1024 to 2048 for better quality
            enhance=True,   # Enable image enhancement
            contrast_factor=1.3  # Slightly higher contrast for better readability
        )

        # Log processed image details
        if "image_url" in student_image_data and "url" in student_image_data["image_url"]:
            image_url = student_image_data["image_url"]["url"]
            if "data" in image_url:
                # Extract base64 data
                base64_data = image_url.split(",")[1]
                logger.info(f"Processed image details:")
                logger.info(f"Base64 length: {len(base64_data)} characters")
                logger.info(f"Estimated base64 token count: {len(base64_data) // 3} tokens (rough estimate)")

        # Let's also check the prompt_utils.py file to see if we need to modify how examples are processed
        logger.info("Checking prompt_utils.py for image processing in examples...")

        # Generate the messages
        # We'll use a smaller max_size for all images to reduce token count
        messages = prompt_generator.create_messages_with_image(
            task_type="task_13",
            task_description="",
            student_solution_image=student_image_data,
            correct_solution_image=correct_solution_image_data,  # Include correct solution if available
            include_examples=False,  # Disable examples
            prompt_variant="detailed"  # Use detailed prompt instead of image_examples
        )

        # Count the number of images in the prompt
        image_count_in_prompt = sum(1 for item in messages[1]["content"] if item["type"] == "image_url")
        logger.info(f"Total number of images in the prompt: {image_count_in_prompt}")

        # Log the system message
        logger.info("System message:")
        logger.info(messages[0]["content"])

        # Log the user message content structure
        logger.info("User message content structure:")
        image_count = 0
        total_estimated_image_tokens = 0
        text_token_estimate = 0

        # Check how example images are processed
        logger.info("Checking example image processing:")
        examples_found = False
        for i, content_item in enumerate(messages[1]["content"]):
            if content_item["type"] == "text":
                # Rough estimate of text tokens (1 token ≈ 4 characters for English, less for other languages)
                text_length = len(content_item["text"])
                estimated_text_tokens = text_length // 3  # Conservative estimate
                text_token_estimate += estimated_text_tokens

                logger.info(f"Content item {i} (text): {content_item['text'][:100]}... - Estimated tokens: {estimated_text_tokens}")

                # Check if this is the examples section
                if "Примеры решений и их оценок" in content_item["text"]:
                    examples_found = True
                    logger.info("Examples section found in the prompt")
            else:
                image_count += 1
                # Estimate token count for the image (rough estimate based on OpenAI's documentation)
                # Images are typically counted as ~85 tokens for low-res and ~255 tokens for high-res
                # We'll use a middle ground estimate of 170 tokens per image
                estimated_tokens = 170
                total_estimated_image_tokens += estimated_tokens

                # Check if this is part of the examples
                if examples_found:
                    logger.info(f"Content item {i} (example image): [Image data] - Estimated tokens: {estimated_tokens}")
                else:
                    logger.info(f"Content item {i} (image): [Image data] - Estimated tokens: {estimated_tokens}")

        # Log token usage summary
        total_estimated_tokens = text_token_estimate + total_estimated_image_tokens
        logger.info(f"\nToken usage summary:")
        logger.info(f"Total images: {image_count}")
        logger.info(f"Estimated text tokens: {text_token_estimate}")
        logger.info(f"Estimated image tokens: {total_estimated_image_tokens}")
        logger.info(f"Total estimated tokens: {total_estimated_tokens}")

        # Print the full prompt before sending the request
        logger.info("FULL PROMPT:")
        for msg in messages:
            logger.info(f"Role: {msg['role']}")
            if isinstance(msg['content'], list):
                for item in msg['content']:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        logger.info(f"Content (text): {item['text']}")
                    elif isinstance(item, dict) and item.get('type') == 'image_url':
                        logger.info(f"Content (image): [Image data]")
            else:
                logger.info(f"Content: {msg['content']}")

        # Call the API
        model_name = "moonshotai/kimi-vl-a3b-thinking:free"  # Use the Moonshot Kimi model
        logger.info(f"Calling API with model: {model_name}")

        response = await client.chat_completion(
            model=model_name,
            messages=messages,
            temperature=0.7,
            extra_body={"thinking": True}  # Enable thinking for more detailed reasoning
        )

        # Log the response
        logger.info("API Response:")
        logger.info(json.dumps(response, indent=2))

        # Extract and log the evaluation
        result_text = response["choices"][0]["message"]["content"]
        logger.info("Evaluation result:")
        logger.info(result_text)

        # Log actual token usage from the API response
        if "usage" in response:
            logger.info("\nActual token usage from API:")
            logger.info(f"Prompt tokens: {response['usage'].get('prompt_tokens', 'N/A')}")
            logger.info(f"Completion tokens: {response['usage'].get('completion_tokens', 'N/A')}")
            logger.info(f"Total tokens: {response['usage'].get('total_tokens', 'N/A')}")

            # Compare with our estimate
            if 'prompt_tokens' in response['usage']:
                actual_prompt_tokens = response['usage']['prompt_tokens']
                logger.info(f"\nComparison with our estimate:")
                logger.info(f"Our estimated prompt tokens: {total_estimated_tokens}")
                logger.info(f"Actual prompt tokens: {actual_prompt_tokens}")
                logger.info(f"Difference: {actual_prompt_tokens - total_estimated_tokens} tokens")
                logger.info(f"Accuracy: {(total_estimated_tokens / actual_prompt_tokens) * 100:.2f}%")

    finally:
        # Close the client
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_task13_evaluation())
