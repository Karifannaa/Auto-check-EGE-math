"""
Script to test the task 14 prompt with a single image containing task, correct solution and student solution.
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
log_file = os.path.join(logs_dir, f"task14_single_image_{timestamp}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def test_task14_single_image():
    """Test the evaluation of a Task 14 solution using a single image."""
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
        examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "14")
        
        # Use the test example image
        test_image_file = os.path.join(examples_dir, "first_example", "test_example_0.png")
        
        if not os.path.exists(test_image_file):
            logger.error(f"Test image not found at {test_image_file}")
            return
        
        # Load the test image
        with open(test_image_file, "rb") as f:
            img_data = f.read()
        
        student_img = Image.open(io.BytesIO(img_data))
        
        # Log image details before processing
        logger.info(f"Test image details:")
        logger.info(f"Format: {student_img.format}, Size: {student_img.size}, Mode: {student_img.mode}")
        logger.info(f"Original file size: {len(img_data)} bytes")
        
        # Prepare image for API with detailed logging and higher quality
        student_image_data = prepare_image_for_api(
            student_img,
            max_size=2048,  # Increased from 1024 to 2048 for better quality
            enhance=True,   # Enable image enhancement
            contrast_factor=1.3  # Slightly higher contrast for better readability
        )
        
        # Generate the messages with a single image
        messages = prompt_generator.create_messages_with_image(
            task_type="task_14",
            task_description="Стереометрическая задача с доказательством и вычислением. Изображение содержит условие задачи, правильное решение и решение ученика.",
            student_solution_image=student_image_data,
            correct_solution_image=None,  # No separate correct solution image
            include_examples=False,
            prompt_variant="detailed"
        )
        
        # Log the system message
        logger.info("System message:")
        logger.info(messages[0]["content"])
        
        # Log the user message structure
        logger.info("User message content structure:")
        for i, content_item in enumerate(messages[1]["content"]):
            if content_item["type"] == "text":
                logger.info(f"Content item {i} (text): {content_item['text'][:100]}...")
            else:
                logger.info(f"Content item {i} (image): [Image data]")
        
        # Select a free model that supports images
        model_id = "qwen/qwen2.5-vl-32b-instruct:free"  # Free Qwen 2.5 VL model
        
        logger.info(f"Using model: {model_id}")
        
        # Call the API
        response = await client.chat_completion(
            model=model_id,
            messages=messages,
            temperature=0.2,  # Lower temperature for more deterministic results
            max_tokens=10000  # Ensure we have enough tokens for a complete response
        )
        
        # Log the response
        logger.info("API Response:")
        logger.info(json.dumps(response, indent=2))
        
        # Extract and log the evaluation
        result_text = response["choices"][0]["message"]["content"]
        logger.info("Evaluation result:")
        logger.info(result_text)
        
        # Extract the score from the response
        score = None
        for line in result_text.split("\n"):
            if "### Итоговая оценка" in line:
                # Look for the next line with the score
                score_line_index = result_text.split("\n").index(line) + 1
                if score_line_index < len(result_text.split("\n")):
                    score_line = result_text.split("\n")[score_line_index]
                    if "[Оценка:" in score_line:
                        score = score_line.strip()
                        break
        
        logger.info(f"Extracted score: {score}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
    finally:
        # Close the client
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_task14_single_image())
