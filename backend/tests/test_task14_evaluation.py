"""
Test script for Task 14 evaluation.

This script tests the evaluation of Task 14 (stereometry) solutions
using reasoning models through the OpenRouter API.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from PIL import Image
import io

# Add the parent directory to the path so we can import from app
sys.path.append(str(Path(__file__).parent.parent))

from app.utils.prompt_utils import PromptGenerator
from app.utils.image_utils import prepare_image_for_api
from app.api.openrouter_client import OpenRouterClient
from app.core.config import settings

# Configure logging
log_filename = f"logs/task14_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def test_task14_evaluation():
    """Test the evaluation of a Task 14 solution."""
    # Initialize the OpenRouter client
    client = OpenRouterClient(
        api_key=settings.OPENROUTER_API_KEY,
        site_url=settings.SITE_URL,
        site_name=settings.SITE_NAME
    )

    try:
        # Load test images
        example_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "examples", "14", "first_example", "test_example_0.png")

        if not os.path.exists(example_path):
            logger.error(f"Example image not found at {example_path}")
            return

        with open(example_path, "rb") as f:
            img_data = f.read()

        # Prepare images for API
        img = Image.open(io.BytesIO(img_data))
        student_image_data = prepare_image_for_api(img)

        # Initialize the prompt generator
        prompt_generator = PromptGenerator()

        # Create messages with a single image containing task, correct solution and student solution
        messages = prompt_generator.create_messages_with_image(
            task_type="task_14",
            task_description="Стереометрическая задача с доказательством и вычислением. Изображение содержит условие задачи, правильное решение и решение ученика.",
            student_solution_image=student_image_data,
            correct_solution_image=None,  # Не используем отдельное изображение для правильного решения
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
        model_id = "qwen/qwen2.5-vl-32b-instruct:free"

        logger.info(f"Using model: {model_id}")

        # Send the request to the API
        response = await client.chat_completion(
            model=model_id,
            messages=messages,
            temperature=0.2,  # Lower temperature for more deterministic results
            max_tokens=10000   # Ensure we have enough tokens for a complete response
        )

        # Log the full response
        logger.info("Full API response:")
        logger.info(response)

        # Extract and log the model's response
        model_response = response["choices"][0]["message"]["content"]
        logger.info("Model response:")
        logger.info(model_response)

        # Extract the score from the response
        score = None
        for line in model_response.split("\n"):
            if "### Итоговая оценка" in line:
                # Look for the next line with the score
                score_line_index = model_response.split("\n").index(line) + 1
                if score_line_index < len(model_response.split("\n")):
                    score_line = model_response.split("\n")[score_line_index]
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
    asyncio.run(test_task14_evaluation())
