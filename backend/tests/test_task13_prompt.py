"""
Test script for Task 13 prompt with examples.

This script tests the functionality of the Task 13 prompt with examples.
"""

import os
import sys
import unittest
import json
import logging
from pathlib import Path

# Add the parent directory to the path so we can import from app
sys.path.append(str(Path(__file__).parent.parent))

from app.utils.prompt_utils import PromptGenerator
from app.utils.image_utils import prepare_image_for_api
from app.api.openrouter_client import OpenRouterClient
from app.core.config import settings
from PIL import Image
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("task13_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TestTask13Prompt(unittest.TestCase):
    """Test case for Task 13 prompt with examples."""

    def setUp(self):
        """Set up the test case."""
        self.prompt_generator = PromptGenerator()

        # Create a dummy image for testing
        self.dummy_image = Image.new('RGB', (100, 100), color='white')
        self.student_image_data = prepare_image_for_api(self.dummy_image)
        self.correct_image_data = prepare_image_for_api(self.dummy_image)

    def test_task13_basic_prompt(self):
        """Test the basic prompt for Task 13."""
        messages = self.prompt_generator.create_messages_with_image(
            task_type="task_13",
            task_description="Test task description",
            student_solution_image=self.student_image_data,
            correct_solution_image=self.correct_image_data,
            include_examples=False,
            prompt_variant="basic"
        )

        # Check that we have the expected number of messages
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

        # Check that the user message contains the expected content
        user_content = messages[1]["content"]
        self.assertTrue(isinstance(user_content, list))

        # The first item should be the text prompt
        self.assertEqual(user_content[0]["type"], "text")
        self.assertIn("Test task description", user_content[0]["text"])

        # The second and third items should be the correct solution image and separator
        self.assertEqual(user_content[1]["type"], "image_url")
        self.assertEqual(user_content[2]["type"], "text")

        # The fourth item should be the student solution image
        self.assertEqual(user_content[3]["type"], "image_url")

    def test_task13_image_examples_prompt(self):
        """Test the image examples prompt for Task 13."""
        # Check if the examples directory exists
        examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "13")
        if not os.path.exists(examples_dir):
            self.skipTest("Examples directory does not exist")

        messages = self.prompt_generator.create_messages_with_image(
            task_type="task_13",
            task_description="Test task description",
            student_solution_image=self.student_image_data,
            correct_solution_image=self.correct_image_data,
            include_examples=True,
            prompt_variant="image_examples"
        )

        # Check that we have the expected number of messages
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

        # Check that the user message contains the expected content
        user_content = messages[1]["content"]
        self.assertTrue(isinstance(user_content, list))

        # The first item should be the text prompt
        self.assertEqual(user_content[0]["type"], "text")
        self.assertIn("Test task description", user_content[0]["text"])

        # Check that we have examples
        example_header_found = False
        for item in user_content:
            if item["type"] == "text" and "Примеры решений и их оценок" in item["text"]:
                example_header_found = True
                break

        self.assertTrue(example_header_found, "Examples header not found in prompt")

        # Check that we have the separator between examples and the solution to evaluate
        separator_found = False
        for item in user_content:
            if item["type"] == "text" and "Задание для оценки" in item["text"]:
                separator_found = True
                break

        self.assertTrue(separator_found, "Separator between examples and solution not found")

        # Check that we have the correct solution image and student solution image
        correct_solution_found = False
        student_solution_found = False
        for item in user_content:
            if item["type"] == "image_url":
                if not correct_solution_found:
                    correct_solution_found = True
                else:
                    student_solution_found = True

        self.assertTrue(correct_solution_found, "Correct solution image not found")
        self.assertTrue(student_solution_found, "Student solution image not found")


    async def test_task13_evaluation(self):
        """Test the evaluation of a Task 13 solution using the OpenRouter API."""
        # Skip this test if no API key is provided
        if not settings.OPENROUTER_API_KEY:
            self.skipTest("OpenRouter API key not configured")

        # Create an OpenRouter client
        client = OpenRouterClient(
            api_key=settings.OPENROUTER_API_KEY,
            site_url=settings.SITE_URL,
            site_name=settings.SITE_NAME
        )

        try:
            # Load a real example image for testing
            examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "13")
            first_example_dir = os.path.join(examples_dir, "first_example")

            if not os.path.exists(first_example_dir):
                self.skipTest("Example image not found")

            # Use the first example as the student solution for testing
            image_file = os.path.join(first_example_dir, "1.png")
            with open(image_file, "rb") as f:
                img_data = f.read()

            img = Image.open(io.BytesIO(img_data))
            student_image_data = prepare_image_for_api(img)

            # Generate the messages
            messages = self.prompt_generator.create_messages_with_image(
                task_type="task_13",
                task_description="",
                student_solution_image=student_image_data,
                include_examples=True,
                prompt_variant="image_examples"
            )

            # Log the full messages
            logger.info("System message:")
            logger.info(messages[0]["content"])

            logger.info("User message content structure:")
            for i, content_item in enumerate(messages[1]["content"]):
                if content_item["type"] == "text":
                    logger.info(f"Content item {i} (text): {content_item['text']}")
                else:
                    logger.info(f"Content item {i} (image): [Image data]")

            # Call the API
            model_name = "openai/gpt-4o-mini"
            logger.info(f"Calling API with model: {model_name}")

            response = await client.chat_completion(
                model=model_name,
                messages=messages,
                temperature=0.7
            )

            # Log the response
            logger.info("API Response:")
            logger.info(json.dumps(response, indent=2))

            # Extract and log the evaluation
            result_text = response["choices"][0]["message"]["content"]
            logger.info("Evaluation result:")
            logger.info(result_text)

            # Check that the response contains an evaluation
            self.assertIn("Анализ решения пункта а)", result_text)
            self.assertIn("Анализ решения пункта б)", result_text)
            self.assertIn("Итоговая оценка", result_text)

        finally:
            # Close the client
            await client.close()


if __name__ == "__main__":
    import asyncio

    # Run the async test
    async def run_tests():
        test = TestTask13Prompt()
        test.setUp()
        await test.test_task13_evaluation()

    asyncio.run(run_tests())

    # Run the regular tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
