"""
Test script for Task 14 prompt with examples.

This script tests the functionality of the Task 14 prompt with examples.
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
        logging.FileHandler(f"backend/logs/task14_prompt_test_{Path(__file__).stem}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TestTask14Prompt(unittest.TestCase):
    """Test case for Task 14 prompt functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.prompt_generator = PromptGenerator()
        
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='white')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        img_data = img_byte_arr.read()
        
        # Prepare images for API
        self.student_image_data = prepare_image_for_api(Image.open(io.BytesIO(img_data)))
        self.correct_image_data = prepare_image_for_api(Image.open(io.BytesIO(img_data)))

    def test_task14_basic_prompt(self):
        """Test the basic prompt for Task 14."""
        messages = self.prompt_generator.create_messages_with_image(
            task_type="task_14",
            task_description="Test task description",
            student_solution_image=self.student_image_data,
            correct_solution_image=self.correct_image_data,
            include_examples=False,
            prompt_variant="basic"
        )
        
        # Check that we have a system message and a user message
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        
        # Check the content of the user message
        user_content = messages[1]["content"]
        self.assertIsInstance(user_content, list)
        
        # The first item should be the prompt text
        self.assertEqual(user_content[0]["type"], "text")
        self.assertIn("Test task description", user_content[0]["text"])
        
        # The second item should be the correct solution image
        self.assertEqual(user_content[1]["type"], "image_url")
        
        # The fourth item should be the student solution image
        self.assertEqual(user_content[3]["type"], "image_url")

    def test_task14_detailed_prompt(self):
        """Test the detailed prompt for Task 14."""
        messages = self.prompt_generator.create_messages_with_image(
            task_type="task_14",
            task_description="Test task description",
            student_solution_image=self.student_image_data,
            correct_solution_image=self.correct_image_data,
            include_examples=False,
            prompt_variant="detailed"
        )
        
        # Check that we have a system message and a user message
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        
        # Check the content of the user message
        user_content = messages[1]["content"]
        self.assertIsInstance(user_content, list)
        
        # The first item should be the prompt text
        self.assertEqual(user_content[0]["type"], "text")
        self.assertIn("Test task description", user_content[0]["text"])
        self.assertIn("Критерии оценивания задания 14", user_content[0]["text"])
        
        # The second item should be the correct solution image
        self.assertEqual(user_content[1]["type"], "image_url")
        
        # The fourth item should be the student solution image
        self.assertEqual(user_content[3]["type"], "image_url")

    def test_task14_image_examples_prompt(self):
        """Test the image examples prompt for Task 14."""
        # Check if the examples directory exists
        examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "14")
        if not os.path.exists(examples_dir):
            self.skipTest("Examples directory does not exist")

        messages = self.prompt_generator.create_messages_with_image(
            task_type="task_14",
            task_description="Test task description",
            student_solution_image=self.student_image_data,
            correct_solution_image=self.correct_image_data,
            include_examples=True,
            prompt_variant="image_examples"
        )
        
        # Check that we have a system message and a user message
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        
        # Check the content of the user message
        user_content = messages[1]["content"]
        self.assertIsInstance(user_content, list)
        
        # The first item should be the prompt text
        self.assertEqual(user_content[0]["type"], "text")
        self.assertIn("Test task description", user_content[0]["text"])
        self.assertIn("Критерии оценивания задания 14", user_content[0]["text"])
        
        # Log the content structure for debugging
        logger.info("Content structure:")
        for i, item in enumerate(user_content):
            if item["type"] == "text":
                logger.info(f"Item {i} (text): {item['text'][:50]}...")
            else:
                logger.info(f"Item {i} (image)")


if __name__ == "__main__":
    unittest.main()
