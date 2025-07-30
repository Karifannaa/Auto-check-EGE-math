"""
Prompt Utilities Module

This module provides utilities for creating effective prompts for
evaluating math exam solutions using reasoning models.
"""

from typing import Dict, List, Optional, Union, Any
from app.utils.specialized_prompts import (
    TASK_13_PROMPTS,
    TASK_14_PROMPTS,
    TASK_15_PROMPTS,
    TASK_16_PROMPTS,
    TASK_17_PROMPTS,
    TASK_18_PROMPTS,
    TASK_19_PROMPTS,
    DEFAULT_PROMPT_VARIANTS,
    AVAILABLE_PROMPT_VARIANTS
)


class PromptGenerator:
    """Class for generating prompts for evaluating math exam solutions."""

    def __init__(self):
        """Initialize the prompt generator."""
        # Base templates for different task types
        self.templates = {
            "task_13": self._get_task_13_template(),
            "task_14": self._get_task_14_template(),
            "task_15": self._get_task_15_template(),
            "task_16": self._get_task_16_template(),
            "task_17": self._get_task_17_template(),
            "task_18": self._get_task_18_template(),
            "task_19": self._get_task_19_template(),
        }

    def _get_task_13_template(self, variant: str = "basic") -> str:
        """Get template for Task 13 (trigonometric, logarithmic, or exponential equations).

        Args:
            variant: Prompt variant to use ("basic", "detailed", "with_examples", or "image_examples")

        Returns:
            Template string for the specified variant
        """
        if variant in TASK_13_PROMPTS:
            return TASK_13_PROMPTS[variant]
        else:
            # Fall back to basic prompt if variant not found
            return TASK_13_PROMPTS["basic"]

    def _get_task_14_template(self, variant: str = "basic") -> str:
        """Get template for Task 14 (stereometry).

        Args:
            variant: Prompt variant to use ("basic", "detailed", "with_examples", or "image_examples")

        Returns:
            Template string for the specified variant
        """
        if variant in TASK_14_PROMPTS:
            return TASK_14_PROMPTS[variant]
        else:
            # Fall back to basic prompt if variant not found
            return TASK_14_PROMPTS["basic"]

    def _get_task_15_template(self, variant: str = "basic") -> str:
        """Get template for Task 15 (inequalities).

        Args:
            variant: Prompt variant to use ("basic", "detailed", or "with_solution")

        Returns:
            Template string for the specified variant
        """
        if variant in TASK_15_PROMPTS:
            return TASK_15_PROMPTS[variant]
        else:
            # Fall back to basic prompt if variant not found
            return TASK_15_PROMPTS["basic"]

    def _get_task_16_template(self, variant: str = "basic") -> str:
        """Get template for Task 16 (economic problems).

        Args:
            variant: Prompt variant to use ("basic", "detailed", "with_solution", or "image_examples")

        Returns:
            Template string for the specified variant
        """
        if variant in TASK_16_PROMPTS:
            return TASK_16_PROMPTS[variant]
        else:
            # Fall back to basic prompt if variant not found
            return TASK_16_PROMPTS["basic"]

    def _get_task_17_template(self, variant: str = "basic") -> str:
        """Get template for Task 17 (planimetry with proof).

        Args:
            variant: Prompt variant to use ("basic", "detailed", "with_solution", or "image_examples")

        Returns:
            Template string for the specified variant
        """
        if variant in TASK_17_PROMPTS:
            return TASK_17_PROMPTS[variant]
        else:
            # Fall back to basic prompt if variant not found
            return TASK_17_PROMPTS["basic"]

    def _get_task_18_template(self, variant: str = "basic") -> str:
        """Get template for Task 18 (parameter problems).

        Args:
            variant: Prompt variant to use ("basic", "detailed", "with_solution", or "image_examples")

        Returns:
            Template string for the specified variant
        """
        if variant in TASK_18_PROMPTS:
            return TASK_18_PROMPTS[variant]
        else:
            # Fall back to basic prompt if variant not found
            return TASK_18_PROMPTS["basic"]

    def _get_task_19_template(self, variant: str = "basic") -> str:
        """Get template for Task 19 (number theory).

        Args:
            variant: Prompt variant to use ("basic", "detailed", "with_solution", or "image_examples")

        Returns:
            Template string for the specified variant
        """
        if variant in TASK_19_PROMPTS:
            return TASK_19_PROMPTS[variant]
        else:
            # Fall back to basic prompt if variant not found
            return TASK_19_PROMPTS["basic"]

    def get_prompt_text(
        self,
        task_type: str,
        prompt_variant: Optional[str] = None
    ) -> str:
        """
        Get the prompt text for a specific task type and variant without formatting.

        Args:
            task_type: Type of task (e.g., "task_13", "task_17")
            prompt_variant: Specific prompt variant to use (e.g., "basic", "detailed")

        Returns:
            Raw prompt text for the specified task type and variant
        """
        if task_type not in self.templates:
            raise ValueError(f"Unknown task type: {task_type}")

        # Determine which prompt variant to use
        if prompt_variant is None:
            # Use default variant for this task type
            prompt_variant = DEFAULT_PROMPT_VARIANTS.get(task_type, "basic")

        # All task types now have specialized prompts with variants
        if task_type == "task_13":
            template = self._get_task_13_template(prompt_variant)
        elif task_type == "task_14":
            template = self._get_task_14_template(prompt_variant)
        elif task_type == "task_15":
            template = self._get_task_15_template(prompt_variant)
        elif task_type == "task_16":
            template = self._get_task_16_template(prompt_variant)
        elif task_type == "task_17":
            template = self._get_task_17_template(prompt_variant)
        elif task_type == "task_18":
            template = self._get_task_18_template(prompt_variant)
        elif task_type == "task_19":
            template = self._get_task_19_template(prompt_variant)
        else:
            # Fallback for unknown task types
            template = self.templates.get(task_type, "Unknown task type: {task_description}")

        return template

    def generate_prompt(
        self,
        task_type: str,
        task_description: str,
        include_examples: bool = False,
        examples: Optional[List[Dict[str, Any]]] = None,
        prompt_variant: Optional[str] = None
    ) -> str:
        """
        Generate a prompt for evaluating a math exam solution.

        Args:
            task_type: Type of task (e.g., "task_13", "task_17")
            task_description: Description of the task
            include_examples: Whether to include examples in the prompt
            examples: List of example solutions with evaluations
            prompt_variant: Specific prompt variant to use (e.g., "basic", "detailed")

        Returns:
            Formatted prompt string
        """
        if task_type not in self.templates:
            raise ValueError(f"Unknown task type: {task_type}")

        # Determine which prompt variant to use
        if prompt_variant is None:
            # Use default variant for this task type
            prompt_variant = DEFAULT_PROMPT_VARIANTS.get(task_type, "basic")

        # All task types now have specialized prompts with variants
        if task_type == "task_13":
            # If include_examples is True, use the image_examples variant
            if include_examples and "image_examples" in AVAILABLE_PROMPT_VARIANTS.get(task_type, []):
                template = self._get_task_13_template("image_examples")
                # Set prompt_variant to image_examples to avoid adding examples twice
                prompt_variant = "image_examples"
            else:
                template = self._get_task_13_template(prompt_variant)
        elif task_type == "task_14":
            # If include_examples is True, use the image_examples variant
            if include_examples and "image_examples" in AVAILABLE_PROMPT_VARIANTS.get(task_type, []):
                template = self._get_task_14_template("image_examples")
                # Set prompt_variant to image_examples to avoid adding examples twice
                prompt_variant = "image_examples"
            else:
                template = self._get_task_14_template(prompt_variant)
        elif task_type == "task_15":
            template = self._get_task_15_template(prompt_variant)
        elif task_type == "task_16":
            # If include_examples is True, use the image_examples variant if available
            if include_examples and "image_examples" in AVAILABLE_PROMPT_VARIANTS.get(task_type, []):
                template = self._get_task_16_template("image_examples")
                prompt_variant = "image_examples"
            else:
                template = self._get_task_16_template(prompt_variant)
        elif task_type == "task_17":
            # If include_examples is True, use the image_examples variant if available
            if include_examples and "image_examples" in AVAILABLE_PROMPT_VARIANTS.get(task_type, []):
                template = self._get_task_17_template("image_examples")
                prompt_variant = "image_examples"
            else:
                template = self._get_task_17_template(prompt_variant)
        elif task_type == "task_18":
            # If include_examples is True, use the image_examples variant if available
            if include_examples and "image_examples" in AVAILABLE_PROMPT_VARIANTS.get(task_type, []):
                template = self._get_task_18_template("image_examples")
                prompt_variant = "image_examples"
            else:
                template = self._get_task_18_template(prompt_variant)
        elif task_type == "task_19":
            # If include_examples is True, use the image_examples variant if available
            if include_examples and "image_examples" in AVAILABLE_PROMPT_VARIANTS.get(task_type, []):
                template = self._get_task_19_template("image_examples")
                prompt_variant = "image_examples"
            else:
                template = self._get_task_19_template(prompt_variant)
        else:
            # Fallback for unknown task types
            template = self.templates.get(task_type, "Unknown task type: {task_description}")

        # Format with task description (use default text if empty)
        if not task_description.strip():
            task_description = "Изображение содержит условие задачи и решение ученика."

        prompt = template.format(task_description=task_description)

        # Add examples if requested and we're not already using a template with examples
        if include_examples and task_type not in ["task_13", "task_14"] and prompt_variant not in ["with_examples", "image_examples"]:
            examples_text = "\n\nПримеры оценивания:\n"
            if examples:
                for i, example in enumerate(examples, 1):
                    examples_text += f"\nПример {i}:\n"
                    examples_text += f"Задача: {example.get('task', '')}\n"
                    examples_text += f"Решение ученика: {example.get('solution', '')}\n"
                    examples_text += f"Оценка: {example.get('score', '')} баллов\n"
                    examples_text += f"Обоснование: {example.get('explanation', '')}\n"
            else:
                # If no examples provided but include_examples is True, add a note
                examples_text += "\nПримеры будут предоставлены в виде изображений.\n"

            prompt += examples_text

        return prompt

    def generate_system_message(self, task_type: str) -> str:
        """
        Generate a system message for the model based on task type.

        Args:
            task_type: Type of task (e.g., "task_13", "task_17")

        Returns:
            System message string
        """
        base_system_message = """
Ты - опытный эксперт по проверке заданий ЕГЭ по математике. Твоя задача - внимательно проанализировать решение ученика и оценить его в соответствии с критериями.

Следуй этим принципам:
1. Будь объективным и справедливым в оценке
2. Внимательно проверяй каждый шаг решения
3. Учитывай все критерии оценивания
4. Предоставляй подробное обоснование своей оценки
5. Структурируй свой ответ четко и понятно

В конце обязательно укажи итоговую оценку в виде числа баллов и подробное обоснование.
"""

        # Add task-specific instructions
        if task_type == "task_13":
            return base_system_message + """
Для задания 13 особенно важно проверить:
- Правильность всех преобразований в пункте а
- Корректность отбора корней в пункте б
- Наличие обоснований для всех шагов решения

КРИТИЧЕСКИ ВАЖНО:
- В ПЕРВУЮ ОЧЕРЕДЬ СРАВНИВАЙ ОТВЕТЫ УЧЕНИКА С ПРАВИЛЬНЫМИ ОТВЕТАМИ!
- ЕСЛИ ОТВЕТ УЧЕНИКА НЕВЕРНЫЙ, ЭТО ОБЯЗАТЕЛЬНО ДОЛЖНО БЫТЬ УЧТЕНО В ОЦЕНКЕ!
- ДАЖЕ ЕСЛИ ВСЕ ПРЕОБРАЗОВАНИЯ ВЕРНЫЕ, НО ОТВЕТ НЕПРАВИЛЬНЫЙ - ЭТО ОШИБКА!

ТЩАТЕЛЬНО ПРОВЕРЯЙ:
- Убедись, что все корни уравнения найдены правильно и полностью
- Проверь, что все корни на указанном отрезке отобраны верно, без пропусков и лишних корней
- НЕ ЗАБУДЬ ОТМЕТИТЬ ВСЕ РАСХОЖДЕНИЯ МЕЖДУ ОТВЕТАМИ УЧЕНИКА И ПРАВИЛЬНЫМИ ОТВЕТАМИ
"""
        elif task_type == "task_14":
            return base_system_message + """
Для задания 14 (стереометрическая задача) особенно важно проверить:
- Корректность и полноту доказательства в пункте а
- Правильность построений и вычислений в пункте б
- Обоснованность каждого шага решения

КРИТИЧЕСКИ ВАЖНО:
- В ПЕРВУЮ ОЧЕРЕДЬ СРАВНИВАЙ ОТВЕТЫ УЧЕНИКА С ПРАВИЛЬНЫМИ ОТВЕТАМИ!
- ЕСЛИ ОТВЕТ УЧЕНИКА НЕВЕРНЫЙ, ЭТО ОБЯЗАТЕЛЬНО ДОЛЖНО БЫТЬ УЧТЕНО В ОЦЕНКЕ!
- РАЗЛИЧАЙ АРИФМЕТИЧЕСКИЕ ОШИБКИ ОТ КОНЦЕПТУАЛЬНЫХ ОШИБОК!

ТЩАТЕЛЬНО ПРОВЕРЯЙ:
- Убедись, что доказательство в пункте а полное и корректное
- Проверь, что все вычисления в пункте б выполнены правильно
- Обрати внимание на правильность применения формул и теорем стереометрии
- НЕ ЗАБУДЬ ОТМЕТИТЬ ВСЕ РАСХОЖДЕНИЯ МЕЖДУ ОТВЕТАМИ УЧЕНИКА И ПРАВИЛЬНЫМИ ОТВЕТАМИ
"""
        elif task_type == "task_17":
            return base_system_message + """
Для задания 17 (планиметрическая задача с доказательством) особенно важно проверить:
- Строгость и полноту доказательства в пункте а
- Правильность вычислений в пункте б
- Логическую связь между пунктами а и б, если она используется в решении
"""
        else:
            return base_system_message

    def create_messages_with_image(
        self,
        task_type: str,
        task_description: str,
        student_solution_image: Dict[str, Any],
        correct_solution_image: Optional[Dict[str, Any]] = None,
        include_examples: bool = False,
        examples: Optional[List[Dict[str, Any]]] = None,
        prompt_variant: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Create a list of messages for the API, including images of the solution.

        Args:
            task_type: Type of task (e.g., "task_13", "task_17")
            task_description: Description of the task
            student_solution_image: Image data of student's solution in the format expected by the API
            correct_solution_image: Optional image data of correct solution in the format expected by the API
            include_examples: Whether to include examples in the prompt
            examples: List of example solutions with evaluations
            prompt_variant: Specific prompt variant to use (e.g., "basic", "detailed")

        Returns:
            List of messages for the API
        """
        # Generate the system message
        system_message = self.generate_system_message(task_type)

        # Generate the user prompt
        prompt = self.generate_prompt(
            task_type=task_type,
            task_description=task_description,
            include_examples=include_examples,
            examples=examples,
            prompt_variant=prompt_variant
        )

        # Create content list with prompt and images
        content = [{"type": "text", "text": prompt}]

        # Add example images if requested and task_type supports image_examples and we're using the image_examples prompt
        if include_examples and task_type in ["task_13", "task_14"] and prompt_variant == "image_examples":
            import os
            from app.utils.image_utils import prepare_image_for_api

            # Path to examples directory - use the appropriate subfolder based on task type
            task_number = task_type.split("_")[1]  # Extract "13" or "14" from "task_13" or "task_14"
            examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "examples", task_number)

            if os.path.exists(examples_dir):
                # Add a header for examples
                content.append({"type": "text", "text": "\n\n## Примеры решений и их оценок:\n\n"})

                # Define example descriptions with their scores and explanations based on task type
                if task_type == "task_13":
                    example_descriptions = {
                        "first_example": {
                            "score": 2,
                            "description": "Решение оценено на 2 балла. Обоснованно получены верные ответы в обоих пунктах. Все шаги решения корректны, и отбор корней на заданном отрезке выполнен правильно."
                        },
                        "second_example": {
                            "score": 1,
                            "description": "Решение оценено на 1 балл. Обоснованно получен верный ответ в пункте а), но в пункте б) допущены ошибки при отборе корней на заданном отрезке."
                        },
                        "third_example": {
                            "score": 1,
                            "description": "Решение оценено на 1 балл. Обоснованно получен верный ответ в пункте а), но в пункте б) неверно определена принадлежность корней заданному отрезку."
                        },
                        "forth_example": {
                            "score": 1,
                            "description": "Решение оценено на 1 балл. Обоснованно получен верный ответ в пункте а), но при использовании тригонометрической окружности в пункте б) не выделена дуга, соответствующая отрезку."
                        }
                    }
                elif task_type == "task_14":
                    example_descriptions = {
                        "first_example": {
                            "score": 3,
                            "description": "Решение оценено на 3 балла. Имеется верное доказательство утверждения пункта а) и обоснованно получен верный ответ в пункте б). Все шаги решения корректны."
                        },
                        "second_example": {
                            "score": 1,
                            "description": "Решение оценено на 1 балл. Имеется верное доказательство утверждения пункта а), но решение пункта б) отсутствует."
                        },
                        "third_example": {
                            "score": 1,
                            "description": "Решение оценено на 1 балл. Имеется верное доказательство утверждения пункта а) и получен неверный ответ в пункте б) из-за арифметической ошибки."
                        },
                        "forth_example": {
                            "score": 0,
                            "description": "Решение оценено на 0 баллов. Решалась другая задача: точка K лежит на стороне основания, а не на боковом ребре."
                        }
                    }
                else:
                    example_descriptions = {}

                # Get subdirectories (each containing an example)
                example_dirs = [d for d in os.listdir(examples_dir) if os.path.isdir(os.path.join(examples_dir, d))]

                for i, example_dir in enumerate(example_dirs, 1):
                    dir_path = os.path.join(examples_dir, example_dir)
                    image_files = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

                    if image_files:
                        # Get example info
                        example_info = example_descriptions.get(example_dir, {"score": "?", "description": "Пример решения задания 13."})

                        # Add example header with description
                        content.append({"type": "text", "text": f"### Пример {i} (оценка: {example_info['score']} балла):\n{example_info['description']}\n\n"})

                        # Add each image in the example directory
                        for img_file in image_files:
                            try:
                                img_path = os.path.join(dir_path, img_file)
                                with open(img_path, "rb") as f:
                                    img_data = f.read()

                                # Prepare image for API with smaller max_size to reduce token count
                                from PIL import Image
                                import io
                                img = Image.open(io.BytesIO(img_data))
                                example_image = prepare_image_for_api(img, max_size=1024)  # Reduced from 4096 to 1024
                                content.append(example_image)
                                content.append({"type": "text", "text": "\n"})
                            except Exception as e:
                                content.append({"type": "text", "text": f"[Ошибка загрузки примера: {str(e)}]\n"})

        # Add correct solution image if provided
        if correct_solution_image:
            # Add a separator after examples if they were included
            if include_examples and task_type in ["task_13", "task_14"] and prompt_variant == "image_examples":
                content.append({"type": "text", "text": "\n\n## Задание для оценки:\n\nТеперь, когда ты изучил примеры, переходим к заданию, которое нужно оценить.\n\n"})

            content.append(correct_solution_image)
            # Add a separator text between images
            content.append({"type": "text", "text": "\n\nВыше представлено условие задачи и ПРАВИЛЬНОЕ РЕШЕНИЕ. Ниже представлено решение ученика, которое нужно оценить. \n\nКРИТИЧЕСКИ ВАЖНО: В ПЕРВУЮ ОЧЕРЕДЬ СРАВНИ ОТВЕТЫ ученика с правильными ответами! Если ответ ученика неверный, это ОБЯЗАТЕЛЬНО должно быть учтено в оценке, даже если все преобразования выполнены верно!\n\nПроверь, что все шаги решения выполнены правильно. Не забудь отметить все расхождения между ответами ученика и правильными ответами. Проанализируй решение в соответствии с критериями и примерами выше:\n\n"})
        else:
            # If no correct solution, but we had examples, add a separator
            if include_examples and task_type in ["task_13", "task_14"] and prompt_variant == "image_examples":
                content.append({"type": "text", "text": "\n\n## Решение ученика для оценки:\n\nТеперь, когда ты изучил примеры, переходим к решению ученика, которое нужно оценить. Проанализируй его в соответствии с критериями и примерами выше:\n\n"})

        # Add student solution image
        content.append(student_solution_image)

        # Create the messages list
        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": content
            }
        ]

        return messages
