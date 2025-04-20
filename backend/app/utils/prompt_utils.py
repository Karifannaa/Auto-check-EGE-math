"""
Prompt Utilities Module

This module provides utilities for creating effective prompts for
evaluating math exam solutions using reasoning models.
"""

from typing import Dict, List, Optional, Union, Any
from app.utils.specialized_prompts import (
    TASK_13_PROMPTS,
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
            variant: Prompt variant to use ("basic", "detailed", or "with_examples")

        Returns:
            Template string for the specified variant
        """
        if variant in TASK_13_PROMPTS:
            return TASK_13_PROMPTS[variant]
        else:
            # Fall back to basic prompt if variant not found
            return TASK_13_PROMPTS["basic"]

    def _get_task_14_template(self) -> str:
        """Get template for Task 14 (stereometry)."""
        return """
Ты - эксперт по проверке задач ЕГЭ по математике. Проанализируй решение стереометрической задачи.

Задача: {task_description}

Критерии оценки:
- 2 балла: Обоснованно получен верный ответ
- 1 балл: Получен неверный ответ из-за вычислительной ошибки, но при этом имеется верная последовательность всех шагов решения
- 0 баллов: Решение не соответствует ни одному из критериев, перечисленных выше

Максимальный балл: 2

Проанализируй решение и оцени его по указанным критериям. Предоставь подробное обоснование своей оценки, указав на конкретные элементы решения.

Твой ответ должен содержать:
1. Анализ хода решения
2. Проверку правильности вычислений
3. Итоговую оценку (0, 1 или 2 балла)
4. Обоснование выставленной оценки
"""

    def _get_task_15_template(self) -> str:
        """Get template for Task 15 (inequalities)."""
        return """
Ты - эксперт по проверке задач ЕГЭ по математике. Проанализируй решение неравенства.

Задача: {task_description}

Критерии оценки:
- 2 балла: Обоснованно получен верный ответ
- 1 балл: Получен неверный ответ из-за вычислительной ошибки, но при этом имеется верная последовательность всех шагов решения
- 0 баллов: Решение не соответствует ни одному из критериев, перечисленных выше

Максимальный балл: 2

Проанализируй решение и оцени его по указанным критериям. Предоставь подробное обоснование своей оценки, указав на конкретные элементы решения.

Твой ответ должен содержать:
1. Анализ метода решения неравенства
2. Проверку правильности всех преобразований
3. Итоговую оценку (0, 1 или 2 балла)
4. Обоснование выставленной оценки
"""

    def _get_task_16_template(self) -> str:
        """Get template for Task 16 (planimetry)."""
        return """
Ты - эксперт по проверке задач ЕГЭ по математике. Проанализируй решение планиметрической задачи.

Задача: {task_description}

Критерии оценки:
- 3 балла: Обоснованно получен верный ответ
- 2 балла: Получен неверный ответ из-за вычислительной ошибки, но при этом имеется верная последовательность всех шагов решения
- 1 балл: Имеется верный ход решения, но решение не доведено до конца
- 0 баллов: Решение не соответствует ни одному из критериев, перечисленных выше

Максимальный балл: 3

Проанализируй решение и оцени его по указанным критериям. Предоставь подробное обоснование своей оценки, указав на конкретные элементы решения.

Твой ответ должен содержать:
1. Анализ геометрических утверждений и их обоснованности
2. Проверку правильности вычислений
3. Итоговую оценку (0, 1, 2 или 3 балла)
4. Обоснование выставленной оценки
"""

    def _get_task_17_template(self) -> str:
        """Get template for Task 17 (planimetry with proof)."""
        return """
Ты - эксперт по проверке задач ЕГЭ по математике. Проанализируй решение планиметрической задачи с доказательством.

Задача: {task_description}

Критерии оценки:
- 3 балла: Имеется верное доказательство утверждения пункта а и обоснованно получен верный ответ в пункте б
- 2 балла: Обоснованно получен верный ответ в пункте б ИЛИ имеется верное доказательство утверждения пункта а и при обоснованном решении пункта б получен неверный ответ из-за арифметической ошибки
- 1 балл: Имеется верное доказательство утверждения пункта а, ИЛИ при обоснованном решении пункта б получен неверный ответ из-за арифметической ошибки, ИЛИ обоснованно получен верный ответ в пункте б с использованием утверждения пункта а, при этом пункт а не выполнен
- 0 баллов: Решение не соответствует ни одному из критериев, перечисленных выше

Максимальный балл: 3

Проанализируй решение и оцени его по указанным критериям. Предоставь подробное обоснование своей оценки, указав на конкретные элементы решения.

Твой ответ должен содержать:
1. Анализ доказательства в пункте а
2. Анализ решения в пункте б
3. Итоговую оценку (0, 1, 2 или 3 балла)
4. Обоснование выставленной оценки
"""

    def _get_task_18_template(self) -> str:
        """Get template for Task 18 (parameter problems)."""
        return """
Ты - эксперт по проверке задач ЕГЭ по математике. Проанализируй решение задачи с параметром.

Задача: {task_description}

Критерии оценки:
- 4 балла: Обоснованно получен верный ответ
- 3 балла: Верно обоснованы все случаи, но в одном из них допущена вычислительная ошибка, не нарушающая общей логики решения, в результате чего получен неверный ответ
- 2 балла: Верно обоснованы 2 случая из 3 возможных
- 1 балл: Верно обоснован 1 случай из 3 возможных
- 0 баллов: Решение не соответствует ни одному из критериев, перечисленных выше

Максимальный балл: 4

Проанализируй решение и оцени его по указанным критериям. Предоставь подробное обоснование своей оценки, указав на конкретные элементы решения.

Твой ответ должен содержать:
1. Анализ рассмотренных случаев
2. Проверку правильности решения для каждого случая
3. Итоговую оценку (0, 1, 2, 3 или 4 балла)
4. Обоснование выставленной оценки
"""

    def _get_task_19_template(self) -> str:
        """Get template for Task 19 (number theory)."""
        return """
Ты - эксперт по проверке задач ЕГЭ по математике. Проанализируй решение задачи по теории чисел.

Задача: {task_description}

Критерии оценки:
- 4 балла: Верно получены все перечисленные в условии объекты
- 3 балла: Верно получено два из трех перечисленных в условии объектов
- 2 балла: Верно получен один из трех перечисленных в условии объектов
- 1 балл: В решении имеются верные рассуждения, но ни один из перечисленных в условии объектов не получен
- 0 баллов: Решение не соответствует ни одному из критериев, перечисленных выше

Максимальный балл: 4

Проанализируй решение и оцени его по указанным критериям. Предоставь подробное обоснование своей оценки, указав на конкретные элементы решения.

Твой ответ должен содержать:
1. Анализ метода решения
2. Проверку правильности всех полученных объектов
3. Итоговую оценку (0, 1, 2, 3 или 4 балла)
4. Обоснование выставленной оценки
"""

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

        # For task_13, we have specialized prompts with variants
        if task_type == "task_13":
            # If the variant includes examples and include_examples is True, use the with_examples variant
            if include_examples and prompt_variant != "with_examples" and "with_examples" in AVAILABLE_PROMPT_VARIANTS[task_type]:
                template = self._get_task_13_template("with_examples")
            else:
                template = self._get_task_13_template(prompt_variant)
        else:
            # For other task types, use the standard template
            template = self.templates[task_type]

        # Format with task description
        prompt = template.format(task_description=task_description)

        # Add examples if requested and we're not already using a template with examples
        if include_examples and examples and task_type != "task_13" and prompt_variant != "with_examples":
            examples_text = "\n\nПримеры оценивания:\n"
            for i, example in enumerate(examples, 1):
                examples_text += f"\nПример {i}:\n"
                examples_text += f"Задача: {example.get('task', '')}\n"
                examples_text += f"Решение ученика: {example.get('solution', '')}\n"
                examples_text += f"Оценка: {example.get('score', '')} баллов\n"
                examples_text += f"Обоснование: {example.get('explanation', '')}\n"

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
Для задания 13 (тригонометрическое, логарифмическое или показательное уравнение) особенно важно проверить:
- Правильность всех преобразований уравнения
- Корректность отбора корней в пункте б
- Наличие обоснований для всех шагов решения
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
        image_data: Dict[str, Any],
        include_examples: bool = False,
        examples: Optional[List[Dict[str, Any]]] = None,
        prompt_variant: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Create a list of messages for the API, including an image of the solution.

        Args:
            task_type: Type of task (e.g., "task_13", "task_17")
            task_description: Description of the task
            image_data: Image data in the format expected by the API
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

        # Create the messages list
        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_data
                ]
            }
        ]

        return messages
