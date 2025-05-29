"""
Score Extractor Module

This module provides utilities for extracting scores from model responses.
"""

import re
import logging

logger = logging.getLogger(__name__)

def extract_score_from_text(result_text: str, task_type: str = None) -> int:
    """
    Extract the score from the model's response text.

    Args:
        result_text: The text response from the model
        task_type: Optional task type for validation (e.g., "task_13")

    Returns:
        Extracted score as an integer
    """
    # Input validation
    if not isinstance(result_text, str):
        logger.error(f"Invalid result_text type: {type(result_text)}")
        return 0

    if not result_text.strip():
        logger.warning("Empty result_text provided")
        return 0

    score = 0

    try:
        # First approach: Semantic section detection - look for the "Итоговая оценка" section
        logger.info("Attempting to extract score using semantic section detection")

        # Split the text into sections by markdown headers
        sections = re.split(r'###\s+', result_text)

        # Look for the "Итоговая оценка" section
        score_section = None
        for section in sections:
            if section.strip().lower().startswith('итоговая оценка'):
                score_section = section.strip()
                logger.info(f"Found 'Итоговая оценка' section: {score_section[:100]}...")
                break

        # If we found the section, extract the score from it
        if score_section:
            # Look for the pattern [Оценка: X баллов] or similar
            score_patterns = [
                r'\[оценка:\s*(\d+)\s*балл',  # [Оценка: 2 балла]
                r'оценка:\s*(\d+)\s*балл',    # Оценка: 2 балла
                r'(\d+)\s*балл'               # 2 балла
            ]

            for pattern in score_patterns:
                matches = re.findall(pattern, score_section.lower())
                if matches:
                    score = int(matches[0])  # Use the first match in the section
                    logger.info(f"Found score {score} in 'Итоговая оценка' section using pattern: {pattern}")
                    break

        # Second approach: If semantic section detection failed, fall back to traditional methods
        if score == 0:
            logger.info("Semantic section detection failed, falling back to traditional methods")

            # Look for explicit score sections
            score_sections = [
                r'итоговая оценка[\s\S]*?(\d+)\s*балл',
                r'оценка[\s\S]*?(\d+)\s*балл',
                r'\[оценка[\s\S]*?(\d+)\s*балл',
                r'итоговый балл[\s\S]*?(\d+)',
                r'итоговая оценка[\s\S]*?(\d+)',
                r'оценка:\s*(\d+)',
                r'выставляется\s*(\d+)\s*балл'
            ]

            for pattern in score_sections:
                matches = re.findall(pattern, result_text.lower())
                if matches:
                    score = int(matches[-1])  # Use the last match as it's likely the final score
                    logger.info(f"Found score {score} using pattern: {pattern}")
                    break

            # Look for lines with "оценка" and extract digits
            if score == 0:
                for line in result_text.split("\n"):
                    if "оценка" in line.lower() or "балл" in line.lower():
                        # Extract all digits from the line
                        digits = [int(s) for s in re.findall(r'\d+', line)]
                        if digits:
                            score = digits[-1]  # Use the last digit as the score
                            logger.info(f"Found score {score} from line: {line}")
                            break

            # Look for specific formats in the entire text
            if score == 0:
                # Look for patterns like "1 балл" or "2 балла" in the explanation
                score_patterns = [
                    r'(\d+)\s*балл',  # "2 балла"
                    r'оценка\s*[:-]\s*(\d+)',  # "Оценка: 2"
                    r'\[(\d+)\s*балл',  # "[2 балла"
                    r'\[оценка\s*[:-]\s*(\d+)\]'  # "[Оценка: 2]"
                ]

                for pattern in score_patterns:
                    matches = re.findall(pattern, result_text.lower())
                    if matches:
                        score = int(matches[-1])
                        logger.info(f"Found score {score} using pattern: {pattern}")
                        break

            # Most aggressive - find any number after "оценка"
            if score == 0:
                # Find any occurrence of "оценка" followed by a number within 30 characters
                score_sections = re.findall(r'оценка.{1,30}?(\d+)', result_text.lower())
                if score_sections:
                    score = int(score_sections[-1])
                    logger.info(f"Found score {score} using aggressive approach")

            # Check for specific score mentions in the text
            if score == 0:
                if "2 балла" in result_text.lower() or "два балла" in result_text.lower():
                    score = 2
                    logger.info("Found score 2 from direct text mention")
                elif "1 балл" in result_text.lower() or "один балл" in result_text.lower():
                    score = 1
                    logger.info("Found score 1 from direct text mention")
                elif "0 баллов" in result_text.lower() or "ноль баллов" in result_text.lower():
                    score = 0
                    logger.info("Found score 0 from direct text mention")

        # Validate the score is within expected range for specific task types
        # Define maximum scores for each task type
        max_scores = {
            "task_13": 2,
            "task_14": 3,
            "task_15": 2,
            "task_16": 2,  # Fixed: Economic task has maximum 2 points
            "task_17": 3,
            "task_18": 4,
            "task_19": 4
        }

        if task_type and task_type in max_scores:
            max_score = max_scores[task_type]
            if score > max_score:
                logger.warning(f"Score {score} exceeds maximum of {max_score} for {task_type}, capping at {max_score}")
                score = max_score
            elif score < 0:
                logger.warning(f"Score {score} is negative for {task_type}, setting to 0")
                score = 0

        # Ensure score is a valid integer
        if not isinstance(score, int):
            try:
                score = int(score)
            except (ValueError, TypeError):
                logger.error(f"Could not convert score to integer: {score}")
                score = 0

    except Exception as e:
        logger.error(f"Error extracting score: {str(e)}")
        score = 0
    
    return score
