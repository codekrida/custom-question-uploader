#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Question Generator CLI

A command-line tool to generate bilingual (English/Hindi) educational questions
in multiple formats (MCQ, True/False, Fill-in-the-Blank, etc.) using Google’s
Gemini API. Optionally persists output to a JSON file or pushes it to a REST API.

Features:
  • Multiple question types: MCQ, MOC, FILL_BLANK, TRUE_FALSE, SHORT_ANSWER,
    MATCHING, ARRANGE, ESSAY
  • Bilingual JSON output with English/Hindi text, options, explanations
  • Difficulty levels (1=Easy, 2=Medium, 3=Hard) with auto-assigned point values
  • Modular prompt templates for each question type
  • Environment-based and CLI-based configuration (API key, DB endpoint, tags…)
  • Robust error handling and response validation
  • Optional storage: local JSON file and/or remote database via REST
  • Colored console logging for better readability
  • Automatic tag fetching from API if not provided via CLI

Usage:
  python question_generator.py --api-key <YOUR_GEMINI_KEY> --num-questions 10 --question-types '["MCQ","TRUE_FALSE"]' --difficulty 2 --output questions.json [--tags '["class-12","calculus","integrals"]'] [--db-endpoint http://…/create-question]

  If --tags is omitted, the script will fetch tags from https://www.examshala.in/api/tags/single-tag.
  API_KEY and DB_API_ENDPOINT can also be set in a .env file or environment variables.
"""

import argparse
import ast
import json
import logging
import os
import random
import re
import sys
from typing import Dict, List, Optional, Union, Any, Tuple

# External libraries
try:
    from dotenv import load_dotenv
    import colorlog  # For colored logging
except ImportError:
    # Handle missing optional dependencies gracefully
    print(
        "Warning: Missing optional dependencies. Install with 'pip install python-dotenv colorlog'."
    )
    load_dotenv = None
    colorlog = None

# Google Gemini API library
from google.generativeai.client import configure
from google.generativeai.generative_models import GenerativeModel
import requests

# --- Logging Configuration ---

# Configure colored logging using colorlog if available, otherwise basic logging
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
COLORED_LOG_FORMAT = "%(log_color)s" + LOG_FORMAT + "%(reset)s"

# Check if handlers already exist to prevent duplicates when running in certain environments
if not logging.getLogger().handlers:
    # Create logger
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    # Create stream handler
    stream_handler = logging.StreamHandler(sys.stdout)

    # Use colored formatter if colorlog is available, otherwise use basic formatter
    if colorlog:
        formatter = colorlog.ColoredFormatter(COLORED_LOG_FORMAT)
        # Configure color mapping for log levels
        # This also adds a handler, so check again after basicConfig
        colorlog.basicConfig(level=LOG_LEVEL, format=COLORED_LOG_FORMAT)
        if not root_logger.handlers:  # Check if basicConfig added a handler
            stream_handler.setFormatter(formatter)
            root_logger.addHandler(stream_handler)  # Add handler if basicConfig didn't
        else:
            # If basicConfig added handlers, ensure they use the colored formatter
            for handler in root_logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setFormatter(formatter)

    else:
        formatter = logging.Formatter(LOG_FORMAT)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)


# Get a logger instance for this module
logger = logging.getLogger(__name__)

# Load environment variables from .env file if python-dotenv is installed
if load_dotenv:
    load_dotenv()
    logger.info("Loaded environment variables from .env")

# --- Constants and Configuration ---

# List of supported question types
QUESTION_TYPES = [
    "MCQ",  # Multiple Choice Question
    "MOC",  # Multiple Options Correct
    "FILL_BLANK",  # Fill in the Blank
    "TRUE_FALSE",  # True or False
    "SHORT_ANSWER",  # Short Answer
    "MATCHING",  # Matching Pairs
    "ARRANGE",  # Arrange in Sequence
    "ESSAY",  # Essay/Long Answer
]

# Default values used if not provided via CLI or environment variables or API
DEFAULT_TAGS = ["class-12", "calculus", "integrals"]
DEFAULT_NUM_QUESTIONS = 10
DEFAULT_QUESTION_TYPES = ["MCQ", "MOC", "SHORT_ANSWER"]
# Default difficulty levels (1: Easy, 2: Medium, 3: Hard)
DEFAULT_DIFFICULTIES = [1, 2, 3]
DEFAULT_OUTPUT_FILE = "questions.json"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-preview-04-17"

# Environment variables for sensitive configuration
ENV_VAR_API_KEY = "GEMINI_API_KEY"
# CHANGE: Updated DB endpoint to create-question
ENV_VAR_DB_ENDPOINT = "https://app.examshala.in/api/create-multiple-questions-in-bank"
TAGS_API_URL = "https://www.examshala.in/api/tags/by-subject?subjectId=62"  # API endpoint to fetch tags

# --- JSON Templates for Prompting ---
# Using a dictionary to store prompt templates for each question type.
# These templates define the expected JSON structure from the LLM response.
QUESTION_JSON_TEMPLATES: Dict[str, str] = {
    "MCQ": """
          {{
            "moduleId": "{module_id}",
            "type": "MCQ",
            "en": {{"text": "<question text>", "options": ["<option1>", "<option2>", "<option3>", "<option4>"]}},
            "hi": {{"text": "<question text in Hindi>", "options": ["<option1 in Hindi>", "<option2 in Hindi>", "<option3 in Hindi>", "<option4 in Hindi>"]}},
            "correctAnswer": {{"index": <index of correct option 0-3>, "value": "<value of correct option>"}},
            "explanation": {{"en": "<explanation in English>", "hi": "<explanation in Hindi>"}},
            "difficultyLevel": {difficulty},
            "points": <points>,
            "tags": {tags_str}
          }}
        """,
    "MOC": """
          {{
            "moduleId": "{module_id}",
            "type": "MOC",
            "en": {{"text": "<question text>", "options": ["<option1>", "<option2>", "<option3>", "<option4>"]}},
            "hi": {{"text": "<question text in Hindi>", "options": ["<option1 in Hindi>", "<option2 in Hindi>", "<option3 in Hindi>", "<option4 in Hindi>"]}},
            "correctAnswer": [<index1>, <index2>, ...], # Array of correct option indices
            "explanation": {{"en": "<explanation in English>", "hi": "<explanation in Hindi>"}},
            "difficultyLevel": {difficulty},
            "points": <points>,
            "tags": {tags_str}
          }}
        """,
    "FILL_BLANK": """
          {{
            "moduleId": "{module_id}",
            "type": "FILL_BLANK",
            "en": {{"text": "<question text with blank as ______>"}},
            "hi": {{"text": "<question text in Hindi with blank as ______>"}},
            "correctAnswer": {{"value": "<value>"}}, # The word or phrase that fills the blank
            "explanation": {{"en": "<explanation in English>", "hi": "<explanation in Hindi>"}},
            "difficultyLevel": {difficulty},
            "points": <points>,
            "tags": {tags_str}
          }}
        """,
    "TRUE_FALSE": """
          {{
            "moduleId": "{module_id}",
            "type": "TRUE_FALSE",
            "en": {{"text": "<statement>"}},
            "hi": {{"text": "<statement in Hindi>"}},
            "correctAnswer": {{"value": "<true or false>"}}, # Boolean true or false
            "explanation": {{"en": "<explanation in English>", "hi": "<explanation in Hindi>"}},
            "difficultyLevel": {difficulty},
            "points": <points>,
            "tags": {tags_str}
          }}
        """,
    "SHORT_ANSWER": """
          {{
            "moduleId": "{module_id}",
            "type": "SHORT_ANSWER",
            "en": {{"text": "<question text>"}},
            "hi": {{"text": "<question text in Hindi>"}},
            "correctAnswer": {{"value": "<value>"}}, # Expected short answer
            "explanation": {{"en": "<explanation in English>", "hi": "<explanation in Hindi>"}},
            "difficultyLevel": {difficulty},
            "points": <points>,
            "tags": {tags_str}
          }}
        """,
    "MATCHING": """
          {{
            "moduleId": "{module_id}",
            "type": "MATCHING",
            "en": {{"text": "<question text>", "items": ["<item1>", "<item2>", "<item3>"], "matches": ["<match1>", "<match2>", "<match3>"]}},
            "hi": {{"text": "<question text in Hindi>", "items": ["<item1 in Hindi>", "<item2 in Hindi>", "<item3 in Hindi>"], "matches": ["<match1 in Hindi>", "<match2 in Hindi>", "<match3 in Hindi>"]}},
            "correctAnswer": {{"pairs": [[<item_index1>, <match_index1>], [<item_index2>, <match_index2>], [<item_index3>, <match_index3>]]}}, # Array of pairs [item_index, match_index]
            "explanation": {{"en": "<explanation in English>", "hi": "<explanation in Hindi>"}},
            "difficultyLevel": {difficulty},
            "points": <points>,
            "tags": {tags_str}
          }}
        """,
    "ARRANGE": """
          {{
            "moduleId": "{module_id}",
            "type": "ARRANGE",
            "en": {{"text": "<question text>", "options": ["<option1>", "<option2>", "<option3>"]}}, # Items to be arranged
            "hi": {{"text": "<question text in Hindi>", "options": ["<option1 in Hindi>", "<option2 in Hindi>", "<option3 in Hindi>"]}},
            "correctAnswer": [<index1>, <index2>, <index3>], # The correct order of indices
            "explanation": {{"en": "<explanation in English>", "hi": "<explanation in Hindi>"}},
            "difficultyLevel": {difficulty},
            "points": <points>,
            "tags": {tags_str}
          }}
        """,
    "ESSAY": """
          {{
            "moduleId": "{module_id}",
            "type": "ESSAY",
            "en": {{"text": "<question text>"}},
            "hi": {{"text": "<question text in Hindi>"}},
            "correctAnswer": {{"value": "Subjective"}}, # Fixed value for essay type
            "explanation": {{"en": "<explanation in English>", "hi": "<explanation in Hindi>"}},
            "difficultyLevel": {difficulty},
            "points": <points>,
            "tags": {tags_str}
          }}
        """,
}

# --- Custom Argument Parsers ---


def parse_list_arg(value: str) -> List[Any]:
    """
    Parse a JSON or Python-style list string from the command line.
    Used for arguments like --tags and --question-types.

    Args:
        value: The string value from the command line.

    Returns:
        A Python list parsed from the string.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be parsed as a list.
    """
    try:
        # Attempt to parse as a JSON string
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            # Fallback to Python literal_eval for single-quoted lists (e.g., ['a', 'b'])
            parsed_list = ast.literal_eval(value)
            if isinstance(parsed_list, list):
                return parsed_list
            raise ValueError("Parsed value is not a list.")
        except (ValueError, SyntaxError) as e:
            raise argparse.ArgumentTypeError(
                f"Invalid list format: {value}. Expected JSON (e.g. \"['a', 'b']\") or Python list string."
            ) from e


def parse_difficulty_arg(value: str) -> Union[int, List[int]]:
    """
    Parse difficulty argument as a single integer (1-3) or a list of integers (1-3).

    Args:
        value: The string value from the command line.

    Returns:
        An integer or a list of integers representing difficulty levels.

    Raises:
        argparse.ArgumentTypeError: If the value is not a valid integer or list of integers (1-3).
    """
    try:
        # Attempt to parse as a single integer
        difficulty_int = int(value)
        if 1 <= difficulty_int <= 3:
            return difficulty_int
        else:
            raise ValueError("Integer must be between 1 and 3.")
    except ValueError:
        # If not an integer, attempt to parse as a list
        try:
            parsed_list = parse_list_arg(value)
            if all(isinstance(x, int) and 1 <= x <= 3 for x in parsed_list):
                return parsed_list
            raise ValueError("List must contain only integers between 1 and 3.")
        except argparse.ArgumentTypeError as e:
            raise argparse.ArgumentTypeError(
                f"Invalid difficulty format: {value}. Expected integer (1-3) or list of integers (e.g. '[1, 3]')."
            ) from e
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"Invalid difficulty format: {value}. {e}"
            ) from e


# --- Core Logic Classes ---


class QuestionGenerator:
    """
    Handles question generation using the Google Gemini API.
    Manages API configuration, prompt creation, calling the API,
    and basic parsing/validation of the response.
    """

    def __init__(self, api_key: str, model_name: str = DEFAULT_GEMINI_MODEL):
        """
        Initialize the Question Generator with API credentials and model name.

        Args:
            api_key: Google Gemini API key.
            model_name: The specific Gemini model to use for generation.
        """
        if not api_key:
            raise ValueError("API key cannot be empty.")
        self.api_key = api_key
        self.model_name = model_name
        self.model: Optional[GenerativeModel] = None
        self._configure_api()

    def _configure_api(self) -> None:
        """
        Configure the Gemini API with credentials and load the model.
        Exits the program if configuration fails.
        """
        try:
            configure(api_key=self.api_key)
            self.model = GenerativeModel(self.model_name)
            logger.info(
                f"Successfully configured Gemini API with model: {self.model_name}"
            )
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            logger.error("Please check your API key and network connection.")
            # Exit the application as API is fundamental
            sys.exit("API Configuration Failed")

    def _assign_points(self, difficulty: int) -> int:
        """
        Assign points to a question based on its difficulty level.

        Args:
            difficulty: The difficulty level (1, 2, or 3).

        Returns:
            The corresponding point value.
        """
        points_map = {1: 5, 2: 10, 3: 15}
        # Default to 5 points if difficulty is unexpected
        return points_map.get(difficulty, 5)

    def _create_prompt(
        self,
        topic: str,
        question_type: str,
        tags: List[str],
        module_id: str,
        difficulty: int,
    ) -> str:
        """
        Create a structured prompt for the Gemini model, requesting a JSON response
        that follows a specific template based on the question type.

        Args:
            topic: Subject topic derived from tags for the question.
            question_type: The type of question to generate (e.g., "MCQ").
            tags: List of tags associated with the question.
            module_id: The module identifier.
            difficulty: The difficulty level (1-3).

        Returns:
            A formatted prompt string ready to be sent to the LLM.

        Raises:
            ValueError: If the requested question_type is not supported or has no template.
        """
        # Check if a template exists for the requested question type
        if question_type not in QUESTION_JSON_TEMPLATES:
            raise ValueError(
                f"Unsupported question type or missing template: {question_type}"
            )

        # Ensure tags are formatted as a JSON string for inclusion in the template
        tags_str = json.dumps(tags)

        # Join tags to form a representative topic string for the LLM context
        # This helps the model understand the subject context broadly.
        topic_context = "->".join(tags)

        logger.debug("Creating prompt for question generation...", topic_context)

        # Retrieve the specific JSON template for the question type
        json_template = QUESTION_JSON_TEMPLATES[question_type].strip()

        # Construct the full prompt
        prompt = f"""
        Generate a single educational question in JSON format.
        Subject Context: {topic_context}
        Specific Topic/Focus: {tags[-1]}
        Tags: {tags_str}
        Module ID: {module_id}
        Question Type: {question_type}
        Difficulty Level: {difficulty} (1=easy, 2=medium, 3=hard)
        Language: Bilingual (English and Hindi)

        Constraints:
        - Respond ONLY with the JSON object. Do not include any other text, comments, or markdown outside the JSON block.
        - Adhere strictly to the JSON template provided below for the "{question_type}" type.
        - Ensure the content is accurate, relevant to the topic and tags, and grammatically correct in both English and Hindi.
        - Use $$...$$ for mathematical expressions.
        - Support markdown syntax (e.g., line breaks \\n, bold **, code `) within text fields like "text" and "explanation".
        - If applicable for the question type, support mermaid syntax for diagrams within markdown.

        JSON Template for {question_type}:
        {json_template.format(
             module_id=module_id,
             difficulty=difficulty,
             tags_str=tags_str
             # Note: <points> and <correctAnswer> details are placeholders for the model to fill,
             # but the structure around them is defined by the template.
        )}

        Generate the question content and provide the complete JSON object formatted as specified by the template.
        """

        return prompt

    def _extract_json_from_response(
        self, response_text: str
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt to extract a JSON object from the raw response text received from the LLM.
        This method handles cases where the model might wrap the JSON in markdown code blocks
        (e.g., ```json ... ```) or include conversational text.

        Args:
            response_text: The raw text response string from the Gemini API.

        Returns:
            The parsed JSON object as a dictionary, or None if extraction or parsing fails.
        """
        # First, try to find and extract content within a ```json ... ``` markdown block
        match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        if match:
            json_str = match.group(
                1
            ).strip()  # Extract content and remove leading/trailing whitespace
            logger.debug("Extracted JSON from ```json``` block.")
        else:
            # If no markdown block is found, assume the entire text is intended to be JSON
            json_str = response_text.strip()
            logger.debug(
                "No ```json``` block found. Assuming entire response text is JSON."
            )

        # Handle cases where the extracted string might be empty or just whitespace
        if not json_str:
            logger.warning("Extracted or received empty string for JSON content.")
            return None

        try:
            # Attempt to parse the extracted string as JSON
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Log a detailed error if JSON parsing fails
            logger.error(f"Failed to parse JSON from response: {e}")
            logger.debug(
                f"Attempted to parse the following string:\n---\n{json_str[:1000]}{'...' if len(json_str) > 1000 else ''}\n---"
            )  # Log start of invalid string
            return None
        except Exception as e:
            # Catch any other unexpected errors during extraction/parsing
            logger.error(
                f"An unexpected error occurred during JSON extraction: {e}",
                exc_info=True,
            )
            return None

    def generate_question(
        self,
        topic: str,
        question_type: str,
        tags: List[str],
        module_id: str,
        difficulty: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a single question using the configured Gemini API model.

        Args:
            topic: Subject topic for the question.
            question_type: Type of question (must be in QUESTION_TYPES).
            tags: List of tags for classification.
            module_id: Unique identifier for the module the question belongs to.
            difficulty: Difficulty level (1=Easy, 2=Medium, 3=Hard).

        Returns:
            The generated question data as a dictionary (conforming to template structure)
            or None if generation failed, was blocked, or response was invalid.
        """
        # Basic check for supported type before proceeding
        if question_type not in QUESTION_TYPES:
            logger.error(
                f"Invalid question type requested: {question_type}. Supported types: {QUESTION_TYPES}"
            )
            return None

        # Ensure the model is loaded
        if self.model is None:
            logger.error(
                "Gemini model is not initialized. API configuration might have failed."
            )
            return None

        try:
            # Create the prompt for the LLM
            prompt = self._create_prompt(
                topic, question_type, tags, module_id, difficulty
            )
            logger.debug(
                f"Generated prompt for {question_type}:\n{prompt[:500]}..."
            )  # Log start of prompt

            # Configure generation parameters
            generation_config = {
                "max_output_tokens": 4096,  # Set a reasonable limit for response length
                "temperature": 0.7,  # Control creativity vs predictability
                # "response_mime_type": "application/json", # Rely on prompt instruction and extraction
            }

            # Safety settings to control content generation
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
            ]

            # Make the API call to generate content
            logger.info(
                f"Calling Gemini API for '{tags}' ({question_type}, difficulty {difficulty})..."
            )
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,  # type: ignore
                safety_settings=safety_settings,
            )

            # Check if the response was blocked or is empty
            if (
                not response
                or not response.candidates
                or not response.candidates[0].content
            ):
                logger.warning(
                    f"Gemini response blocked or empty for '{topic}' ({question_type}, difficulty {difficulty})."
                )
                if response and response.prompt_feedback:
                    logger.warning(f"Prompt Feedback: {response.prompt_feedback}")
                if (
                    response
                    and response.candidates
                    and response.candidates[0].finish_reason
                ):
                    logger.warning(
                        f"Finish reason: {response.candidates[0].finish_reason}"
                    )
                return None

            # Extract and parse the JSON from the response text
            question_data = self._extract_json_from_response(response.text)

            # Validate if JSON extraction/parsing was successful
            if not question_data:
                logger.error(
                    f"Failed to extract valid JSON from response for '{topic}' ({question_type}, difficulty {difficulty}). Raw text received:\n{response.text}..."
                )
                return None

            logger.debug(
                f"Successfully parsed JSON for {question_type}. Performing post-generation processing."
            )

            # --- Post-generation Validation and Normalization ---
            # These steps ensure consistency and add/correct fields the model might miss or misunderstand.

            # Force essential fields based on the request parameters
            question_data["moduleId"] = module_id
            question_data["tags"] = tags
            question_data["difficultyLevel"] = difficulty
            question_data["type"] = (
                question_type  # Ensure the type in the JSON matches the requested type
            )

            # Assign or correct points based on difficulty if missing or invalid in response
            current_points = question_data.get("points")
            if not isinstance(current_points, (int, float)) or current_points <= 0:
                assigned_points = self._assign_points(difficulty)
                question_data["points"] = assigned_points
                logger.debug(
                    f"Assigned default points ({assigned_points}) for difficulty {difficulty}."
                )
            else:
                logger.debug(f"Using points provided by model: {current_points}")

            # Ensure explanation field structure exists and has default values if needed
            if "explanation" not in question_data or not isinstance(
                question_data["explanation"], dict
            ):
                question_data["explanation"] = {}
            question_data["explanation"].setdefault("en", "No explanation provided.")
            question_data["explanation"].setdefault(
                "hi", "कोई स्पष्टीकरण प्रदान नहीं किया गया है।"
            )

            # Basic structural validation (can be extended for more specific type checks)
            if question_type in ["MCQ", "MOC", "ARRANGE"]:
                if not isinstance(question_data.get("en", {}).get("options"), list):
                    logger.warning(
                        f"Question type {question_type} expected 'options' list in 'en', but not found or invalid type."
                    )
                if not isinstance(question_data.get("hi", {}).get("options"), list):
                    logger.warning(
                        f"Question type {question_type} expected 'options' list in 'hi', but not found or invalid type."
                    )
            # Add more specific validation for other types (e.g., correctAnswer format) as needed

            logger.info(
                f"Successfully generated and validated question data for '{tags[-1]}' ({question_type}, difficulty {difficulty})."
            )
            return question_data

        except Exception as e:
            # Catch any unexpected exceptions during the generation process
            logger.error(
                f"An unexpected error occurred during question generation for '{topic}' ({question_type}, difficulty {difficulty}): {e}",
                exc_info=True,  # Log the full traceback
            )
            # Optionally log the raw response text if available before the error occurred
            if "response" in locals() and hasattr(response, "text"):  # type: ignore
                logger.debug(f"Raw response text before error: {response.text[:1000]}{'...' if len(response.text) > 1000 else ''}")  # type: ignore
            return None


class QuestionManager:
    """
    Orchestrates the question generation process.
    Handles batch generation, derives module IDs, and manages storage options
    (saving to file and pushing to a database API).
    """

    def __init__(
        self,
        api_key: str,
        db_endpoint: Optional[str] = None,  # Database endpoint is optional
    ):
        """
        Initialize the Question Manager.

        Args:
            api_key: Google Gemini API key.
            db_endpoint: Optional URL endpoint for database storage API.
        """
        # Initialize the question generator instance
        self.generator = QuestionGenerator(api_key)
        self.db_endpoint = db_endpoint
        if self.db_endpoint:
            logger.info(f"Database endpoint configured: {self.db_endpoint}")
        else:
            logger.info("No database endpoint configured. DB push will be skipped.")

    def _derive_module_id(self, tags: List[str]) -> str:
        """
        Derive a module ID string from a list of tags.
        This provides a consistent, simplified identifier based on the question's
        categorization, suitable for grouping questions.
        Prioritizes tags containing 'class-' and then the first few non-class tags.

        Args:
            tags: The list of tags for the question set.

        Returns:
            A lowercase, hyphen-separated string suitable as a module ID.
        """
        # Separate class-specific tags from general topic tags
        class_tags = [tag for tag in tags if "class-" in tag.lower()]
        other_tags = [tag for tag in tags if "class-" not in tag.lower()]

        module_id_parts = []
        # Include the first class tag if available
        if class_tags:
            module_id_parts.append(class_tags[0])

        # Include up to the first two other relevant tags
        module_id_parts.extend(other_tags[:2])

        # Fallback to a default if no relevant tags are found
        if not module_id_parts:
            module_id_parts = ["general"]
            logger.warning(
                "No relevant tags found to derive module ID. Using 'general'."
            )

        # Format the parts into a valid ID string
        return (
            "-".join(module_id_parts).lower().replace(" ", "-").replace("_", "-")
        )  # Normalize formatting

    def generate_questions(
        self,
        tags: List[str],
        num_questions: int,
        question_types: List[str],
        difficulty_input: Union[int, List[int]],
    ) -> List[Dict[str, Any]]:
        """
        Generates a specified number of questions, distributing generation across
        the provided question types and difficulty levels.

        Args:
            tags: List of tags for classification, used for context and module ID.
            num_questions: The total number of questions to attempt to generate.
            question_types: List of supported question types to include in generation.
            difficulty_input: Single integer (1-3) or a list of integers (1-3)
                              specifying the target difficulty levels.

        Returns:
            A list containing the generated question data dictionaries. Returns
            an empty list if no questions could be generated or inputs were invalid.
        """
        # Validate question types against supported types (already done in parse_arguments, but good practice)
        valid_question_types = [qt for qt in question_types if qt in QUESTION_TYPES]
        if not valid_question_types:
            # This should have been caught earlier, but defensively check
            logger.error(
                f"No valid question types provided. Supported types: {QUESTION_TYPES}. Generation aborted."
            )
            return []
        # Use only the valid types for generation
        question_types_to_use = valid_question_types

        # Determine the specific difficulty levels to target for each question
        difficulties_to_use: List[int] = []
        if isinstance(difficulty_input, int):
            # If a single difficulty is given, use it for all questions
            # Validation for 1-3 range done in parse_arguments
            difficulties_to_use = [difficulty_input] * num_questions
        elif isinstance(difficulty_input, list) and difficulty_input:
            # If a list is given, select randomly from valid difficulties in the list
            # Validation for 1-3 range and non-empty list done in parse_arguments
            difficulties_to_use = random.choices(
                difficulty_input, k=num_questions
            )  # difficulty_input is already validated list
        else:
            # This case should be caught by parse_arguments, but defensively check
            logger.error(
                "Invalid difficulty input. Must be an integer (1-3) or a list of integers (1-3). Generation aborted."
            )
            return []

        # Derive a consistent module ID for this batch of questions
        module_id = self._derive_module_id(tags)
        questions: List[Dict[str, Any]] = (
            []
        )  # List to store successfully generated questions
        generated_count = 0  # Counter for successful generations

        logger.info(
            f"Starting generation of {num_questions} questions for module '{module_id}' "
            f"with tags {tags}. Target types: {question_types_to_use}. Target difficulties: {difficulties_to_use}"
        )

        # Prepare a list of generation tasks (type, difficulty)
        # Cycle through available question types and pair with determined difficulties
        types_cycle = (
            question_types_to_use * ((num_questions // len(question_types_to_use)) + 1)
        )[:num_questions]
        # Shuffle the types to distribute them randomly across the batch
        random.shuffle(types_cycle)

        # Ensure we have exactly num_questions pairs
        generation_tasks: List[Tuple[str, int]] = list(
            zip(types_cycle, difficulties_to_use)
        )
        if len(generation_tasks) < num_questions:
            # This shouldn't happen with the current logic, but safety check
            logger.warning(
                f"Planned generation tasks ({len(generation_tasks)}) less than requested ({num_questions})."
            )
            # Truncate or pad if necessary, though zip should handle this correctly up to the shortest list

        logger.debug(f"Planned generation tasks queue: {generation_tasks}")

        # Iterate through the planned tasks and attempt to generate each question
        for i, (q_type, diff) in enumerate(generation_tasks):
            # Select a specific topic from the tags for the prompt's 'Specific Topic/Focus' field
            # Prioritize non-class tags if available
            available_topics = [t for t in tags if "class-" not in t.lower()]
            # Choose a topic randomly; fall back to the first tag or 'general' if no specific topics found
            topic = (
                random.choice(available_topics)
                if available_topics
                else (tags[0] if tags else "general")
            )

            logger.info(
                f"Attempting question {i+1}/{num_questions} (Type: {q_type}, Difficulty: {diff}, Topic: '{topic}')..."
            )

            # Call the QuestionGenerator to get a single question
            question_data = self.generator.generate_question(
                topic=topic,
                question_type=q_type,
                tags=tags,
                module_id=module_id,
                difficulty=diff,
            )

            # Process the result of the generation attempt
            if question_data:
                questions.append(
                    question_data
                )  # Add the generated question to the list
                generated_count += 1
                logger.info(
                    f"Successfully generated question {generated_count}. Total collected: {generated_count}/{num_questions}."
                )
            else:
                # Log failure specifically for this attempt
                logger.warning(
                    f"Failed to generate question {i+1}/{num_questions} (Type: {q_type}, Difficulty: {diff}). Skipping."
                )

            # Optional: Add a small delay between API calls to respect rate limits or cool down
            # import time
            # time.sleep(0.5) # Example: wait for 0.5 seconds

        logger.info(
            f"Finished all generation attempts. Successfully generated {generated_count} out of {num_questions} requested questions."
        )

        return questions

    def save_to_file(
        self, questions: List[Dict[str, Any]], output_file: str = DEFAULT_OUTPUT_FILE
    ) -> bool:
        """
        Save the list of generated questions to a JSON file.

        Args:
            questions: A list of question data dictionaries.
            output_file: The path to the file where questions will be saved.

        Returns:
            True if saving was successful, False otherwise.
        """
        if not questions:
            logger.info("No questions available to save to file. Skipping file save.")
            return False

        try:
            # Ensure the directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"Created output directory: {output_dir}")

            # Write the questions list to the JSON file
            with open(output_file, "w", encoding="utf-8") as f:
                # Use ensure_ascii=False to allow non-ASCII characters (like Hindi)
                # use indent=2 for pretty printing the JSON
                json.dump(questions, f, ensure_ascii=False, indent=2)
            logger.info(
                f"Successfully saved {len(questions)} questions to {output_file}"
            )
            return True
        except IOError as e:
            logger.error(
                f"Error saving questions to file {output_file}: {e}", exc_info=True
            )
            return False
        except Exception as e:
            # Catch any other unexpected errors during file saving
            logger.error(
                f"An unexpected error occurred while saving to file: {e}", exc_info=True
            )
            return False

    # CHANGE: Modified to push questions one by one and only unique ones
    def push_to_database(
        self, questions: List[Dict[str, Any]]
    ) -> int:  # Changed return type to int (count)
        """
        Push unique questions one by one to a configured database API endpoint
        using an HTTP POST request. Uniqueness is checked based on English and Hindi text
        within the provided list of questions.

        Args:
            questions: A list of question data dictionaries to be sent.

        Returns:
            The number of successfully pushed unique questions.
            Returns 0 if the database endpoint is not configured,
            no questions are available, or all pushes failed.
        """
        # Check if the database endpoint is configured
        if not self.db_endpoint:
            logger.warning("Database endpoint not configured. Skipping DB push.")
            return 0

        # Check if there are questions to push
        if not questions:
            logger.warning("No questions available to push to database.")
            return 0

        # Define headers for the API request
        headers = {
            "Content-Type": "application/json",
            # Add authentication headers here if required by your DB API
            # E.g., "Authorization": f"Bearer {os.environ.get('DB_API_TOKEN')}"
        }

        logger.info(
            f"Attempting to push {len(questions)} questions one by one to DB endpoint: {self.db_endpoint}"
        )

        successfully_pushed_count = 0
        # Use a set to track unique questions based on their text content
        pushed_texts = set()

        # Iterate through each question in the generated list
        for i, question in enumerate(questions):
            # Extract text for uniqueness check. Use tuple of texts as key.
            # Provide default empty strings in case 'en' or 'hi' or 'text' is missing (shouldn't happen with validation but safe)
            en_text = question.get("en", {}).get("text", "")
            hi_text = question.get("hi", {}).get("text", "")
            question_text_key = (en_text, hi_text)

            # Check for uniqueness within the current batch based on text
            if question_text_key in pushed_texts:
                logger.info(
                    f"Question {i+1}/{len(questions)}: Skipping duplicate question based on text."
                )
                continue  # Skip this question

            # Check if the question text is empty (indicates a potentially bad generation)
            if not en_text and not hi_text:
                logger.warning(
                    f"Question {i+1}/{len(questions)}: Skipping question with empty English and Hindi text."
                )
                continue  # Skip questions with no text

            logger.info(
                f"Question {i+1}/{len(questions)}: Attempting to push question (Type: {question.get('type')}, Difficulty: {question.get('difficultyLevel')})..."
            )

            try:
                # The payload for a single question endpoint is the question object in array
                payload = [question]  # Wrap in a list for the API
                print(f"Payload: {payload}")  # Debugging line to check payload
                response = requests.post(
                    self.db_endpoint, headers=headers, json=payload
                )

                # Raise an HTTPError for bad responses (4xx or 5xx status codes)
                response.raise_for_status()

                # If successful, add the text key to the set and increment count
                pushed_texts.add(question_text_key)
                successfully_pushed_count += 1
                logger.info(f"Question {i+1}/{len(questions)}: Successfully pushed.")

            except requests.exceptions.RequestException as e:
                # Handle errors for this specific question push
                logger.error(
                    f"Question {i+1}/{len(questions)}: Error pushing question to database API: {e}"
                )
                # Log details about the API response if available
                if hasattr(e, "response") and e.response is not None:
                    logger.error(
                        f"Question {i+1}/{len(questions)}: DB API Response Status Code: {e.response.status_code}"
                    )
                    try:
                        # Attempt to log the response body
                        logger.error(
                            f"Question {i+1}/{len(questions)}: DB API Response Body: {e.response.text[:500]}{'...' if len(e.response.text) > 500 else ''}"
                        )
                    except Exception as text_e:
                        logger.error(
                            f"Question {i+1}/{len(questions)}: Could not read DB API response body: {text_e}"
                        )
                # Continue to the next question even if one fails
                continue
            except Exception as e:
                # Catch any other unexpected errors during the push process for this question
                logger.error(
                    f"Question {i+1}/{len(questions)}: An unexpected error occurred during DB push: {e}",
                    exc_info=True,
                )
                # Continue to the next question
                continue

        logger.info(
            f"Finished attempting DB pushes. Successfully pushed {successfully_pushed_count} unique questions."
        )

        # Returning the count of successfully pushed unique questions
        return successfully_pushed_count


# --- Command Line Interface ---


def parse_arguments():
    """
    Parse command line arguments using argparse.
    Includes custom type validation for lists and difficulties.

    Returns:
        An argparse.Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate educational questions using Google's Gemini API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Show default values in help message
    )

    # Argument for Gemini API Key
    # Prioritize environment variable, fallback to command line argument.
    api_key_default = os.environ.get(ENV_VAR_API_KEY)
    parser.add_argument(
        "--api-key",
        default=api_key_default,
        help=f"Google Gemini API Key. Can also be set via {ENV_VAR_API_KEY} environment variable.",
        # Require argument only if the environment variable is NOT set
        required=api_key_default is None,
    )

    # Argument for Database Endpoint
    # Prioritize environment variable, fallback to argument, default is None.
    db_endpoint_default = os.environ.get(ENV_VAR_DB_ENDPOINT)
    parser.add_argument(
        "--db-endpoint",
        default=db_endpoint_default,
        # CHANGE: Updated help message to reflect the new endpoint expectation (single question)
        help=f"Optional Database API endpoint for storing questions (e.g., http://localhost:8000/api/v1/create-question). Can also be set via {ENV_VAR_DB_ENDPOINT} environment variable. This endpoint is expected to accept a single question JSON object per POST request.",
    )

    # Argument for tags - Make it truly optional by having default=None
    # Validation for the content of the list is moved to main()
    parser.add_argument(
        "--tags",
        type=parse_list_arg,  # Use custom parser for list format
        default=None,  # Default is explicitly None if not provided by the user
        help="Optional list of tags (JSON or Python-style string), e.g., \"['class-10', 'physics', 'motion']\". If not provided, tags are fetched from a pre-configured API endpoint.",
    )

    # Argument for number of questions
    parser.add_argument(
        "--num-questions",
        type=int,
        default=DEFAULT_NUM_QUESTIONS,
        help="Number of questions to attempt to generate.",
    )

    # Argument for question types
    parser.add_argument(
        "--question-types",
        type=parse_list_arg,  # Use custom parser for list format
        default=DEFAULT_QUESTION_TYPES,
        help=f"List of question types to generate (JSON or Python-style string). Supported types: {', '.join(QUESTION_TYPES)}. e.g., \"['MCQ', 'TRUE_FALSE']\"",
    )

    # Argument for difficulty level(s)
    parser.add_argument(
        "--difficulty",
        type=parse_difficulty_arg,  # Use custom parser for integer or list of integers
        default=DEFAULT_DIFFICULTIES,
        help="Difficulty level(s): integer 1-3 or list of integers 1-3 (JSON or Python-style string), e.g., '2' or '[1, 3]'.",
    )

    # Argument for output file path
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSON file path to save generated questions.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # --- Post-Parsing Validation for Critical Arguments ---
    # Ensure API key is present (required if not in env)
    if not args.api_key:
        logger.error(
            f"API key is required. Set {ENV_VAR_API_KEY} environment variable or use the --api-key argument."
        )
        sys.exit(1)  # Exit if critical configuration is missing

    # Validate the format and content of the question-types list
    if not isinstance(args.question_types, list) or not all(
        qt in QUESTION_TYPES for qt in args.question_types
    ):
        logger.error(
            f"Invalid value for --question-types. Must be a list containing only supported types: {QUESTION_TYPES}"
        )
        sys.exit(1)  # Exit on invalid question types
    # Ensure the question type list is not empty after validation
    if not args.question_types:
        logger.error("No valid question types specified. Generation aborted.")
        sys.exit(1)

    # Validate difficulty argument content
    if isinstance(args.difficulty, list):
        if not args.difficulty or not all(
            isinstance(d, int) and 1 <= d <= 3 for d in args.difficulty
        ):
            logger.error(
                "--difficulty list must contain only integers between 1 and 3."
            )
            sys.exit(1)  # Exit on invalid difficulty list content
    elif not isinstance(args.difficulty, int) or not 1 <= args.difficulty <= 3:
        # This case should ideally be caught by parse_difficulty_arg, but double-check
        logger.error(
            "--difficulty must be an integer between 1 and 3, or a list of such integers."
        )
        sys.exit(1)

    # Note: Validation for 'tags' is moved to main() after checking if API fetch is needed

    return args


def get_tags_from_api() -> List[str]:
    """
    Fetch tags from an external API endpoint (hardcoded URL).
    This function attempts to retrieve a list of relevant tags to use for generation
    if they are not explicitly provided via command line arguments.

    Returns:
        A list of strings representing tags, or an empty list if fetching fails.
    """
    logger.info(f"Attempting to fetch tags from API: {TAGS_API_URL}")
    try:
        response = requests.get(TAGS_API_URL)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # Attempt to extract the list of tags from the specific API response structure
        # Adjust this logic based on the actual API response format if needed
        if isinstance(data, dict) and "tags" in data and isinstance(data["tags"], list):
            fetched_tags = data["tags"]
            logger.info(f"Successfully fetched {len(fetched_tags)} tags from API.")
            logger.debug(f"Fetched tags: {fetched_tags}")
            # Filter out any non-string or empty string items just in case
            return [
                tag.strip()
                for tag in fetched_tags
                if isinstance(tag, str) and tag.strip()
            ]
        else:
            logger.warning(
                "API response structure unexpected or missing/invalid 'tags' field."
            )
            # Log the start of the raw response for debugging API issues
            raw_data_str = json.dumps(data, indent=2)
            logger.debug(
                f"Raw API response (first 500 chars):\n{raw_data_str[:500]}{'...' if len(raw_data_str) > 500 else ''}"
            )
            return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching tags from API: {e}")
        # Log response status/body if available on error
        if hasattr(e, "response") and e.response is not None:
            logger.error(f"Tags API Response Status Code: {e.response.status_code}")
            try:
                logger.error(
                    f"Tags API Response Body: {e.response.text[:500]}{'...' if len(e.response.text) > 500 else ''}"
                )
            except Exception as text_e:
                logger.error(f"Could not read Tags API response body: {text_e}")
        return []
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while fetching tags: {e}", exc_info=True
        )
        return []


def main():
    """
    Main entry point for the script.
    Parses arguments, handles fetching tags if not provided, initializes the
    QuestionManager, generates questions, and handles saving to a file and/or
    pushing to a database API based on configuration.
    """
    # Parse command line arguments
    args = parse_arguments()

    # --- Tag Handling Logic ---
    # If tags were NOT provided via the command line (--tags is still None),
    # attempt to fetch them from the API. Otherwise, use the tags provided by the user.
    if args.tags is None:
        logger.info("No --tags argument provided. Attempting to fetch tags from API.")
        fetched_tags = get_tags_from_api()
        if fetched_tags:
            args.tags = fetched_tags
            logger.info(f"Using {len(args.tags)} tags fetched from API.")
        else:
            # Fallback if API fetch failed or returned no valid tags
            logger.warning(
                f"Failed to fetch valid tags from API. Using default tags: {DEFAULT_TAGS}."
            )
            args.tags = DEFAULT_TAGS
    else:
        logger.info(f"Using tags provided via command line: {args.tags}")

    # --- Final Validation for Tags (now that we know the source) ---
    # Ensure that the final list of tags (whether from CLI, API, or default) is valid.
    # It must be a non-empty list containing only non-empty strings.
    if (
        not isinstance(args.tags, list)
        or not args.tags
        or not all(isinstance(tag, str) and tag.strip() for tag in args.tags)
    ):
        logger.error("Tags must be a non-empty list of non-empty strings.")
        # If the fallback default tags are also invalid (which shouldn't happen but defensive check), exit.
        if args.tags == DEFAULT_TAGS:
            logger.critical("Default tags are also invalid! Exiting.")
        sys.exit(1)  # Exit if the final tag list is invalid

    # Initialize the QuestionManager with API key and optional DB endpoint
    # The manager initializes the QuestionGenerator, which handles API config and exit on failure.
    try:
        manager = QuestionManager(api_key=args.api_key, db_endpoint=args.db_endpoint)
    except SystemExit:
        # QuestionGenerator already logged the error and exited via sys.exit
        # Re-raise SystemExit to ensure the program terminates
        raise
    except Exception as e:
        logger.error(f"Failed to initialize QuestionManager: {e}", exc_info=True)
        sys.exit(1)  # Exit on unexpected initialization error

    # Generate the batch of questions
    generated_questions = manager.generate_questions(
        tags=args.tags,  # Use the tags determined above
        num_questions=args.num_questions,
        question_types=args.question_types,
        difficulty_input=args.difficulty,
    )

    # Check if any questions were successfully generated
    if not generated_questions:
        logger.warning("No questions were successfully generated during this run.")
        sys.exit(0)  # Exit gracefully if nothing was generated

    # --- Handle Output ---

    # Save generated questions to a local JSON file
    logger.info("Attempting to save generated questions to file...")
    file_saved = manager.save_to_file(generated_questions, args.output)
    if not file_saved:
        logger.error("Failed to save questions to file. Check permissions or path.")

    # Push generated questions to the database API if an endpoint is configured
    if args.db_endpoint:
        logger.info(
            "Attempting to push generated questions to database API one by one..."
        )
        # Call the modified push method
        pushed_count = manager.push_to_database(generated_questions)
        if pushed_count > 0:
            logger.info(
                f"Database push completed successfully for {pushed_count} unique questions."
            )
        else:
            # Note: This will print if 0 questions were generated *or* if all pushes failed.
            # The logs within push_to_database provide more detail.
            logger.warning(
                "Database push completed with 0 successful unique pushes. See previous logs for details."
            )
    else:
        logger.info("Database endpoint not provided. Skipping database push.")

    logger.info("Question generation and storage process finished.")


# Standard Python entry point check
if __name__ == "__main__":
    main()
