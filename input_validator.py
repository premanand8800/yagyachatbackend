from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from ..models.validation_result import ValidationResult, InputType
from ..models.user_input import UserInput
from ..models.conversation_memory import ConversationMemory, ConversationTurn
from ..models.preference import PreferenceValue, PreferenceUpdate, PreferenceOperation, PreferenceAnalysisResult
from ..utils.llm_client import get_llm_client
from ..utils.exceptions import (
    ValidationError, LLMError, PreferenceError,
    InputTypeError, GuardrailsError, ConversationMemoryError, ParsingError
)
from ..utils.content_safety import ContentSafetyChecker
import json
import re
import logging
from datetime import datetime
from typing import Dict, Any, Optional

class InputValidator:
    """Custom input validator using pattern-based analysis"""
    def __init__(self):
        self.output_parser = PydanticOutputParser(pydantic_object=ValidationResult)
        self.conversation_memory = ConversationMemory()
        self.safety_checker = ContentSafetyChecker()
        
        # Enhanced validation prompt that better handles preferences and input types
        self.validation_prompt = """
        Analyze the following user input and provide a detailed validation result.
        
        User Input: {input}
        Previous Context: {metadata}
        Recent Conversation:
        {conversation_history}
        
        Validation Requirements:
        1. Check if the input is clear and actionable
        2. Identify the type of input (new query, preference update, clarification, etc.)
        3. Detect any user preferences
        4. Assess input safety and appropriateness
        5. Evaluate background information and goals
        6. Consider conversation history for context
        
        Guidelines for Input Types:
        - NEW_QUERY: Fresh questions or requests
        - REWRITE_REQUEST: Modifications to previous queries
        - INPUT_ENHANCEMENT: Adding more details to existing query
        - CLARIFICATION_RESPONSE: Answering previous clarification questions
        - PREFERENCE_UPDATE: Expressing new preferences
        - PREFERENCE_REMOVAL: Removing previous preferences
        - PREFERENCE_QUERY: Asking about current preferences
        
        {format_instructions}
        """
        
        self.prompt = ChatPromptTemplate.from_template(
            template=self.validation_prompt,
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )

    def _detect_input_type(self, current_input: str, previous_input: str = None) -> InputType:
        """Enhanced detection of input type based on content and context"""
        current_lower = current_input.lower()
        
        # Preference-related patterns
        preference_patterns = {
            'add': [r'i prefer', r'i like', r'i want', r'i would rather', r'set preference', r'update preference'],
            'remove': [r'remove preference', r"don't prefer", r"don't like", r'remove', r'delete preference'],
            'query': [r'what are my preferences', r'show preferences', r'list preferences', r'get preferences']
        }
        
        # Check for preference operations
        for operation, patterns in preference_patterns.items():
            if any(re.search(pattern, current_lower) for pattern in patterns):
                if operation == 'add':
                    return InputType.PREFERENCE_UPDATE
                elif operation == 'remove':
                    return InputType.PREFERENCE_REMOVAL
                elif operation == 'query':
                    return InputType.PREFERENCE_QUERY
        
        # Check for clarification responses
        clarification_starters = [
            r'^yes', r'^no', r'^actually', r'^i meant', r'^to clarify',
            r'^let me clarify', r'^what i meant'
        ]
        if previous_input and any(re.search(pattern, current_lower) for pattern in clarification_starters):
            return InputType.CLARIFICATION_RESPONSE
        
        # Check for rewrite requests
        rewrite_patterns = [r'rewrite', r'rephrase', r'change', r'modify']
        if any(re.search(pattern, current_lower) for pattern in rewrite_patterns):
            return InputType.REWRITE_REQUEST
        
        # Check for input enhancements
        enhancement_patterns = [r'add more', r'enhance', r'elaborate', r'provide more']
        if any(re.search(pattern, current_lower) for pattern in enhancement_patterns):
            return InputType.INPUT_ENHANCEMENT
        
        # Default to new query if no other type matches
        return InputType.NEW_QUERY

    def _is_clarification(self, current_input: str, previous_input: str) -> bool:
        """Check if current input is clarifying a previous input"""
        # Check if current input starts with clarification indicators
        clarification_starters = ["yes", "no", "actually", "i meant", "to clarify"]
        return any(current_input.lower().startswith(start) for start in clarification_starters)

    def _extract_preferences(self, input_text: str) -> dict:
        """Extract user preferences from input"""
        preferences = {}
        
        # Movie preferences
        if "movie" in input_text.lower():
            genres = ["action", "comedy", "drama", "horror", "fantasy", "sci-fi", "bollywood", "hollywood"]
            for genre in genres:
                if genre in input_text.lower():
                    preferences["movie_genre"] = genre
                    
        # Other preferences can be added here
        
        return preferences

    def _apply_safety_rules(self, validation_result: ValidationResult, user_input: UserInput) -> ValidationResult:
        """Apply additional safety rules to the validation result"""
        # Check for unsafe content using word boundaries
        unsafe_patterns = [
            r'\brm\s+-rf\b',  # rm -rf command
            r'\bsudo\b',      # sudo command
            r'\bdel\b',       # del command
            r'\bformat\b',    # format command
            r'\bDROP\s+TABLE\b',  # SQL DROP TABLE
            r'\bDELETE\s+FROM\b',  # SQL DELETE FROM
            r'\b(porn|xxx|adult)\b',  # Adult content
            r'\b(hack|crack|exploit)\b'  # Security-related
        ]
        
        if any(re.search(pattern, user_input.raw_input, re.IGNORECASE) for pattern in unsafe_patterns):
            return ValidationResult(
                is_valid=False,
                input_type=InputType.INVALID_INPUT,
                error_message="The input contains harmful or inappropriate content.",
                has_background=False,
                has_goals=False,
                background_completeness=0.0,
                goals_clarity=0.0,
                clarity_score=0.0,
                safety_score=0.0,
                detected_preferences={},
                validation_details={"unsafe_patterns_found": True}
            )
        return validation_result

    def _parse_llm_response(self, response_text: str) -> ValidationResult:
        """Parse LLM response into ValidationResult, handling nulls and missing fields"""
        try:
            # Extract JSON from the response if it's wrapped in markdown
            json_str = response_text
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            
            # Parse JSON response
            data = json.loads(json_str)
            
            # Ensure all numeric fields have valid values
            numeric_fields = [
                'background_completeness', 'goals_clarity', 
                'clarity_score', 'safety_score'
            ]
            for field in numeric_fields:
                if field not in data or data[field] is None:
                    data[field] = 0.0

            # Create ValidationResult
            return ValidationResult(**data)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response. Raw response: {response_text}")

    def _combine_with_context(self, current_input: str, previous_context: Optional[str]) -> str:
        """Combines current input with previous context if relevant"""
        if not previous_context:
            return current_input
            
        # Check if current input is a short response to a clarification
        is_short_response = len(current_input.split()) <= 2
        
        # Get previous clarification questions
        prev_result = self.conversation_memory.get_last_validation_result()
        if prev_result and prev_result.clarification_questions:
            last_question = prev_result.clarification_questions[0]
            
            # If previous question was about type/field/specifics
            if any(keyword in last_question.lower() for keyword in ['type', 'field', 'specific', 'which']):
                if is_short_response:
                    # Combine previous goal with specific response
                    if 'engineer' in previous_context.lower():
                        return f"I want to be a {current_input} engineer"
                    elif 'developer' in previous_context.lower():
                        return f"I want to be a {current_input} developer"
                    # Add more profession patterns as needed
        
        return current_input

    async def validate_input(self, user_input: UserInput) -> ValidationResult:
        """Validate user input using pattern-based analysis"""
        try:
            # Input sanity checks
            if not user_input.raw_input or not isinstance(user_input.raw_input, str):
                raise ValidationError("Invalid input: Text must be a non-empty string")
            
            if len(user_input.raw_input.strip()) == 0:
                raise ValidationError("Input contains only whitespace")

            # Get previous input from metadata
            previous_input = user_input.metadata.get("previous_input") if user_input.metadata else None
            
            # Content safety check first
            try:
                is_safe, safety_result = self.safety_checker.check_content(user_input.raw_input)
            except Exception as e:
                raise GuardrailsError(
                    "Content safety check failed",
                    {"error": str(e), "input": user_input.raw_input}
                )

            # Detect background and goals
            background_patterns = [
                r'\b(background|experience|worked|familiar|know|learned)\b',
                r'I (have|had|am|was)',
                r'(studied|learning|learned)',
                r'(using|used|worked with)'
            ]
            
            goal_patterns = [
                r'\b(want|need|trying|going|plan)\b.*\b(to|for)\b',
                r'\b(goal|aim|objective|purpose)\b',
                r'(build|create|develop|implement|make)',
                r'(looking|searching|seeking)\s+for'
            ]
            
            has_background = any(re.search(pattern, user_input.raw_input, re.IGNORECASE) for pattern in background_patterns)
            has_goals = any(re.search(pattern, user_input.raw_input, re.IGNORECASE) for pattern in goal_patterns)
            
            # Calculate scores
            background_completeness = 0.8 if has_background else 0.0
            goals_clarity = 0.9 if has_goals else 0.0
            clarity_score = (background_completeness + goals_clarity) / 2 if (has_background or has_goals) else 0.0
            
            # Detect input type
            input_type = self._detect_input_type(user_input.raw_input, previous_input)
            
            # Create validation result
            return ValidationResult(
                is_valid=is_safe,
                input_type=input_type,
                has_background=has_background,
                has_goals=has_goals,
                background_completeness=background_completeness,
                goals_clarity=goals_clarity,
                clarity_score=clarity_score,
                safety_score=1.0 if is_safe else 0.0,
                error_message=None if is_safe else "Input contains unsafe content",
                context_score=1.0 if previous_input else 0.0,
                suggestions=[],
                clarification_questions=[],
                detected_preferences={},
                validation_details={"safety_result": safety_result}
            )
            
        except Exception as e:
            logging.error(f"Validation error: {str(e)}")
            return ValidationResult(
                is_valid=False,
                input_type=InputType.INVALID_INPUT,
                has_background=False,
                has_goals=False,
                background_completeness=0.0,
                goals_clarity=0.0,
                clarity_score=0.0,
                safety_score=0.0,
                error_message=str(e),
                context_score=0.0,
                suggestions=[],
                clarification_questions=[],
                detected_preferences={},
                validation_details={"error": str(e)}
            )

    def analyze_preferences(self, text: str, input_type: InputType) -> PreferenceAnalysisResult:
        """Analyze text for preferences"""
        try:
            # Simple pattern matching for now
            preference_patterns = {
                'want': r'(?:i\s+)?want\s+to\s+(.*)',
                'goal': r'(?:my\s+)?goal\s+is\s+to\s+(.*)',
                'like': r'(?:i\s+)?(?:would\s+)?like\s+to\s+(.*)',
                'prefer': r'(?:i\s+)?prefer\s+(.*)',
                'remove': r'(?:remove|delete|don\'t\s+want)\s+(.*)'
            }
            
            detected_prefs = []
            operation = PreferenceOperation.UPDATE
            confidence = 0.0
            needs_clarification = False
            questions = []
            
            # Check for removal patterns first
            for pattern in preference_patterns.values():
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    pref_text = match.group(1).strip()
                    if 'remove' in pattern or 'don\'t want' in text.lower():
                        operation = PreferenceOperation.REMOVE
                    detected_prefs.append(
                        PreferenceValue(
                            value=pref_text,
                            confidence=0.8,
                            source="user_input"
                        )
                    )
                    
            # Set confidence based on clarity
            if detected_prefs:
                confidence = 0.8
                # Add clarification questions for vague preferences
                if any(len(p.value.split()) < 2 for p in detected_prefs):
                    needs_clarification = True
                    questions.append("Could you be more specific about your preference?")
                if operation == PreferenceOperation.REMOVE and not any('remove' in p.value.lower() for p in detected_prefs):
                    questions.append("Are you sure you want to remove this preference?")
            else:
                # No clear preferences found
                confidence = 0.3
                needs_clarification = True
                questions.append("Could you rephrase that with 'I want' or 'I prefer'?")
            
            return PreferenceAnalysisResult(
                detected_operation=operation,
                detected_preferences=detected_prefs,
                confidence_score=confidence,
                needs_clarification=needs_clarification,
                clarification_questions=questions
            )
            
        except Exception as e:
            logging.error(f"Error analyzing preferences: {str(e)}")
            raise PreferenceError(
                "Failed to analyze preferences",
                {"error": str(e), "input": text}
            )

    def _create_error_result(self, error_type: str, message: str, details: Dict[str, Any]) -> ValidationResult:
        """Creates a detailed error result"""
        try:
            return ValidationResult(
                is_valid=False,
                has_background=False,
                has_goals=False,
                background_completeness=0.0,
                goals_clarity=0.0,
                clarity_score=0.0,
                safety_score=0.0,
                context_score=0.0,
                input_type=InputType.INVALID_INPUT,
                error_message=message,
                validation_details={
                    "error_type": error_type,
                    "details": details,
                    "timestamp": datetime.now().isoformat()
                },
                suggestions=[
                    "Please try again with valid input",
                    "If the error persists, try rephrasing your request"
                ],
                clarification_questions=[
                    "Could you please provide your input in a different way?"
                ]
            )
        except Exception as e:
            logging.error(f"Error creating error result: {str(e)}")
            # Fallback error result with minimal fields
            return ValidationResult(
                is_valid=False,
                has_background=False,
                has_goals=False,
                background_completeness=0.0,
                goals_clarity=0.0,
                clarity_score=0.0,
                safety_score=0.0,
                context_score=0.0,
                input_type=InputType.INVALID_INPUT,
                error_message="Internal validation error"
            )

    def _create_blocked_result(self, guardrails_result: Dict) -> ValidationResult:
        """Creates a result for blocked content"""
        return ValidationResult(
            is_valid=False,
            has_background=False,
            has_goals=False,
            background_completeness=0.0,
            goals_clarity=0.0,
            clarity_score=0.0,
            safety_score=0.0,
            context_score=0.0,
            input_type=InputType.INVALID_INPUT,
            error_message="Content blocked by safety checks",
            missing_elements=['Blocked by guardrails'],
            suggestions=[
                "Please rephrase your query",
                "Please avoid sensitive or prohibited topics"
            ],
            clarification_questions=[
                "Could you please rephrase your request in a more appropriate way?"
            ],
            guardrails_result=guardrails_result
        )

    def _create_preference_clarification_result(self, analysis: PreferenceAnalysisResult) -> ValidationResult:
        """Creates result for preference clarification needed"""
        return ValidationResult(
            is_valid=False,
            has_background=False,
            has_goals=False,
            background_completeness=0.0,
            goals_clarity=0.0,
            clarity_score=0.0,
            safety_score=1.0,
            context_score=0.0,
            input_type=InputType.PREFERENCE_UPDATE,
            error_message="Preference clarification needed",
            clarification_questions=analysis.clarification_questions,
            suggestions=[
                "Please provide more details about your preferences",
                "Try to be more specific about what you prefer"
            ]
        )
