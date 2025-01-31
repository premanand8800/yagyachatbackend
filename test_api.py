# app/tests/test_validation.py
import pytest
from typing import Dict
from app.nodes.validation import (
    InputValidator,
    ValidationScope,
    ValidationNode,
    GraphState,
    InputType
)
from app.models.user_input import UserInput
import asyncio
import json

# Create a mock LLM call function for testing
async def mock_llm_call(prompt: str, model="mixtral-8x7b-32768") -> Dict:
    if "invalid" in prompt.lower():
        return json.loads(json.dumps( {
            "is_valid": False,
            "has_background": False,
            "has_goals": False,
            "background_completeness": 0.2,
            "goals_clarity": 0.3,
            "missing_elements": ["Missing background", "Missing goals"],
            "suggestions": ["Add more background", "Add specific goals"],
            "clarification_questions": ["What is your educational background?", "What are your specific goals?"],
            "input_type": "invalid_input"
        }))
    if "preference" in prompt.lower():
        return json.loads(json.dumps({
            "detected_operation": "add",
            "detected_preferences": [
                {
                    "category": "technology",
                    "value": "python"
                }
            ],
            "confidence_score": 0.9,
            "needs_clarification": False,
            "clarification_questions": []
        }))
    if "rewrite" in prompt.lower():
        return json.loads(json.dumps({
           'rewritten_input': "This is a rewritten input"
        }))
    if "enhance" in prompt.lower():
        return json.loads(json.dumps({
           'enhanced_input': "This is an enhanced input"
        }))
    return json.loads(json.dumps({
        "is_valid": True,
        "has_background": True,
        "has_goals": True,
        "background_completeness": 0.8,
        "goals_clarity": 0.7,
        "missing_elements": [],
        "suggestions": ["Consider adding more specific timeline for your goals"],
        "clarification_questions": [],
        "input_type": "new_query"
    }))


def create_test_validation_node(mocker):
    """Creates a test validation node with a mocked LLM call"""
    scope = ValidationScope()
    validator = InputValidator(scope=scope)
    # Mock the llm call in validator
    mocker.patch("app.utils.llm_client.run_llm_call", side_effect=mock_llm_call)
    mocker.patch("app.utils.llm_client.run_llm_analysis", side_effect=mock_llm_call)
    return ValidationNode(validator)

@pytest.mark.asyncio
async def test_valid_input(mocker):
    """Test case for a valid input"""
    validation_node = create_test_validation_node(mocker)
    state = GraphState(user_input="I have a background in software engineering and want to learn AI", messages=[], next_step="")
    result = await validation_node(state)
    print("Test Valid Input - Passed")
    print("Validation Result:", result['validation_result'])
    assert result['validation_result']['is_valid'] is True
    assert result['next_step'] == 'analysis_node'

@pytest.mark.asyncio
async def test_invalid_input(mocker):
    """Test case for an invalid input"""
    validation_node = create_test_validation_node(mocker)
    state = GraphState(user_input="invalid", messages=[], next_step="")
    result = await validation_node(state)
    print("Test Invalid Input - Passed")
    print("Validation Result:", result['validation_result'])
    assert result['validation_result']['is_valid'] is False
    assert result['next_step'] == 'clarification_node'

@pytest.mark.asyncio
async def test_rewrite_request(mocker):
    """Test case for a rewrite request"""
    validation_node = create_test_validation_node(mocker)
    state = GraphState(user_input="rewrite this", messages=[], next_step="")
    result = await validation_node(state)
    print("Test Rewrite Request - Passed")
    print("Validation Result:", result['validation_result'])
    assert result['validation_result']['input_type'] == InputType.REWRITE_REQUEST
    assert result['next_step'] == 'input_processed'

@pytest.mark.asyncio
async def test_enhance_request(mocker):
    """Test case for an enhance request"""
    validation_node = create_test_validation_node(mocker)
    state = GraphState(user_input="enhance this", messages=[], next_step="")
    result = await validation_node(state)
    print("Test Enhance Request - Passed")
    print("Validation Result:", result['validation_result'])
    assert result['validation_result']['input_type'] == InputType.INPUT_ENHANCEMENT
    assert result['next_step'] == 'input_processed'

@pytest.mark.asyncio
async def test_preference_update(mocker):
    """Test case for a preference update"""
    validation_node = create_test_validation_node(mocker)
    state = GraphState(user_input="I prefer python", messages=[], next_step="")
    result = await validation_node(state)
    print("Test Preference Update - Passed")
    print("Validation Result:", result['validation_result'])
    assert result['validation_result']['input_type'] == InputType.PREFERENCE_UPDATE
    assert result['next_step'] == 'preference_processor'

@pytest.mark.asyncio
async def test_error_handling(mocker):
    """Test case to ensure that the system handles errors"""
    validation_node = create_test_validation_node(mocker)
    state = GraphState(user_input="", messages=[], next_step="")
    result = await validation_node(state)
    print("Test Error Handling - Passed")
    print("Next Step:", result['next_step'])
    assert result['next_step'] == 'error_handler'