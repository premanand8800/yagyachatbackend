
# app/nodes/validation.py
from typing import Dict, List, TypedDict, Optional, Union, Callable, Any
from datetime import datetime
import json
from enum import Enum
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field, field_validator
from nemoguardrails import LLMRails, RailsConfig
import logging
from app.utils.llm_client import run_llm_call, run_llm_analysis
from app.config.settings import (
    ANALYSIS_PROMPT_PATH,
    MAX_RETRIES,
    LOG_LEVEL,
    LOG_FORMAT
)
from app.nodes.clarification import clarification_node
from app.models.preference import PreferenceAnalysisResult, PreferenceValue, PreferenceUpdate, PreferenceOperation, PreferenceCategory
from app.models.analysis_models import AnalysisResults
from app.utils.exceptions import AnalysisError, LLMError, ValidationError
# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class InputType(str, Enum):
    """Extended enumeration of input types including preferences"""
    NEW_QUERY = "new_query"
    REWRITE_REQUEST = "rewrite_request"
    CLARIFICATION_RESPONSE = "clarification_response"
    PREFERENCE_UPDATE = "preference_update"
    PREFERENCE_REMOVAL = "preference_removal"
    PREFERENCE_QUERY = "preference_query"
    INVALID_INPUT = "invalid_input"
    INPUT_ENHANCEMENT = "input_enhancement"

class PreferenceCategory(str, Enum):
    """Categories of user preferences"""
    LEARNING_STYLE = "learning_style"
    CAREER_PATH = "career_path"
    INDUSTRY = "industry"
    TECHNOLOGY = "technology"
    WORK_ENVIRONMENT = "work_environment"
    LOCATION = "location"
    EXPERIENCE_LEVEL = "experience_level"
    ROLE_TYPE = "role_type"
    COMPANY_SIZE = "company_size"
    DOMAIN = "domain"

class PreferenceOperation(str, Enum):
    """Types of preference operations"""
    ADD = "add"
    REMOVE = "remove"
    UPDATE = "update"
    QUERY = "query"

class PreferenceValue(BaseModel):
    """Structure for preference values"""
    category: PreferenceCategory
    value: Union[str, List[str]]
    priority: Optional[int] = None
    constraints: Optional[Dict] = None
    
    @field_validator('priority')
    def validate_priority(cls, v):
        if v is not None and not (1 <= v <= 10):
            raise ValueError("Priority must be between 1 and 10")
        return v

class PreferenceUpdate(BaseModel):
    """Structure for preference updates"""
    operation: PreferenceOperation
    preferences: List[PreferenceValue]
    reason: Optional[str] = None

class ValidationScope(BaseModel):
    """Enhanced scope including preference validation"""
    max_length: int = 2000
    min_length: int = 50
    required_elements: List[str] = ["background", "goals"]
    min_background_score: float = 0.7
    min_goals_score: float = 0.7
    allowed_preference_categories: List[PreferenceCategory] = list(PreferenceCategory)
    max_preferences_per_category: int = 5
    max_total_preferences: int = 20

class ValidationResult(BaseModel):
    """Enhanced validation results including preference handling"""
    is_valid: bool = Field(description="Whether the input is valid")
    has_background: bool = Field(description="Whether background information is detected")
    has_goals: bool = Field(description="Whether goals are detected")
    background_completeness: float = Field(description="Score for background completeness (0-1)")
    goals_clarity: float = Field(description="Score for goals clarity (0-1)")
    missing_elements: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    clarification_questions: List[str] = Field(default_factory=list)
    input_type: InputType
    preference_updates: Optional[PreferenceUpdate] = None
    guardrails_result: Dict = Field(default_factory=dict)
    analysis_results: Optional[Dict] = Field(default_factory=dict)

class GraphState(TypedDict):
    """State maintained between nodes in the graph"""
    user_input: str
    messages: List[Dict[str, Any]]
    next_step: str
    rewritten_input: Optional[str] = None
    enhanced_input: Optional[str] = None
    validation_result: Optional[Dict[str, Any]] = None
    analysis_results: Optional[Dict] = None
    analysis_type: Optional[str] = None
    retry_count: int = 0
    preference_updates: Optional[PreferenceUpdate] = None
    categories: Optional[List[Dict[str, Any]]] = None
    subcategories: Optional[List[Dict[str, Any]]] = None
    examples: Optional[List[Dict[str, Any]]] = None
    verification_results: Optional[Dict[str, Any]] = None

class InputValidator:
    """Enhanced input validator with preference handling"""
    
    def __init__(self, scope: ValidationScope):
        self.scope = scope
        self.rails = LLMRails(RailsConfig.from_path("path/to/nemo/config"))
        self.llm_prompt = self._build_llm_prompt()
        self.preference_prompt = self._build_preference_prompt()
        self.analysis_prompt = self._build_analysis_prompt()
        
    def _build_llm_prompt(self) -> str:
        """Builds prompt for validating user input and identifying type"""
        return """You are a specialized input validator for an AI guidance system. 
        Your task is to analyze user input that should contain their background and goals and identify its type.
        
        Analyze the following aspects:
        1. Type of input: Based on the structure and keywords, identify the type of input (new_query, rewrite_request, clarification_response, preference_update, preference_removal, preference_query, input_enhancement).
        2. Presence of background information (education, experience, skills)
        3. Presence of goals (short-term, long-term, specific objectives)
        4. Completeness of background information: Assign a score (0-1) to indicate how comprehensive their background description is.
        5. Clarity of goals: Assign a score (0-1) to indicate the clarity and specificity of their goals.
        6. Identify any missing critical information: Specify any important elements missing from the user's input
        7. Generate clarifying questions if needed: Create specific, open-ended questions that can help gather missing information or resolve ambiguities.

        The input text is: {input_text}
        
        Provide your analysis in the following JSON format:
        {{
            "is_valid": boolean,
            "has_background": boolean,
            "has_goals": boolean,
            "background_completeness": float (0-1),
            "goals_clarity": float (0-1),
            "missing_elements": [list of missing important elements],
            "suggestions": [list of suggestions for improvement],
            "clarification_questions": [list of specific questions to ask],
            "input_type": "new_query" | "rewrite_request" | "clarification_response" | "preference_update" | "preference_removal" | "preference_query" | "input_enhancement" | "invalid_input"
        }}
        """
        
    def _build_preference_prompt(self) -> str:
        """Builds prompt for preference analysis"""
        return """Analyze the user input for preference-related information.
        
        Detect:
        1. Type of preference operation (add/remove/update/query)
        2. Specific preferences being mentioned
        3. Categories they belong to
        4. Priority or importance (if mentioned)
        5. Any constraints or conditions
        
        The input text is: {input_text}
        
        Provide analysis in JSON format:
        {{
            "detected_operation": "add|remove|update|query",
            "detected_preferences": [
                {{
                    "category": "category_name",
                    "value": "preference_value",
                    "priority": optional_priority_number,
                    "constraints": {{}}
                }}
            ],
            "confidence_score": float (0-1),
            "needs_clarification": boolean,
            "clarification_questions": []
        }}
        """
    def _build_analysis_prompt(self) -> str:
        """Builds prompt for comprehensive input analysis"""
        return """Analyze the user's input comprehensively to develop internal
        understanding:
        1. User Summary (Internal)
            Understand:
                - Professional/personal background
                - Level of expertise/experience
                - Current situation/challenges
                - Vision and aspirations
                - Underlying motivations
        2. Keywords Analysis (Internal)
            Identify:
                - Primary domain keywords
                - Secondary interest areas
                - Technical/professional terms
                - Skill indicators
                - Contextual markers
                - Geographic/cultural references
        3. Needs Assessment (Internal)
            Recognize:
                - Explicit stated needs
                - Implicit requirements
                - Short-term goals
                - Long-term aspirations
                - Resource requirements
                - Knowledge gaps
                - Network needs
        4. Segment Understanding (Internal)
           Determine:
                - Primary professional category
                - Experience level
                - Industry context
                - Role characteristics
                - Growth stage
                - Impact potential
        5. Contextual Synthesis (Internal)
            Map:
                 - Opportunity alignment
                 - Potential pathways
                 - Resource matches
                 - Connection possibilities
                 - Growth trajectories
         Provide your analysis in the following JSON format:
         {{
                "user_summary": {{
                    "background": "user background information here",
                    "expertise_level": "user expertise level here",
                    "goals": [],
                    "key_interests": []
                }},
                "keywords": {{
                    "domain_terms": [],
                    "technical_terms": [],
                    "relationships": []
                }},
                "needs": {{
                    "explicit": [],
                    "implicit": [],
                    "gaps": []
                }},
                 "segment": {{
                    "professional_category": "",
                    "experience_level": "",
                    "industry_context": "",
                    "role_characteristics": "",
                     "growth_stage":"",
                     "impact_potential":""
                  }}
            }}
        """

    async def analyze_preferences(self, text: str) -> PreferenceAnalysisResult:
        """Analyzes input for preference-related content"""
        # Call LLM with preference analysis prompt
        llm_response = await run_llm_call(
            self.preference_prompt.format(input_text=text)
        )
        
        # Parse and validate response
        try:
            analysis = PreferenceAnalysisResult(**json.loads(llm_response))
            
            # Validate against scope
            if not self._validate_preference_scope(analysis.detected_preferences):
                analysis.needs_clarification = True
                analysis.clarification_questions.append(
                    "Some preferences exceed allowed limits. Please revise."
                )
            
            return analysis
            
        except Exception as e:
            logging.error(f"Preference analysis error: {str(e)}")
            return PreferenceAnalysisResult(
                detected_operation=PreferenceOperation.QUERY,
                detected_preferences=[],
                confidence_score=0.0,
                needs_clarification=True,
                clarification_questions=[
                    "Could you please clarify your preference-related request?"
                ]
            )

    def _validate_preference_scope(self, preferences: List[PreferenceValue]) -> bool:
        """Validates preferences against defined scope"""
        # Check total preferences
        if len(preferences) > self.scope.max_total_preferences:
            return False
        
        # Count preferences per category
        category_counts = {}
        for pref in preferences:
            category_counts[pref.category] = category_counts.get(pref.category, 0) + 1
            if category_counts[pref.category] > self.scope.max_preferences_per_category:
                return False
        
        return True

    async def validate(self, text: str) -> ValidationResult:
        """Enhanced validation with preference handling"""
        try:
            # First check if this is a preference-related input
            preference_analysis = await self.analyze_preferences(text)
            
            if preference_analysis.confidence_score > 0.7:
                # This is a preference-related input
                if preference_analysis.needs_clarification:
                    return self._create_preference_clarification_result(preference_analysis)
                
                return ValidationResult(
                    is_valid=True,
                    has_background=False,
                    has_goals=False,
                    background_completeness=0.0,
                    goals_clarity=0.0,
                    input_type=InputType.PREFERENCE_UPDATE,
                    preference_updates=PreferenceUpdate(
                        operation=preference_analysis.detected_operation,
                        preferences=preference_analysis.detected_preferences
                    ),
                     guardrails_result = {}
                )
            
            # If not preference-related, proceed with regular validation
            # Check through NeMo Guardrails
            guardrails_result = await self.rails.generate(
                messages=[{"role": "user", "content": text}]
            )
            
            if guardrails_result.get("blocked", False):
                return self._create_blocked_result(guardrails_result)
            
            # Continue with regular validation...
            llm_response = await run_llm_call(
                    self.llm_prompt.format(input_text=text)
                )
            
            analysis = ValidationResult(**json.loads(llm_response))
            analysis.guardrails_result = guardrails_result
            
            if analysis.input_type in [InputType.REWRITE_REQUEST, InputType.INPUT_ENHANCEMENT]:
                     # If the input type is rewrite or enhance, call the corresponding methods to handle the logic
                     return await self._handle_rewrite_enhance(text, analysis)
            
            if analysis.input_type == InputType.NEW_QUERY:
              # if it is a new query we should run the comprehensive analysis
               analysis_results = await self._run_comprehensive_analysis(text)
               analysis.analysis_results = analysis_results
            return analysis
            
        except Exception as e:
            logging.error(f"Validation error: {str(e)}")
            return self._create_error_result(str(e))

    def _create_preference_clarification_result(
        self, 
        analysis: PreferenceAnalysisResult
    ) -> ValidationResult:
        """Creates result for preference clarification needed"""
        return ValidationResult(
            is_valid=False,
            has_background=False,
            has_goals=False,
            background_completeness=0.0,
            goals_clarity=0.0,
            input_type=InputType.PREFERENCE_UPDATE,
            clarification_questions=analysis.clarification_questions,
           guardrails_result = {}
        )
    
    def _create_blocked_result(self, guardrails_result: Dict) -> ValidationResult:
        return ValidationResult(
            is_valid=False,
            has_background=False,
            has_goals=False,
            background_completeness=0.0,
            goals_clarity=0.0,
            input_type = InputType.INVALID_INPUT,
            missing_elements=['Blocked by guardrails'],
            suggestions=[
                "Please rephrase your query",
               "Please avoid sensitive or prohibited topics"
            ],
            clarification_questions=[],
            guardrails_result=guardrails_result
          )
    
    def _create_error_result(self, error: str) -> ValidationResult:
         return ValidationResult(
            is_valid=False,
            has_background=False,
            has_goals=False,
            background_completeness=0.0,
            goals_clarity=0.0,
            input_type = InputType.INVALID_INPUT,
            missing_elements=['Validation failed'],
            suggestions=['Please try again'],
            clarification_questions=[
                "Could you please rephrase your background and goals?"
            ],
            guardrails_result={}
        )
    
    async def _handle_rewrite_enhance(self, text: str, analysis: ValidationResult) -> ValidationResult:
              """Handle rewrite and enhance requests"""

              rewrite_enhance_prompt = ""

              if analysis.input_type == InputType.REWRITE_REQUEST:
                rewrite_enhance_prompt = f"""Rewrite the user's input to be:
                    1. More concise
                    2. Better structured
                    3. Clearly articulated
                    While maintaining:
                        - Original intent
                        - Key information
                        - Personal tone
                    Format:
                    [Original Input]
                    {text}
                    [Rewritten Input]
                    """

              elif analysis.input_type == InputType.INPUT_ENHANCEMENT:
                   rewrite_enhance_prompt = f"""Enhance the user's input by:
                    1. Expanding key points
                    2. Adding relevant context
                    3. Clarifying goals
                    4. Incorporating industry-standard terminology
                    Format:
                    [Original/Rewritten Input]
                    {text}
                    [Enhanced Input with:
                        - Deeper context
                        - Specific goals
                        - Relevant terminology
                        - Action-oriented language]
                    """
              llm_response = await run_llm_call(rewrite_enhance_prompt)
              
              try:
                 return ValidationResult(
                    is_valid=True,
                    has_background=False,
                    has_goals=False,
                    background_completeness=0.0,
                    goals_clarity=0.0,
                    input_type=analysis.input_type,
                    guardrails_result = {},
                    suggestions=[llm_response['rewritten_input'] if analysis.input_type == InputType.REWRITE_REQUEST else llm_response['enhanced_input']]
                )
                
              except Exception as e:
                  return self._create_error_result(f"Error during rewrite/enhance: {e}")
    async def _run_comprehensive_analysis(self, text: str) -> Dict:
        """
        Function to run comprehensive analysis using LLM
        """
        # Call LLM with comprehensive analysis prompt
        analysis_prompt = self._build_analysis_prompt()
        llm_response = await run_llm_analysis(
            analysis_prompt.format(input=text)
         )
        
        return llm_response

class ValidationNode:
    """Enhanced validation node with preference handling"""
    
    def __init__(self, validator: InputValidator):
        self.validator = validator
    
    async def __call__(self, state: GraphState) -> GraphState:
        """Process the validation node"""
        try:
            # Perform validation
            result = await self.validator.validate(state['user_input'].raw_input)
            
            # Update state
            state['validation_result'] = result
            
            # Determine next step based on result
            if result.input_type in [
                InputType.PREFERENCE_UPDATE,
                InputType.PREFERENCE_REMOVAL,
                InputType.PREFERENCE_QUERY
            ]:
                if result.is_valid:
                    state['next_step'] = 'preference_processor'
                else:
                    state['next_step'] = 'preference_clarification'
            elif not result.is_valid:
                if result.guardrails_result.get("blocked", False):
                    state['next_step'] = 'blocked_content_handler'
                else:
                    state['next_step'] = 'clarification_node'
            elif result.input_type == InputType.NEW_QUERY:
                state['next_step'] = 'analysis_node'
            else:
                state['next_step'] = 'input_processed'
            
            return state
            
        except Exception as e:
            logging.error(f"ValidationNode error: {str(e)}")
            state['next_step'] = 'error_handler'
            return state
        
def create_validation_workflow() -> Callable:
    """Creates the complete validation workflow"""
    scope = ValidationScope()
    validator = InputValidator(scope)
    validation_node = ValidationNode(validator)
    
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("validate_input", validation_node)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "validate_input",
        lambda state: state['next_step'],
        {
            "analysis_node": "analysis_node",
            "clarification_node": "clarification_node",
             "preference_processor": "preference_processor",
            "preference_clarification": "preference_clarification",
            "blocked_content_handler": "blocked_content_handler",
             "input_processed": "input_processed",
            "error_handler": "error_handler"
        }
    )
    
    return workflow.compile()

from app.models.user_input import UserInput

def generate_clarifying_questions(validation_result: ValidationResult, user_input: UserInput) -> List[str]:
    """
    Function to generate clarifying questions using LLM
    In practice, replace this with actual LLM call
    """
    # TODO: Implement actual LLM call
    # This is a mock implementation
    return validation_result.clarification_questions