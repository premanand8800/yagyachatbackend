# app/utils/exceptions.py
from typing import Optional

class BaseError(Exception):
    """Base exception class for custom errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class AnalysisError(BaseError):
    """Exception raised for errors during analysis processing."""
    pass

class ConfigurationError(BaseError):
    """Exception raised for configuration-related errors."""
    pass

class ValidationError(BaseError):
    """Exception raised for validation errors."""
    pass

class LLMError(BaseError):
    """Exception raised for LLM-related errors."""
    pass