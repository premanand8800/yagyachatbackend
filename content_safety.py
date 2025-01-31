"""Content safety checker implementation"""
from typing import Dict, List, Tuple
import re
import logging

class ContentSafetyChecker:
    """Simple pattern-based content safety checker"""
    
    def __init__(self):
        # Patterns for unsafe content
        self.unsafe_patterns = {
            'adult_content': [
                r'\b(porn|xxx|adult|nsfw)\b',
                r'\b(sex|nude|naked)\b'
            ],
            'harmful_commands': [
                r'\b(rm\s+-rf|sudo|del|format)\b',
                r'\b(DROP\s+TABLE|DELETE\s+FROM)\b'
            ],
            'security_risks': [
                r'\b(hack|crack|exploit)\b',
                r'\b(password|credentials)\b'
            ],
            'hate_speech': [
                r'\b(hate|racist|discrimination)\b'
            ]
        }
        
        # Initialize pattern cache
        self.compiled_patterns = {
            category: [re.compile(pattern, re.IGNORECASE) 
                      for pattern in patterns]
            for category, patterns in self.unsafe_patterns.items()
        }
    
    def check_content(self, text: str) -> Tuple[bool, Dict]:
        """
        Check if content is safe
        Returns: (is_safe, details)
        """
        if not text or not isinstance(text, str):
            return True, {"message": "Empty or invalid input"}
            
        violations = []
        categories_found = set()
        
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    violations.extend(matches)
                    categories_found.add(category)
        
        is_safe = len(violations) == 0
        
        details = {
            "is_safe": is_safe,
            "blocked": not is_safe,
            "categories": list(categories_found),
            "violations_found": len(violations),
            "message": "Content is safe" if is_safe else "Unsafe content detected",
            "suggestions": self._get_suggestions(categories_found) if not is_safe else []
        }
        
        if not is_safe:
            logging.warning(f"Unsafe content detected: {details}")
        
        return is_safe, details
    
    def _get_suggestions(self, categories: set) -> List[str]:
        """Get appropriate suggestions based on violated categories"""
        suggestions = []
        
        if 'adult_content' in categories:
            suggestions.append("Please avoid adult or inappropriate content")
            
        if 'harmful_commands' in categories:
            suggestions.append("System commands are not allowed in this context")
            
        if 'security_risks' in categories:
            suggestions.append("Security-related terms are not appropriate here")
            
        if 'hate_speech' in categories:
            suggestions.append("Please maintain respectful and appropriate language")
            
        suggestions.append("Try rephrasing your request in a more appropriate way")
        return suggestions
