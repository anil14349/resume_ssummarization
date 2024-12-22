"""
Base parser class for resume parsing.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseResumeParser(ABC):
    """Base class for resume parsers."""
    
    def __init__(self, file_path=None):
        self.file_path = file_path

    @abstractmethod
    def parse(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse resume data according to specific format requirements."""
        pass

    @abstractmethod
    def format_output(self, parsed_data: Dict[str, Any]) -> str:
        """Format parsed data into a string representation."""
        pass

    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data has required fields."""
        if not data:
            return False
            
        required_fields = ['name', 'sections']
        return all(field in data for field in required_fields)
