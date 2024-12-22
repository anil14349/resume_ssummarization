"""
Base parser class for resume parsing.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseResumeParser(ABC):
    """Base class for resume parsers."""
    
    def __init__(self, file_path: str):
        """Initialize parser with file path.
        
        Args:
            file_path (str): Path to the resume file
        """
        self.file_path = file_path

    @abstractmethod
    def parse_docx_to_json(self) -> Dict[str, Any]:
        """Parse resume file into JSON format.
        
        Returns:
            Dict[str, Any]: Parsed resume data
            
        Raises:
            ValueError: If file cannot be parsed
        """
        pass

    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data has required fields.
        
        Args:
            data (Dict[str, Any]): Data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not data:
            return False
            
        required_fields = ['name', 'sections']
        return all(field in data for field in required_fields)

    def clean_text(self, text: str) -> str:
        """Clean and normalize text.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        return ' '.join(text.strip().split())
