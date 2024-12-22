"""
Base parser class for resume parsing.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseParser(ABC):
    """Abstract base class for resume parsers."""
    
    @abstractmethod
    def parse(self, file_path: str) -> Dict[str, Any]:
        """Parse a resume file and extract relevant information.
        
        Args:
            file_path: Path to the resume file to parse
            
        Returns:
            Dictionary containing parsed resume data with the following keys:
            - name: Candidate's name
            - current_role: Current or most recent role
            - companies: List of companies worked at
            - years_experience: Total years of experience
            - skills: List of skills
            - achievements: List of key achievements
        """
        pass
