import docx
import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ResumeParser:
    """Parser for resume files in various formats."""
    
    def __init__(self):
        """Initialize the parser."""
        pass
        
    def parse(self, file_path: str) -> Dict[str, Any]:
        """Parse a resume file and extract relevant information."""
        try:
            # Try to parse as Word document first
            try:
                return self._parse_docx(file_path)
            except Exception as e:
                logger.info(f"Not a Word document, trying text file: {e}")
                return self._parse_text(file_path)
                
        except Exception as e:
            logger.error(f"Error parsing resume: {e}")
            raise
            
    def _parse_docx(self, file_path: str) -> Dict[str, Any]:
        """Parse a Word document resume."""
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return self._parse_content(text)
        
    def _parse_text(self, file_path: str) -> Dict[str, Any]:
        """Parse a text file resume."""
        with open(file_path, 'r') as f:
            text = f.read()
        return self._parse_content(text)
        
    def _parse_content(self, text: str) -> Dict[str, Any]:
        """Parse the text content of a resume."""
        # Initialize result dictionary
        result = {
            'name': '',
            'current_role': '',
            'years_experience': 0,
            'companies': [],
            'skills': [],
            'achievements': [],
            'contact_info': {'email': '', 'phone': ''}
        }
        
        # Split into sections
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            if line.lower().endswith(':'):
                current_section = line[:-1].lower()
                continue
                
            # Parse based on current section
            if current_section == 'name':
                result['name'] = line
            elif current_section == 'current role':
                result['current_role'] = line
            elif current_section == 'years of experience':
                try:
                    result['years_experience'] = float(line)
                except:
                    pass
            elif current_section == 'companies':
                if line.startswith('-'):
                    result['companies'].append(line[1:].strip())
            elif current_section == 'skills':
                if line.startswith('-'):
                    result['skills'].append(line[1:].strip())
            elif current_section == 'achievements':
                if line.startswith('-'):
                    result['achievements'].append(line[1:].strip())
            elif current_section == 'contact':
                if 'email:' in line.lower():
                    result['contact_info']['email'] = line.split(':')[1].strip()
                elif 'phone:' in line.lower():
                    result['contact_info']['phone'] = line.split(':')[1].strip()
                    
        return result
