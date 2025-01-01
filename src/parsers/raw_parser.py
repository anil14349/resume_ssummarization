import logging
from typing import Dict, Any, Optional
from .base_parser import BaseParser

logger = logging.getLogger(__name__)

class RawParser(BaseParser):
    """Parser that reads raw text from files."""
    
    def __init__(self, file_path: str):
        """Initialize the parser with a file path.
        
        Args:
            file_path: Path to the resume file
        """
        super().__init__(file_path)
    
    def parse(self) -> str:
        """Parse the resume file and return its raw content.
        
        Returns:
            Raw text content of the resume file
        """
        try:
            # Read file content based on extension
            ext = self.file_path.lower().split('.')[-1]
            
            if ext == 'txt':
                # Plain text file
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
            elif ext == 'docx':
                # Word document
                from docx import Document
                doc = Document(self.file_path)
                content = '\n'.join(paragraph.text for paragraph in doc.paragraphs if paragraph.text)
                
            elif ext == 'pdf':
                # PDF document
                import PyPDF2
                with open(self.file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    content = '\n'.join(page.extract_text() for page in pdf_reader.pages)
                    
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            
            # Basic cleaning
            content = content.strip()
            content = '\n'.join(line.strip() for line in content.split('\n') if line.strip())
            
            return content
            
        except Exception as e:
            logger.error(f"Error parsing file {self.file_path}: {str(e)}")
            raise
