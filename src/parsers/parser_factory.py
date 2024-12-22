"""Factory for creating resume parsers."""
import logging
import os
from pathlib import Path
from typing import Union

from .ats_parser import ATSParser
from .base_parser import BaseParser
from .industry_manager_parser import IndustryManagerParser

logger = logging.getLogger(__name__)

class ParserFactory:
    """Factory class for creating resume parsers."""
    
    @staticmethod
    def create_parser(parser_type: str, file_path: str) -> BaseParser:
        """Create a parser instance based on type."""
        logger.info(f"Creating parser of type '{parser_type}' for file: {file_path}")
        
        try:
            # Validate file path
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Create parser based on type
            if parser_type.lower() == 'ats':
                return ATSParser()
            elif parser_type.lower() == 'industry':
                return IndustryManagerParser()
            else:
                raise ValueError(f"Unknown parser type: {parser_type}")
                
        except Exception as e:
            logger.error(f"Error creating parser: {e}")
            raise
