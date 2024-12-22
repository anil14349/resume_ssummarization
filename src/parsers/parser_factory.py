"""Factory for creating resume parsers."""
import logging
from pathlib import Path
from typing import Union

from .ats_parser import ATSParser
from .industry_manager_parser import IndustryManagerParser

logger = logging.getLogger(__name__)

class ParserFactory:
    """Factory class for creating resume parsers."""
    
    @staticmethod
    def create_parser(parser_type: str, file_path: Union[str, Path]) -> Union[ATSParser, IndustryManagerParser]:
        """Create and return a parser instance based on parser type."""
        logger.info(f"Creating parser of type '{parser_type}' for file: {file_path}")
        
        try:
            # Convert to Path object if string
            if isinstance(file_path, str):
                file_path = Path(file_path)
                
            # Validate file exists
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Create appropriate parser
            if parser_type.lower() == "ats":
                logger.debug("Creating ATSParser")
                return ATSParser(file_path)
            elif parser_type.lower() == "industry":
                logger.debug("Creating IndustryManagerParser")
                return IndustryManagerParser(file_path)
            else:
                logger.error(f"Unknown parser type: {parser_type}")
                raise ValueError(f"Unknown parser type: {parser_type}")
                
        except Exception as e:
            logger.error(f"Error creating parser: {e}")
            raise
