"""Factory for creating parser instances."""
import os
import logging
from typing import Any
from src.parsers.base_parser import BaseParser
from src.parsers.ats_parser import ATSParser
from src.parsers.industry_manager_parser import IndustryManagerParser

logger = logging.getLogger(__name__)

class ParserFactory:
    """Factory class for creating parser instances."""

    @staticmethod
    def create_parser(parser_type: str, file_path: str) -> BaseParser:
        """Create a parser instance based on the parser type."""
        logger.info(f"Creating parser of type '{parser_type}' for file: {file_path}")

        try:
            # Validate input parameters
            if not file_path:
                raise ValueError("Input file is required")

            # Validate file path
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Create parser based on type
            if parser_type.lower() == 'ats':
                from .ats_parser import ATSParser
                return ATSParser(file_path)
            elif parser_type.lower() == 'industry':
                from .industry_manager_parser import IndustryManagerParser
                return IndustryManagerParser(file_path)
            elif parser_type.lower() == 'raw':
                from .raw_parser import RawParser
                return RawParser(file_path)
            else:
                raise ValueError(f"Unknown parser type: {parser_type}")

        except Exception as e:
            logger.error(f"Error creating parser: {str(e)}")
            raise
