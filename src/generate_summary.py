#!/usr/bin/env python3
"""Generate summaries from resumes using various models."""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from models.model_factory import create_model
from parsers.parser_factory import ParserFactory

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('resume_summary.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate resume summary")
    parser.add_argument("input_file", help="Path to input resume file")
    parser.add_argument(
        "--model", 
        choices=["gpt2", "t5", "bart"], 
        default="gpt2",
        help="Model to use for generation"
    )
    parser.add_argument(
        "--parser",
        choices=["ats", "industry"],
        default="ats",
        help="Parser to use for resume"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    return args

def main(input_file: str, model_type: str = "gpt2", parser_type: str = "ats", debug: bool = False) -> str:
    """Main function for generating resume summaries.
    
    Args:
        input_file: Path to input resume file
        model_type: Type of model to use (gpt2, t5, or bart)
        parser_type: Type of parser to use (ats or industry)
        debug: Enable debug logging
        
    Returns:
        str: Generated summary
        
    Raises:
        FileNotFoundError: If input file not found
        ValueError: If resume parsing fails
    """
    try:
        logger.info(f"Starting summary generation with: model={model_type}, parser={parser_type}")
        
        # Validate input file
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        # Get absolute path
        input_file = os.path.abspath(input_file)
        logger.info(f"Processing resume file: {input_file}")
        
        # Create parser
        parser = ParserFactory.create_parser(parser_type, input_file)
        logger.info(f"Created parser of type '{parser_type}' for file: {input_file}")
        
        # Parse resume
        resume_data = parser.parse(input_file)
        if not resume_data:
            logger.error("Failed to parse resume data")
            raise ValueError("Failed to parse resume data")
        
        # Create model
        try:
            model = create_model(model_type)
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
        
        # Generate summary
        try:
            summary = model.generate_summary(resume_data)
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    # Parse command line arguments when run as script
    args = parse_arguments()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Call main function with parsed arguments
        summary = main(
            input_file=args.input_file,
            model_type=args.model,
            parser_type=args.parser,
            debug=args.debug
        )
        
        # Print summary
        print("\nGenerated Summary:\n-----------------")
        print(summary)
        
    except Exception as e:
        sys.exit(1)
