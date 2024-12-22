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

def main():
    """Main function."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        logger.info(f"Starting summary generation with args: {args}")
        
        # Validate input file
        if not os.path.exists(args.input_file):
            logger.error(f"Input file not found: {args.input_file}")
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
            
        # Get absolute path
        input_file = os.path.abspath(args.input_file)
        logger.info(f"Processing resume file: {input_file}")
        
        # Create parser
        parser = ParserFactory.create_parser(args.parser, input_file)
        logger.info(f"Created parser of type '{args.parser}' for file: {input_file}")
        
        # Parse resume
        resume_data = parser.parse(input_file)
        if not resume_data:
            logger.error("Failed to parse resume data")
            raise ValueError("Failed to parse resume data")
        
        # Create model
        try:
            model = create_model(args.model)
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            sys.exit(1)
        
        # Generate summary
        try:
            summary = model.generate_summary(resume_data)
            print("\nGenerated Summary:\n-----------------")
            print(summary)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
