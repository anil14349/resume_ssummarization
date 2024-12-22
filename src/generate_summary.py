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

from models.enhanced_model_factory import EnhancedModelFactory
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
        "--enhancer",
        choices=["few_shot", "cot", "rag"],
        help="Optional enhancer to use"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    return args

def main():
    """Main function to generate summaries."""
    try:
        # Parse arguments
        args = parse_arguments()
        logger.info(f"Starting summary generation with args: {args}")
        
        # Set logging level
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
        
        # Validate input file
        input_path = Path(args.input_file)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        logger.info(f"Processing resume file: {input_path}")
        
        # Create parser
        try:
            parser = ParserFactory.create_parser(args.parser, input_path)
            logger.debug(f"Created parser: {parser.__class__.__name__}")
        except Exception as e:
            logger.error(f"Failed to create parser: {e}")
            raise
            
        # Parse resume
        try:
            resume_data = parser.parse()
            logger.debug(f"Parsed resume data: {resume_data}")
        except Exception as e:
            logger.error(f"Failed to parse resume: {e}")
            raise
            
        # Create model
        try:
            model = EnhancedModelFactory.create_model(
                model_type=args.model,
                enhancer_type=args.enhancer
            )
            logger.debug(f"Created model: {model.__class__.__name__}")
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise
            
        # Generate summary
        try:
            summary = model.generate_summary(resume_data)
            logger.info("Successfully generated summary")
            logger.debug(f"Generated summary: {summary}")
            
            print("\nGenerated Summary:")
            print("-----------------")
            print(summary)
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            raise
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
