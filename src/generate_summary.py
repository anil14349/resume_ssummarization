#!/usr/bin/env python3
"""
Script to generate resume summaries using different models.
"""

import argparse
import os
import sys
from pathlib import Path
from docx import Document

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from parsers.ats_parser import ATSParser
from parsers.industry_manager_parser import IndustryManagerParser
from models.model_factory import ResumeModelFactory

def main():
    parser = argparse.ArgumentParser(description='Generate resume summary from docx file')
    parser.add_argument('resume_file', help='Path to the resume docx file')
    parser.add_argument('--model', 
                      choices=['t5-base', 't5-large', 'gpt2', 'gpt2-medium', 'bart-base', 'bart-large'],
                      default='t5-base', 
                      help='Model to use for generation')
    parser.add_argument('--parser', choices=['ats', 'industry'],
                      default='ats', help='Parser to use (ats or industry)')
    args = parser.parse_args()

    # Ensure resume file exists
    resume_path = Path(args.resume_file)
    if not resume_path.exists():
        print(f"Error: Resume file not found: {resume_path}")
        return 1

    try:
        # Parse resume based on template
        if args.parser == 'ats':
            parser = ATSParser(str(resume_path))
        else:
            parser = IndustryManagerParser(str(resume_path))

        # Parse the resume
        print("Parsing resume...")
        input_data = parser.parse_docx_to_json()
        
        # Map model name to type and size
        model_type, model_size = args.model.split('-', 1) if '-' in args.model else (args.model, 'base')
        
        # Initialize model with caching
        print(f"Initializing {args.model} model...")
        factory = ResumeModelFactory()
        model = factory.create_model(model_type, model_size)
        
        # Generate summary
        print("Generating summary...")
        summary = model.generate_summary(input_data)
        
        print("\nGenerated Summary:")
        print("-----------------")
        print(summary)
        print("-----------------")
        
        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == '__main__':
    exit(main())
