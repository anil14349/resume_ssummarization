from models.model_factory import ResumeModelFactory
from extractor.docx_extractor import DocumentParser
import json
import os

def main():
    # Get the directory where the script is located
    resume_path = 'data/Industry manager resume 1.docx'

    # Initialize the document parser
    doc_parser = DocumentParser()
    input_json = doc_parser.extract_docx_text(resume_path)

    # Try each model type
    model_configs = [
        ("t5", "base"),
        ("gpt2", "medium"),
        ("bart", "large")
    ]

    for model_type, model_size in model_configs:
        print(f"\nTrying {model_type.upper()}-{model_size} model:")
        print("-" * 50)
        
        try:
            # Create model
            model = ResumeModelFactory.create_model(model_type, model_size)
            
            # Generate summary
            print("Generating summary (this may take a moment)...")
            summary = model.generate_summary(input_json)
            
            print("\nGenerated Professional Summary:")
            print(summary)
            
        except Exception as e:
            print(f"Error with {model_type}-{model_size} model: {str(e)}")

if __name__ == "__main__":
    main()
