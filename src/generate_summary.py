from models.model_factory import ResumeModelFactory
from datetime import datetime
import sys
import os
import argparse
from rouge_score import rouge_scorer
from parsers.ats_parser import ATSParser as ATSParser
from parsers.industry_manager_parser import IndustryManagerParser as IndustryManagerParser

def get_available_models():
    """Return a list of available models and their descriptions."""
    return {
        "t5-base": {
            "description": "T5 model - Good at following structured prompts",
            "strengths": ["Follows templates well", "Good at structured tasks", "Fast generation"],
            "best_for": "Professional, template-based summaries"
        },
        "gpt2-medium": {
            "description": "GPT-2 model - Natural language generation",
            "strengths": ["Natural writing style", "Creative outputs", "Good context understanding"],
            "best_for": "Natural-sounding, creative summaries"
        },
        "bart-large": {
            "description": "BART model - Good at both understanding and generation",
            "strengths": ["Balanced performance", "Good comprehension", "Reliable outputs"],
            "best_for": "Well-balanced, comprehensive summaries"
        }
    }

def display_model_info(model_info):
    """Display detailed information about a model."""
    print(f"\nModel Details:")
    print("-" * 50)
    print(f"Description: {model_info['description']}")
    print("\nStrengths:")
    for strength in model_info['strengths']:
        print(f"  • {strength}")
    print(f"\nBest for: {model_info['best_for']}")

def select_model():
    """Interactive model selection with detailed information."""
    available_models = get_available_models()
    
    while True:
        print("\nAvailable Models:")
        print("-" * 50)
        for i, (model_name, info) in enumerate(available_models.items(), 1):
            print(f"{i}. {model_name}: {info['description']}")
        print("\nOptions:")
        print("  • Enter model number (1-3) to select a model")
        print("  • Enter 'i' followed by model number for more info (e.g., 'i1')")
        print("  • Enter 'a' to try all models")
        print("  • Enter 'q' to quit")
        
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == 'q':
            print("\nExiting program.")
            sys.exit(0)
        
        if choice == 'a':
            return None
        
        if choice.startswith('i'):
            try:
                model_num = int(choice[1:]) - 1
                if 0 <= model_num < len(available_models):
                    model_info = list(available_models.values())[model_num]
                    display_model_info(model_info)
                    input("\nPress Enter to continue...")
                else:
                    print(f"Please enter a number between 1 and {len(available_models)}")
            except ValueError:
                print("Invalid input. Please try again.")
            continue
        
        try:
            choice = int(choice)
            if 1 <= choice <= len(available_models):
                selected_model = list(available_models.keys())[choice - 1]
                model_type, model_size = selected_model.split('-')
                
                # Display selected model info
                print(f"\nSelected: {selected_model}")
                display_model_info(available_models[selected_model])
                
                confirm = input("\nProceed with this model? (y/n): ").strip().lower()
                if confirm == 'y':
                    return [(model_type, model_size)]
            else:
                print(f"Please enter a number between 1 and {len(available_models)}")
        except ValueError:
            print("Invalid input. Please try again.")

def parse_args():
    parser = argparse.ArgumentParser(description='Professional Summary Generator')
    parser.add_argument('input_file', help='Input resume file path')
    parser.add_argument('--model', '-m', default='t5-base', help='Model to use (e.g., t5-base)')
    parser.add_argument('--parser', '-p', default='ats', choices=['ats', 'industry_manager'], help='Parser type')
    parser.add_argument('--metadata', action='store_true', help='Return metadata')
    return parser.parse_args()

def generate_summary(model_type, model_size, file_path, parser_type, return_metadata):
    # Initialize model factory
    factory = ResumeModelFactory()
    
    # Load input data
    if parser_type == 'ats':
        input_data = ATSParser(file_path).parse_docx_to_json()
    elif parser_type == 'industry_manager':
        input_data = IndustryManagerParser(file_path).parse_docx_to_json()
    else:
        raise ValueError("Invalid parser type")
    
    # Generate summary
    model = factory.create_model(model_type, model_size)
    summary = model.generate_summary(input_data)
    
    return summary

def main():
    """Main function to run the summary generator."""
    args = parse_args()
    
    if args.model:
        model_type, model_size = args.model.split('-')
        model_configs = [(model_type, model_size)]
    else:
        model_configs = select_model()
        
    if not model_configs:
        print("No model selected. Exiting.")
        return
        
    try:
        # Validate input file
        if not os.path.exists(args.input_file):
            print("Please provide a valid input file path.")
            return
            
        # Process each model configuration
        for model_type, model_size in model_configs:
            print(f"\nGenerating summary using {model_type}-{model_size}...")
            
            # Generate summary
            summary = generate_summary(
                model_type=model_type,
                model_size=model_size,
                file_path=args.input_file,
                parser_type=args.parser,
                return_metadata=args.metadata
            )
            
            # Print the summary
            print("\nGenerated Summary:")
            print("=" * 50)
            print(summary)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user.")
        sys.exit(0)
