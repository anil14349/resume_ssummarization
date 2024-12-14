from models.model_factory import ResumeModelFactory
from datetime import datetime
import argparse

def get_available_models():
    """Return a list of available models and their descriptions."""
    return {
        "t5-base": "T5 model - Good at following structured prompts",
        "gpt2-medium": "GPT-2 model - Natural language generation",
        "bart-large": "BART model - Good at both understanding and generation"
    }

def select_model():
    """Interactive model selection."""
    available_models = get_available_models()
    
    print("\nAvailable models:")
    print("-" * 50)
    for i, (model_name, description) in enumerate(available_models.items(), 1):
        print(f"{i}. {model_name}: {description}")
    
    while True:
        try:
            choice = input("\nSelect a model (1-3) or press Enter to try all models: ").strip()
            if not choice:  # Empty input - try all models
                return None
            
            choice = int(choice)
            if 1 <= choice <= len(available_models):
                selected_model = list(available_models.keys())[choice - 1]
                model_type, model_size = selected_model.split('-')
                return [(model_type, model_size)]
            else:
                print(f"Please enter a number between 1 and {len(available_models)}")
        except ValueError:
            print("Please enter a valid number")

def main():
    parser = argparse.ArgumentParser(description='Generate professional summaries using different models.')
    parser.add_argument('--model', choices=['t5-base', 'gpt2-medium', 'bart-large'], 
                       help='Specific model to use (optional)')
    args = parser.parse_args()

    # Example input data
    input_data = {
        'name': 'May Riley',
        'current_role': 'Restaurant Manager',
        'years_experience': 5,
        'companies': [
            'Contoso Bar and Grill',
            'Fourth Coffee Bistro'
        ],
        'achievements': [
            'reduced costs by 7% via controls on overtime, operational efficiencies, and reduced waste',
            'consistently exceed monthly sales goals by a least 10% by training FOH staff on upselling techniques via a featured food and beverage program'
        ],
        'skills': [
            'Accounting & Budgeting',
            'Proficient with POS systems',
            'Excellent interpersonal and communication skills'
        ],
        'education': [
            'B.S. in Business Administration',
            'A.A. in Hospitality Management'
        ],
        'recognition': 'one of the top restaurant managers in our area'
    }

    # Initialize model factory
    factory = ResumeModelFactory()

    # Determine which models to use
    if args.model:
        model_type, model_size = args.model.split('-')
        model_configs = [(model_type, model_size)]
    else:
        model_configs = select_model()
        if model_configs is None:  # Try all models
            model_configs = [
                ("t5", "base"),
                ("gpt2", "medium"),
                ("bart", "large")
            ]
    
    # Generate summaries
    for model_type, model_size in model_configs:
        print(f"\nTrying {model_type.upper()}-{model_size} model:")
        print("-" * 50)
        print("Generating summary (this may take a moment)...")
        
        try:
            model = factory.create_model(model_type, model_size)
            summary = model.generate_summary(input_data)
            print(f"\nGenerated Professional Summary:\n{summary}")
        except Exception as e:
            print(f"Error with {model_type}-{model_size} model: {e}")

if __name__ == "__main__":
    main()
