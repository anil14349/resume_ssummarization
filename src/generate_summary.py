from models.model_factory import ResumeModelFactory

def main():
    # Example input data
    input_data = {
        'name': 'May Riley',
        'current_role': 'Restaurant Manager',
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
        ]
    }

    # Initialize model factory
    factory = ResumeModelFactory()

    # Try each model
    model_configs = [
        ("t5", "base"),
        ("gpt2", "medium"),
        ("bart", "large")
    ]
    
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
