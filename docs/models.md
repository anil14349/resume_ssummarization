# Models Documentation

## GenericGPT2Model

The `GenericGPT2Model` class is responsible for generating video scripts from parsed resume data using GPT-2.

### Class Definition
```python
class GenericGPT2Model:
    def __init__(self, model_name="gpt2", device="cpu"):
        """Initialize GPT2 model with specified configuration.
        
        Args:
            model_name (str): Name of the pretrained model to use
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
```

### Key Methods

#### generate_summary
```python
def generate_summary(self, resume_data: Dict) -> str:
    """Generate a video script from parsed resume data.
    
    Args:
        resume_data (Dict): Parsed resume data containing sections like:
            - name (str): Full name
            - current_role (str): Current job title
            - skills (List[str]): List of skills
            - experience (List[Dict]): Work experience
            - education (List[Dict]): Educational background
            - achievements (List[str]): Notable achievements
        
    Returns:
        str: Generated video script
        
    Raises:
        ModelError: If generation fails
    """
```

#### preprocess_text
```python
def preprocess_text(self, text: str) -> str:
    """Clean and format input text for the model.
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Preprocessed text ready for model input
    """
```

#### postprocess_output
```python
def postprocess_output(self, output: str) -> str:
    """Clean and format model output into a well-structured video script.
    
    Args:
        output (str): Raw model output
        
    Returns:
        str: Formatted video script
    """
```

### Usage Example

```python
# Initialize model
model = GenericGPT2Model()

# Example resume data
resume_data = {
    'name': 'John Doe',
    'current_role': 'Senior Software Engineer',
    'skills': ['Python', 'Machine Learning', 'Leadership'],
    'achievements': ['Led team of 5 engineers', 'Reduced costs by 40%']
}

# Generate video script
script = model.generate_summary(resume_data)
```

## Integration

The model is integrated with:
1. FastAPI backend for HTTP API access
2. Streamlit UI for interactive usage
3. ATSParser and IndustryManagerParser for resume data extraction
