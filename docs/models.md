# Models Documentation

## GPT2ResumeModel

The `GPT2ResumeModel` class implements the GPT-2 based resume summarization model.

### Class Definition
```python
class GPT2ResumeModel(BaseResumeModel):
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
def generate_summary(self, text: str) -> str:
    """Generate a resume summary from input text.
    
    Args:
        text (str): Input resume text
        
    Returns:
        str: Generated summary
        
    Raises:
        ModelError: If generation fails
    """
```

#### preprocess_text
```python
def preprocess_text(self, text: str) -> str:
    """Clean and format input text.
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Preprocessed text
    """
```

## EnhancedModelFactory

The `EnhancedModelFactory` class manages the creation and enhancement of language models.

### Class Definition
```python
class EnhancedModelFactory:
    def __init__(
        self,
        model_type: str,
        cleanup_enhancer: Optional[MLCleanupEnhancer] = None,
        config: Optional[Dict] = None
    ):
        """Initialize model factory.
        
        Args:
            model_type (str): Type of model to create ('gpt2', 't5', 'bart')
            cleanup_enhancer (Optional[MLCleanupEnhancer]): ML cleanup system
            config (Optional[Dict]): Model configuration
        """
```

### Key Methods

#### create_model
```python
def create_model(self) -> BaseResumeModel:
    """Create and return a configured model instance.
    
    Returns:
        BaseResumeModel: Configured model instance
        
    Raises:
        ValueError: If model_type is invalid
    """
```

#### add_enhancement
```python
def add_enhancement(self, enhancement_type: str):
    """Add specified enhancement to model.
    
    Args:
        enhancement_type (str): Type of enhancement to add
        
    Raises:
        ValueError: If enhancement_type is invalid
    """
```
