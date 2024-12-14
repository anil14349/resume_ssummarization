from .t5_model import T5ResumeModel
from .gpt2_model import GPT2ResumeModel
from .bart_model import BartResumeModel

class ResumeModelFactory:
    """Factory class for creating resume summary generation models."""
    
    @staticmethod
    def create_model(model_type="t5", model_size="base"):
        """Create and return a model instance based on type and size.
        
        Args:
            model_type (str): Type of model ('t5', 'gpt2', or 'bart')
            model_size (str): Size of the model ('base', 'medium', 'large', etc.)
        
        Returns:
            BaseResumeModel: An instance of the requested model
        """
        if model_type == "t5":
            model_name = f"t5-{model_size}"
            return T5ResumeModel(model_name)
        
        elif model_type == "gpt2":
            if model_size == "base":
                model_name = "gpt2"
            else:
                model_name = f"gpt2-{model_size}"
            return GPT2ResumeModel(model_name)
        
        elif model_type == "bart":
            if model_size == "base":
                model_name = "facebook/bart-base"
            else:
                model_name = f"facebook/bart-{model_size}"
            return BartResumeModel(model_name)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# Example usage:
"""
# Create a T5 model
t5_model = ResumeModelFactory.create_model("t5", "base")
summary = t5_model.generate_summary(input_json)

# Create a GPT-2 model
gpt2_model = ResumeModelFactory.create_model("gpt2", "medium")
summary = gpt2_model.generate_summary(input_json)

# Create a BART model
bart_model = ResumeModelFactory.create_model("bart", "large")
summary = bart_model.generate_summary(input_json)
"""