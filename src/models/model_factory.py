# models/model_factory.py
"""Factory for creating model instances."""
import logging
from typing import Any
from .gpt2_model import GPT2Model
from .t5_model import T5ResumeModel
from .bart_model import BartResumeModel

logger = logging.getLogger(__name__)

def create_model(model_type: str, model_size: str = 'base') -> Any:
    """Create a model of the specified type and size.

    Args:
        model_type: Type of model to create (gpt2, t5, or bart)
        model_size: Size of the model (base, small, medium, large)

    Returns:
        Model instance

    Raises:
        ValueError: If model type is invalid
    """
    try:
        logger.info(f"Creating model of type '{model_type}' with size '{model_size}'")

        if model_type.lower() == 'gpt2':
            return GPT2Model()
        elif model_type.lower() == 't5':
            return T5ResumeModel()
        elif model_type.lower() == 'bart':
            return BartResumeModel()
        else:
            raise ValueError(f"Invalid model type: {model_type}")
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise
