# models/model_factory.py
"""Factory for creating resume models."""
import logging
from typing import Any

from models.bart_model import BartResumeModel   
from .gpt2_model import GPT2Model
from .t5_model import T5ResumeModel


logger = logging.getLogger(__name__)

def create_model(model_type: str) -> Any:
    """Create a model of the specified type.
    
    Args:
        model_type: Type of model to create
        
    Returns:
        Model instance
    """
    try:
        logger.info(f"Creating model of type '{model_type}'")
        
        if model_type.lower() == 'gpt2':
            return GPT2Model()
        elif model_type.lower() == 't5':
            return T5ResumeModel()
        elif model_type.lower() == 'bart':
            return BartResumeModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise
