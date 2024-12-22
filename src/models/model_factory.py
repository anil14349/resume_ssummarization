# models/model_factory.py
"""Factory for creating resume models."""
import logging
from typing import Any

from .gpt2_model import GPT2Model

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
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise
