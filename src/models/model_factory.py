# models/model_factory.py
"""Factory for creating resume models."""
import logging
from typing import Optional

from .gpt2_model import GPT2Model

logger = logging.getLogger(__name__)


class ResumeModelFactory:
    """Factory for creating resume models."""
    
    @staticmethod
    def create_model(model_type: str) -> object:
        """Create a resume model.
        
        Args:
            model_type: Type of model to create ('gpt2')
            
        Returns:
            Model instance
        """
        logger.info(f"Creating model of type '{model_type}'")
        
        try:
            if model_type.lower() == 'gpt2':
                return GPT2Model()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
