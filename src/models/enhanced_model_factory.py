"""Factory for creating enhanced models."""
import logging
from typing import Optional

from .gpt2_model import GPT2Model
from .few_shot_enhancer import FewShotEnhancer

logger = logging.getLogger(__name__)


class EnhancedModelFactory:
    """Factory for creating enhanced models."""
    
    @staticmethod
    def create_model(model_type: str, enhancer_type: Optional[str] = None):
        """Create an enhanced model.
        
        Args:
            model_type: Type of model to create ('gpt2')
            enhancer_type: Type of enhancer to use ('few_shot')
            
        Returns:
            Enhanced model instance
        """
        logger.info(f"Creating model of type '{model_type}' with enhancer '{enhancer_type}'")
        
        try:
            # Create base model
            if model_type.lower() == 'gpt2':
                model = GPT2Model()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Add enhancer if specified
            if enhancer_type:
                if enhancer_type.lower() == 'few_shot':
                    model = FewShotEnhancer(model)
                else:
                    raise ValueError(f"Unknown enhancer type: {enhancer_type}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
