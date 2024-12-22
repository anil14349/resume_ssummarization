"""Factory for creating enhanced resume models."""
import logging
from typing import Optional

from .model_factory import ResumeModelFactory
from .few_shot_enhancer import FewShotEnhancer

logger = logging.getLogger(__name__)


class EnhancedModelFactory:
    """Factory for creating enhanced resume models."""
    
    @staticmethod
    def create_model(model_type: str, enhancer_type: Optional[str] = None) -> object:
        """Create an enhanced resume model.
        
        Args:
            model_type: Type of model to create ('gpt2')
            enhancer_type: Type of enhancer to use ('few_shot' or None)
            
        Returns:
            Enhanced model instance
        """
        logger.info(f"Creating enhanced model of type '{model_type}' with enhancer '{enhancer_type}'")
        
        try:
            # Create base model
            base_model = ResumeModelFactory.create_model(model_type)
            
            # Return base model directly
            return base_model
            
        except Exception as e:
            logger.error(f"Error creating enhanced model: {e}")
            raise
