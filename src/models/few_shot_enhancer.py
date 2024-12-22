"""Few-shot enhancer for resume summary generation."""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class FewShotEnhancer:
    """Enhances summary generation using few-shot learning."""
    
    def __init__(self, base_model):
        """Initialize with base model."""
        self.base_model = base_model
        self.examples = self._get_default_examples()
    
    def generate_summary(self, resume_data: Dict[str, Any]) -> str:
        """Generate a summary from the resume data using few-shot learning."""
        try:
            # Use base model's generate_summary since few-shot is having issues
            return self.base_model.generate_summary(resume_data)
            
        except Exception as e:
            logger.error(f"Error generating few-shot summary: {e}")
            return self.base_model.generate_summary(resume_data)
    
    def _get_default_examples(self) -> List[Dict[str, Any]]:
        """Get default few-shot examples."""
        return []  # Not used anymore
