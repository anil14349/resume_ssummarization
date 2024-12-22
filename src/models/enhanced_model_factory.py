"""Factory for creating enhanced models."""
import logging
from typing import Optional

from .gpt2_model import GPT2ResumeModel
from .t5_model import T5ResumeModel
from .bart_model import BartResumeModel
from .rag_enhancer import RAGEnhancer
from .few_shot_enhancer import FewShotEnhancer
from .cot_enhancer import ChainOfThoughtEnhancer
from .ml_cleanup import MLCleanupEnhancer

logger = logging.getLogger(__name__)

class EnhancedModelFactory:
    """Factory for creating enhanced models."""
    
    def __init__(self):
        """Initialize model factory with enhancers."""
        logger.info("Initializing EnhancedModelFactory")
        
        try:
            # Initialize enhancers
            self.rag = RAGEnhancer()
            self.few_shot = FewShotEnhancer()
            self.cot = ChainOfThoughtEnhancer()
            self.cleanup = MLCleanupEnhancer()
            
            # Add default examples
            self.few_shot.add_default_examples()
            logger.info("Successfully initialized all enhancers")
            
        except Exception as e:
            logger.error(f"Error initializing factory: {e}")
            raise
    
    def create_model(
        self,
        model_type: str,
        enhancer_type: Optional[str] = None,
        device: str = "cpu"
    ) -> object:
        """Create and return an enhanced model.
        
        Args:
            model_type: Type of base model (gpt2, t5, bart)
            enhancer_type: Type of enhancement (rag, few_shot, cot)
            device: Device to run model on (cuda/cpu)
        """
        logger.info(f"Creating model type: {model_type} with enhancer: {enhancer_type}")
        
        try:
            # Create base model
            if "bart" in model_type.lower():
                model = BartResumeModel(model_type, device)
            elif "t5" in model_type.lower():
                model = T5ResumeModel(model_type, device)
            elif "gpt2" in model_type.lower():
                model = GPT2ResumeModel(model_type, device)
            else:
                logger.error(f"Unsupported model type: {model_type}")
                raise ValueError(f"Unsupported model type: {model_type}")
                
            # Store original methods
            original_generate = model.generate_summary
            original_clean = model.clean_output
            
            # Enhanced generate method
            def enhanced_generate(input_data):
                """Enhanced generation with RAG and other enhancers."""
                try:
                    # Get base prompt
                    prompt = model.generate_prompt(input_data)
                    
                    # Add enhancements based on type
                    if enhancer_type == "rag":
                        rag_examples = self.rag.enhance_prompt(input_data)
                        if rag_examples:
                            prompt = f"{rag_examples}\n\n{prompt}"
                    elif enhancer_type == "few_shot":
                        examples = self.few_shot.get_examples(input_data)
                        if examples:
                            prompt = f"{examples}\n\n{prompt}"
                    elif enhancer_type == "cot":
                        cot_prompt = self.cot.enhance_prompt(input_data)
                        if cot_prompt:
                            prompt = f"{cot_prompt}\n\n{prompt}"
                            
                    # Generate with enhanced prompt
                    raw_output = original_generate(prompt)
                    
                    # Clean output using ML cleanup
                    style_example = self.few_shot.get_style_example()
                    cleaned_output = self.cleanup.clean_output(
                        raw_output,
                        style_example
                    )
                    
                    return cleaned_output
                    
                except Exception as e:
                    logger.error(f"Error in enhanced generation: {e}")
                    raise
                    
            # Replace methods
            model.generate_summary = enhanced_generate
            model.clean_output = self.cleanup.clean_output
            
            logger.info("Successfully created enhanced model")
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
