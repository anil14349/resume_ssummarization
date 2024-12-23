"""T5 model for resume summary generation."""
import logging
from typing import Dict, Any, List, Optional, Tuple
import torch
import re
import random

from transformers import T5ForConditionalGeneration, T5Tokenizer

logger = logging.getLogger(__name__)


class T5ResumeModel:
    """T5 model for generating resume summaries."""
    
    def __init__(self):
        """Initialize T5 model."""
        logger.info("Initializing T5 model")
        
        try:
            # Load model and tokenizer
            self.model_name = "t5-base"
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            
            # Set generation parameters
            self.max_length = 512
            self.min_length = 100
            self.num_beams = 4
            self.temperature = 0.7
            self.top_p = 0.9
            self.top_k = 50
            
            logger.info("Successfully initialized T5 model")
            
        except Exception as e:
            logger.error(f"Error initializing T5 model: {e}")
            raise
    
    def generate_summary(self, resume_data: Dict[str, Any]) -> str:
        """Generate a summary from the resume data."""
        try:
            name = resume_data.get('name', '')
            years = resume_data.get('years_experience', 0)
            skills = resume_data.get('skills', [])
            companies = resume_data.get('companies', [])
            achievements = resume_data.get('achievements', [])
            contact_info = resume_data.get('contact_info', {})
            
            # Build a basic summary starting with greeting
            summary = f"Hi, this is {name}"
            if years:
                summary += f" with {int(years)} years of experience"
            if skills:
                summary += f" specializing in {', '.join(skills[:5])}"
            summary += ". "
                
            if companies:
                summary += f"Currently working at {companies[0]}. "
                
            if achievements:
                summary += f"In my professional journey, {achievements[0]} "
                
            summary += "I am passionate about delivering exceptional results through innovative solutions."
            
            # Add contact information
            if contact_info:
                email = contact_info.get('email', '')
                phone = contact_info.get('phone', '')
                if email or phone:
                    summary += f" You can reach me at {email}"
                    if email and phone:
                        summary += f" or {phone}"
                    elif phone:
                        summary += f"{phone}"
                    summary += "."
            
            return self._clean_summary(summary)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Error generating summary."
    
    def _clean_summary(self, summary: str) -> str:
        """Clean and format the generated summary."""
        # Remove extra whitespace
        summary = re.sub(r'\s+', ' ', summary).strip()
        
        # Ensure proper sentence capitalization
        summary = '. '.join(s.capitalize() for s in summary.split('. '))
        
        # Remove any trailing periods
        summary = summary.rstrip('.')
        
        return summary
    
    def _validate_summary(self, summary: str) -> bool:
        """Validate generated summary.
        
        Args:
            summary: Summary text to validate
            
        Returns:
            True if summary is valid, False otherwise
        """
        if not summary:
            return False
            
        # Check minimum length
        if len(summary.split()) < 10:
            return False
            
        # Check maximum length
        if len(summary.split()) > 200:
            return False
            
        return True
