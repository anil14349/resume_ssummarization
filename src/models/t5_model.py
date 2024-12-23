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
    
    def _format_name(self, name: str) -> str:
        """Format name with proper capitalization."""
        return ' '.join(word.capitalize() for word in name.split())
    
    def generate_summary(self, resume_data: Dict[str, Any]) -> str:
        """Generate a summary from the resume data."""
        try:
            # Extract data
            name = self._format_name(resume_data.get('name', ''))
            years = resume_data.get('years_experience', 0)
            skills = resume_data.get('skills', [])
            companies = resume_data.get('companies', [])
            achievements = resume_data.get('achievements', [])
            contact_info = resume_data.get('contact_info', {})
            current_role = resume_data.get('current_role', '')
            
            # Build professional summary
            summary_parts = []
            
            # Introduction with name and role
            intro = f"I'm {name}"
            if current_role:
                intro += f", a {current_role.lower()}"
            summary_parts.append(intro)
            
            # Experience and expertise
            if years:
                summary_parts.append(f"with {int(years)} years of experience")
            if skills:
                summary_parts.append(f"specializing in {', '.join(skills[:5])}")
            
            # Current company
            if companies:
                summary_parts.append(f"Currently working at {companies[0]}")
                
            # Key achievement
            if achievements:
                summary_parts.append(f"Notable achievement: {achievements[0]}")
                
            # Join parts with proper punctuation
            summary = '. '.join(summary_parts) + '.'
            
            # Add contact information at the end
            if contact_info:
                contact_parts = []
                if contact_info.get('email'):
                    contact_parts.append(contact_info['email'])
                if contact_info.get('phone'):
                    contact_parts.append(contact_info['phone'])
                
                if contact_parts:
                    summary += f" Contact me at: {' or '.join(contact_parts)}."
            
            return self._clean_summary(summary)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Error generating summary."
    
    def _clean_summary(self, summary: str) -> str:
        """Clean and format the generated summary."""
        try:
            # Basic cleanup
            summary = summary.strip()
            if not summary:
                return summary
            
            # Fix email addresses (remove spaces in domain)
            summary = re.sub(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+)\s*\.\s*([a-zA-Z]{2,})', r'\1.\2', summary)
            
            # Remove extra whitespace
            summary = re.sub(r'\s+', ' ', summary)
            
            # Ensure proper sentence capitalization
            summary = '. '.join(s.capitalize() for s in summary.split('. '))
            
            # Remove trailing period (except for email addresses)
            if not re.search(r'@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', summary):
                summary = summary.rstrip('.')
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning summary: {e}")
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
