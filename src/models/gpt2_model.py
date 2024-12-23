"""GPT-2 model for resume summary generation."""
from typing import Dict, Any
import re
import random
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .base_model import BaseModel

logger = logging.getLogger(__name__)

class GPT2Model(BaseModel):
    """GPT-2 model for generating resume summaries."""

    def __init__(self):
        """Initialize GPT-2 model and tokenizer."""
        super().__init__()
        logger.info("Initializing GPT-2 model")
        
        try:
            self.model_name = "gpt2"
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            
            # Set generation parameters
            self.max_length = 512
            self.min_length = 100
            self.num_beams = 4
            self.length_penalty = 2.0
            self.early_stopping = True
            self.no_repeat_ngram_size = 2
            self.temperature = 0.7
            
        except Exception as e:
            logger.error(f"Error initializing GPT-2 model: {e}")
            raise

    def generate_summary(self, resume_data: Dict[str, Any]) -> str:
        """Generate a summary from resume data."""
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
                if contact_info.get('linkedin'):
                    contact_parts.append(contact_info['linkedin'])
                
                if contact_parts:
                    summary += f" Contact me at: {' or '.join(contact_parts)}."
            
            # Clean and return the summary
            return self._clean_summary(summary)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Error generating summary."

    def _format_name(self, name: str) -> str:
        """Format name with proper capitalization."""
        return ' '.join(word.capitalize() for word in name.split())

    def _validate_summary(self, summary: str) -> bool:
        """Validate the generated summary."""
        # Check if summary is empty
        if not summary:
            return False
            
        # Check length constraints
        words = summary.split()
        if len(words) < 10:  # Too short
            return False
        if len(words) > 200:  # Too long
            return False
            
        return True

    def _clean_summary(self, summary: str) -> str:
        """Clean and format the generated summary."""
        try:
            # Basic cleanup
            summary = summary.strip()
            if not summary:
                return summary
            
            # Clean up formatting artifacts
            summary = re.sub(r'\s+•\s+', ' ', summary)  # Remove bullets
            summary = re.sub(r'\[(?:skills and expertise|achievements and impact|current role and company|[^\]]+)\]', '', summary)  # Remove section headers
            
            # Clean up self-references
            summary = re.sub(r'(?i)\b(?:my name is|i am called|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', r'\1', summary)
            summary = re.sub(r'(?i)\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:speaking|here)\b', r'\1', summary)
            
            # Improve transitions
            summary = re.sub(r'(?i)\b(?:recently|in my current role|presently)\b', 
                           lambda m: random.choice(['In my current position', 'As part of my role', 'In my professional journey']), 
                           summary)
            
            # Fix spacing around punctuation
            summary = re.sub(r'\s+([.,!?])', r'\1', summary)
            summary = re.sub(r'([.,!?])(\S)', r'\1 \2', summary)
            summary = re.sub(r'\s{2,}', ' ', summary)
            
            # Fix email addresses (remove spaces in domain)
            summary = re.sub(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+)\s*\.\s*([a-zA-Z]{2,})', r'\1.\2', summary)
            
            # Capitalize first letter and after periods
            summary = summary[0].upper() + summary[1:]
            summary = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), summary)
            
            # Remove trailing punctuation (except for email addresses)
            if not re.search(r'@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', summary):
                summary = re.sub(r'[.,!?]+$', '', summary)
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning summary: {e}")
            return summary

    def _clean_metrics(self, text: str) -> str:
        """Clean and format metrics in text."""
        # Format percentages
        text = re.sub(r'\*(\d+(?:\.\d+)?)\*\s*%', r'\1%', text)
        text = re.sub(r'\*(\d+(?:\.\d+)?%)\*', r'\1', text)
        
        # Format currency
        text = re.sub(r'\*\$(\d+(?:\.\d+)?[KMB]?)\*', r'$\1', text)
        text = re.sub(r'\*(\d+(?:\.\d+)?)\*\s*(?=million|billion|thousand)', r'\1', text)
        
        # Format numbers with units
        text = re.sub(r'\*(\d+(?:\.\d+)?)\*\s+(times|x)', r'\1 \2', text)
        text = re.sub(r'\*(\d+(?:\.\d+)?x)\*', r'\1x', text)
        
        # Format ranges
        text = re.sub(r'\*(\d+)\*\s*-\s*\*(\d+)\*', r'\1-\2', text)
        text = re.sub(r'\*(\d+)\*\s*to\s*\*(\d+)\*', r'\1 to \2', text)
        
        return text
