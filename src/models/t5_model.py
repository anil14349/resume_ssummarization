"""T5 model for resume summary generation."""
import logging
from typing import Dict, Any, List, Optional, Tuple
import torch
import re
import random

from transformers import T5ForConditionalGeneration, T5Tokenizer
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class T5ResumeModel(BaseModel):
    """T5 model for generating resume summaries."""
    
    def __init__(self):
        """Initialize T5 model."""
        super().__init__()
        logger.info("Initializing T5 model")
        
        try:
            # Load model and tokenizer
            self.model_name = "t5-base"
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            
            # Set generation parameters for professional summaries
            self.max_length = 1024  # Comprehensive summaries
            self.min_length = 256   # Ensure complete information
            self.num_beams = 6      # Higher quality beam search
            self.length_penalty = 2.0  # Balanced length
            self.early_stopping = True
            self.no_repeat_ngram_size = 3  # Avoid repetition
            self.temperature = 0.6   # Conservative sampling
            self.top_p = 0.85       # Professional focus
            self.top_k = 40         # Limited vocabulary range
            self.repetition_penalty = 1.2  # Avoid redundancy
            self.do_sample = False  # Deterministic output
            
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
            education = resume_data.get('education', [])
            projects = resume_data.get('projects', [])
            
            # Build professional summary
            summary_parts = []
            
            # Professional introduction
            intro = f"I'm {name}"
            if current_role:
                intro += f", a seasoned {current_role.lower()}"
            summary_parts.append(intro)
            
            # Experience and expertise
            if years:
                summary_parts.append(f"with {int(years)} years of demonstrated experience")
            
            # Core skills and expertise
            if skills:
                core_skills = skills[:5]  # Focus on core skills
                additional_skills = skills[5:8]  # Limit additional skills
                summary_parts.append(f"specializing in {', '.join(core_skills)}")
                if additional_skills:
                    summary_parts.append(f"with proficiency in {', '.join(additional_skills)}")
            
            # Professional experience
            if companies:
                if len(companies) > 1:
                    summary_parts.append(f"Currently contributing to {companies[0]}, previously gained valuable experience at {', '.join(companies[1:2])}")
                else:
                    summary_parts.append(f"Currently contributing to {companies[0]}")
            
            # Key achievements (focused on most significant)
            if achievements:
                summary_parts.append("Key professional achievements include:")
                for achievement in achievements[:2]:  # Limited to top 2 achievements
                    summary_parts.append(f"• {achievement}")
            
            # Education (if relevant)
            if education:
                edu_parts = []
                for edu in education[:1]:  # Focus on highest education
                    if isinstance(edu, dict):
                        degree = edu.get('degree', '')
                        institution = edu.get('institution', '')
                        if degree and institution:
                            edu_parts.append(f"{degree} from {institution}")
                if edu_parts:
                    summary_parts.append("Education: " + ", ".join(edu_parts))
            
            # Notable projects (if highly relevant)
            if projects:
                project_parts = []
                for project in projects[:1]:  # Focus on most significant project
                    if isinstance(project, dict):
                        project_name = project.get('name', '')
                        project_desc = project.get('description', '')
                        if project_name and project_desc:
                            project_parts.append(f"{project_name}: {project_desc}")
                if project_parts:
                    summary_parts.append("Significant project: " + project_parts[0])
            
            # Professional objective (more focused)
            summary_parts.append("Seeking to leverage proven expertise to deliver value and drive success in challenging roles")
            
            # Join parts with proper punctuation and formatting
            summary = '. '.join(part.strip() for part in summary_parts if part.strip())
            summary = summary.replace('..','.').replace('. .','.').strip()
            
            # Add contact information at the end
            if contact_info:
                contact_parts = []
                if contact_info.get('email'):
                    contact_parts.append(f"Email: {contact_info['email']}")
                if contact_info.get('phone'):
                    contact_parts.append(f"Phone: {contact_info['phone']}")
                if contact_info.get('linkedin'):
                    contact_parts.append(f"LinkedIn: {contact_info['linkedin']}")
                
                if contact_parts:
                    summary += f"\n\nContact Information: {', '.join(contact_parts)}"
            
            # Clean and return the summary
            return self._clean_summary(summary)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Error generating summary."
    
    def _clean_summary(self, text: str) -> str:
        """Clean and format the generated summary."""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Fix punctuation spacing
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        # Capitalize first letter of sentences
        text = '. '.join(s.strip().capitalize() for s in text.split('.') if s.strip())
        return text.strip()
