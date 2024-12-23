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
            self.max_length = 1024  # Keep longer length for comprehensive summaries
            self.min_length = 256   # Keep minimum length for completeness
            self.num_beams = 6      # Increased beams for better quality and consistency
            self.length_penalty = 2.0  # Slightly reduced to not over-extend
            self.early_stopping = True
            self.no_repeat_ngram_size = 3
            self.temperature = 0.6   # Reduced temperature for more focused output
            self.top_p = 0.85       # More conservative nucleus sampling
            self.top_k = 40         # More conservative top-k
            self.repetition_penalty = 1.2  # Added to avoid repetition
            self.do_sample = False  # Disabled sampling for more deterministic output
            
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
