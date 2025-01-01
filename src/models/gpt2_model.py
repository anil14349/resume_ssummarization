"""GPT-2 model for resume summary generation."""
from typing import Dict, Any
import re
import random
import logging
from transformers import pipeline
from .base_model import BaseModel

logger = logging.getLogger(__name__)

class GPT2Model(BaseModel):
    """GPT-2 model for generating resume summaries."""

    def __init__(self):
        """Initialize GPT-2 model and tokenizer."""
        super().__init__()
        logger.info("Initializing GPT-2 model")
        
        try:
            # Initialize the pipeline
            self.generator = pipeline("text-generation", model="gpt2")
            
            # Set generation parameters
            self.max_length = 1024     # Set to GPT-2's max context length
            self.min_length = 256      # Minimum length for a good response
            self.num_return_sequences = 1
            self.temperature = 0.8     # Slightly increased for more creative scene descriptions
            self.top_p = 0.92         # Increased for more diverse outputs
            self.top_k = 50           # Increased for more vocabulary variety
            
        except Exception as e:
            logger.error(f"Error initializing GPT-2 model: {e}")
            raise

    def generate_summary(self, resume_data: Dict[str, Any]) -> str:
        """Generate a summary from resume data."""
        try:
            # Extract and format data
            name = self._format_name(resume_data.get('name', ''))
            years = resume_data.get('years_experience', 0)
            skills = resume_data.get('skills', [])
            companies = resume_data.get('companies', [])
            achievements = resume_data.get('achievements', [])
            current_role = resume_data.get('current_role', '')
            education = resume_data.get('education', [])
            
            # Ensure all lists are properly formatted
            skills_str = ', '.join(str(skill) for skill in skills[:5]) if isinstance(skills, list) else str(skills)  # Limit to top 5 skills
            companies_str = ', '.join(str(company) for company in companies[:2]) if isinstance(companies, list) else str(companies)  # Limit to 2 companies
            achievements_str = ', '.join(str(achievement) for achievement in achievements[:2]) if isinstance(achievements, list) else str(achievements)  # Limit to 2 achievements
            
            # Format education string
            education_list = []
            for edu in education[:1]:  # Take only the first education entry
                if isinstance(edu, dict):
                    degree = edu.get('degree', '')
                    institution = edu.get('institution', '')
                    if degree and institution:
                        education_list.append(f"{degree} from {institution}")
            education_str = ', '.join(education_list)
            
            # Create a more concise prompt
            prompt = (
                f"Create a professional video script for {name}. Experience: {years} years. "
                f"Skills: {skills_str}. Current company: {companies_str}. "
                f"Key achievement: {achievements_str}. Education: {education_str}\n\n"
                "Format:\n"
                "1. Intro\n- Caption: Professional intro\n- Audio: Role and expertise\n- Visual: Office setting\n\n"
                "2. Experience\n- Caption: Career highlights\n- Audio: Experience summary\n- Visual: Industry symbols\n\n"
                "3. Skills\n- Caption: Core competencies\n- Audio: Key skills\n- Visual: Skill icons\n\n"
                "4. Achievement\n- Caption: Success story\n- Audio: Key accomplishment\n- Visual: Success imagery\n\n"
                "5. Future\n- Caption: Career goals\n- Audio: Aspirations\n- Visual: Growth symbols\n\n"
                "6. Contact\n- Caption: Connect\n- Audio: Call to action\n- Visual: Contact details\n\n"
                "Script:"
            )
            
            try:
                # Generate text using the pipeline with error handling
                outputs = self.generator(
                    prompt,
                    max_length=self.max_length,
                    num_return_sequences=self.num_return_sequences,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    pad_token_id=50256  # GPT-2's EOS token ID
                )
                
                if not outputs or not isinstance(outputs, list):
                    logger.error("No output generated from the model")
                    return self._generate_template_summary(resume_data)
                
                output = outputs[0]
                if not isinstance(output, dict) or 'generated_text' not in output:
                    logger.error("Unexpected output format from the model")
                    return self._generate_template_summary(resume_data)
                
                generated_text = output['generated_text']
                
                # Extract the generated script (remove the prompt)
                parts = generated_text.split("Script:")
                if len(parts) < 2:
                    logger.error("Generated text does not contain expected format")
                    return self._generate_template_summary(resume_data)
                
                script = parts[-1].strip()
                
                # Clean and validate the script
                script = self._clean_summary(script)
                if not self._validate_summary(script):
                    logger.warning("Generated script failed validation, falling back to template")
                    return self._generate_template_summary(resume_data)
                    
                return script
                
            except Exception as model_error:
                logger.error(f"Error during model generation: {str(model_error)}")
                return self._generate_template_summary(resume_data)
            
        except Exception as e:
            logger.error(f"Error in generate_summary: {str(e)}")
            return self._generate_template_summary(resume_data)

    def _generate_template_summary(self, resume_data: Dict[str, Any]) -> str:
        """Fallback method to generate summary using templates."""
        try:
            name = self._format_name(resume_data.get('name', 'the candidate'))
            years = resume_data.get('years_experience', 0)
            skills = resume_data.get('skills', [])
            companies = resume_data.get('companies', [])
            achievements = resume_data.get('achievements', [])
            current_role = resume_data.get('current_role', '')
            education = resume_data.get('education', [])
            contact = resume_data.get('contact_info', {})
            
            # Format lists
            skills_str = ', '.join(str(skill) for skill in skills[:5]) if skills else "various professional skills"
            companies_str = ', '.join(str(company) for company in companies[:2]) if companies else "various companies"
            achievements_str = achievements[0] if achievements else "demonstrated success in professional endeavors"
            
            # Format education
            education_str = ""
            if education and isinstance(education, list) and len(education) > 0:
                edu = education[0]
                if isinstance(edu, dict):
                    degree = edu.get('degree', '')
                    institution = edu.get('institution', '')
                    if degree and institution:
                        education_str = f"{degree} from {institution}"
            
            # Create structured video script
            script = (
                "1. Introduction\n"
                f"- Caption: Meet {name}, A Professional Journey\n"
                f"- Audio: Welcome to my professional story. I'm {name}, with {years} years of experience in {companies_str}.\n"
                "- Visual: Professional headshot in a modern office setting\n\n"
                
                "2. Experience Overview\n"
                "- Caption: Professional Excellence\n"
                f"- Audio: Throughout my {years}-year career, I've developed expertise in {skills_str}.\n"
                "- Visual: Dynamic montage of professional environments\n\n"
                
                "3. Key Skills\n"
                "- Caption: Core Competencies\n"
                f"- Audio: My key strengths include {skills_str}.\n"
                "- Visual: Animated icons representing each skill\n\n"
                
                "4. Major Achievement\n"
                "- Caption: Success Story\n"
                f"- Audio: One of my proudest achievements is that I {achievements_str}.\n"
                "- Visual: Graphs and charts showing success metrics\n\n"
                
                "5. Professional Goals\n"
                "- Caption: Looking Forward\n"
                f"- Audio: With my background in {skills_str[:2]}, I'm excited to take on new challenges.\n"
                "- Visual: Forward-looking imagery of innovation and growth\n\n"
                
                "6. Connect\n"
                "- Caption: Let's Connect\n"
                f"- Audio: I'm always open to new opportunities and professional connections. You can reach me at {contact.get('email', '')}.\n"
                "- Visual: Professional contact information display\n"
            )
            
            return script
            
        except Exception as e:
            logger.error(f"Error generating template summary: {str(e)}")
            return f"Professional video script for {name}"

    def _format_name(self, name: str) -> str:
        """Format name with proper capitalization."""
        return ' '.join(word.capitalize() for word in name.split())

    def _validate_summary(self, summary: str) -> bool:
        """Validate that the summary meets our requirements."""
        if not summary:
            return False
            
        # Check minimum length
        if len(summary) < 100:  # Require at least 100 characters
            return False
            
        # Check for required sections
        required_sections = ["1.", "2.", "3.", "4.", "5.", "6."]
        section_count = sum(1 for section in required_sections if section in summary)
        if section_count < 4:  # Allow some flexibility but require most sections
            return False
            
        # Check for required components
        required_components = ["Caption:", "Audio:", "Visual:"]
        component_count = sum(1 for component in required_components if component in summary)
        if component_count < 6:  # Require at least 2 complete sections
            return False
            
        return True

    def _clean_summary(self, summary: str) -> str:
        """Clean and format the generated summary."""
        try:
            summary = summary.strip()
            if not summary:
                return summary
            
            # Remove any remaining prompt text
            summary = re.sub(r'.*?Professional Summary:', '', summary, flags=re.DOTALL)
            
            # Basic cleanup
            summary = re.sub(r'\s+', ' ', summary)  # Remove extra whitespace
            summary = re.sub(r'([.,!?])(\S)', r'\1 \2', summary)  # Fix spacing after punctuation
            summary = re.sub(r'\s+([.,!?])', r'\1', summary)  # Fix spacing before punctuation
            
            # Capitalize first letter
            summary = summary[0].upper() + summary[1:] if summary else summary
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning summary: {e}")
            return summary
