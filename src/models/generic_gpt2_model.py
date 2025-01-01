from typing import Dict, Any, Optional
import logging
from transformers import pipeline
from .base_model import BaseModel
import re

logger = logging.getLogger(__name__)

class GenericGPT2Model(BaseModel):
    """A GPT-2 model that can generate video scripts from raw resume text."""
    
    def __init__(self):
        """Initialize GPT-2 model and tokenizer."""
        super().__init__()
        logger.info("Initializing Generic GPT-2 model")
        
        try:
            # Initialize the pipeline
            self.generator = pipeline("text-generation", model="gpt2")
            
            # Set generation parameters for more factual output
            self.max_length = 1024      # Set to GPT-2's max context length
            self.min_length = 256       # Minimum length for a good response
            self.num_return_sequences = 1
            self.temperature = 0.3      # Lower temperature for more focused and less random output
            self.top_p = 0.85          # Lower top_p for more conservative sampling
            self.top_k = 30            # Lower top_k to limit vocabulary diversity
            self.repetition_penalty = 1.2  # Penalize repetition
            
        except Exception as e:
            logger.error(f"Error initializing Generic GPT-2 model: {e}")
            raise

    def generate_summary(self, resume_text: str) -> str:
        """Generate a video script summary from raw resume text.
        
        Args:
            resume_text: Raw text content of the resume
            
        Returns:
            A structured video script
        """
        try:
            # Clean and preprocess the resume text
            cleaned_text = self._preprocess_text(resume_text)
            
            # Extract key information for more focused generation
            name = self._extract_name(cleaned_text)
            email = self._extract_email(cleaned_text)
            key_points = self._extract_key_points(cleaned_text)
            
            # Create a more focused prompt that emphasizes factual content
            prompt = (
                "Create a professional video script based strictly on the following resume information. "
                "Do not add any information that is not present in the resume. "
                f"Resume:\n{cleaned_text}\n\n"
                "Format each section with only factual information from the resume:\n\n"
                "1. Intro\n- Caption: Name and current role\n- Audio: Factual introduction\n- Visual: Professional setting\n\n"
                "2. Experience\n- Caption: Years and companies\n- Audio: Actual experience\n- Visual: Work environment\n\n"
                "3. Skills\n- Caption: Key skills from resume\n- Audio: Listed skills and expertise\n- Visual: Skill representation\n\n"
                "4. Achievement\n- Caption: Verified accomplishment\n- Audio: Specific achievement details\n- Visual: Achievement visual\n\n"
                "5. Future\n- Caption: Career direction\n- Audio: Based on experience\n- Visual: Professional growth\n\n"
                "6. Contact\n- Caption: Contact details\n- Audio: Actual contact information\n- Visual: Contact display\n\n"
                "Script:"
            )
            
            try:
                # Generate text using the pipeline with strict parameters
                outputs = self.generator(
                    prompt,
                    max_length=self.max_length,
                    min_length=self.min_length,
                    num_return_sequences=self.num_return_sequences,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    repetition_penalty=self.repetition_penalty,
                    pad_token_id=50256  # GPT-2's EOS token ID
                )
                
                if not outputs or not isinstance(outputs, list):
                    logger.error("No output generated from the model")
                    return self._generate_template_summary(resume_text)
                
                output = outputs[0]
                if not isinstance(output, dict) or 'generated_text' not in output:
                    logger.error("Unexpected output format from the model")
                    return self._generate_template_summary(resume_text)
                
                generated_text = output['generated_text']
                
                # Extract the generated script (remove the prompt)
                parts = generated_text.split("Script:")
                if len(parts) < 2:
                    logger.error("Generated text does not contain expected format")
                    return self._generate_template_summary(resume_text)
                
                script = parts[-1].strip()
                
                # Clean and validate the script
                script = self._clean_summary(script)
                if not self._validate_summary(script, resume_text):
                    logger.warning("Generated script failed validation, falling back to template")
                    return self._generate_template_summary(resume_text)
                    
                return script
                
            except Exception as model_error:
                logger.error(f"Error during model generation: {str(model_error)}")
                return self._generate_template_summary(resume_text)
            
        except Exception as e:
            logger.error(f"Error in generate_summary: {str(e)}")
            return self._generate_template_summary(resume_text)

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess the input resume text."""
        if not text:
            return ""
            
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (to fit in GPT-2's context window)
        max_chars = 500  # Leave room for the prompt and generation
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
            
        return text

    def _generate_template_summary(self, resume_text: str) -> str:
        """Generate a template-based summary when model generation fails."""
        try:
            # Extract key information
            name = self._extract_name(resume_text)
            email = self._extract_email(resume_text)
            key_points = self._extract_key_points(resume_text)
            
            # Format skills and achievements nicely
            hr_skills = [s for s in key_points['skills'] if s in [
                'recruitment', 'talent acquisition', 'employee relations', 'hr policies',
                'benefits administration', 'performance management', 'training',
                'workforce planning', 'compensation', 'labor relations', 'osha compliance'
            ]]
            
            business_skills = [s for s in key_points['skills'] if s in [
                'project management', 'team leadership', 'strategic planning',
                'budget management', 'risk management', 'stakeholder management',
                'communication', 'leadership', 'problem solving'
            ]]
            
            technical_skills = [s for s in key_points['skills'] if s not in hr_skills + business_skills]
            
            # Format skills by category
            skill_text = ""
            if hr_skills:
                skill_text += f"HR expertise in {', '.join(hr_skills[:3])}"
            if business_skills:
                skill_text += f"{', and ' if skill_text else ''}business skills including {', '.join(business_skills[:2])}"
            if technical_skills:
                skill_text += f"{', along with ' if skill_text else ''}technical proficiency in {', '.join(technical_skills[:2])}"
            
            # Get company experience
            companies = key_points['companies']
            company_text = f"experience at {', '.join(companies)}" if companies else "professional experience in various organizations"
            
            # Get key achievements
            achievements = key_points['achievements']
            achievement = achievements[0] if achievements else "proven track record of success in HR management and process improvement"
            
            # Create a factual template
            template = f"""1. Introduction
- Caption: {name or 'Professional'} - HR Professional
- Audio: Hi, I'm {name or 'a seasoned HR professional'} with {skill_text}.
- Visual: Professional headshot with HR environment background

2. Experience Overview
- Caption: Professional Journey
- Audio: Throughout my {company_text}, I've focused on implementing effective HR strategies and improving organizational efficiency.
- Visual: Timeline showing career progression and company logos

3. Key Skills
- Caption: Core HR Competencies
- Audio: My expertise spans {skill_text}.
- Visual: Infographic showing skill categories: HR, Business, and Technical

4. Major Achievement
- Caption: Notable Impact
- Audio: {achievement}
- Visual: Data visualization showing improvement metrics and results

5. Professional Goals
- Caption: Future Vision
- Audio: I aim to leverage my experience in {', '.join(hr_skills[:2] if hr_skills else ['HR management', 'process improvement'])} to drive organizational excellence.
- Visual: Forward-looking imagery of modern HR practices and innovation

6. Connect
- Caption: Contact Details
- Audio: Let's connect to discuss how I can bring value to your organization. Reach me at {email or 'my professional email'}.
- Visual: Professional contact information with LinkedIn and email
"""
            return template
            
        except Exception as e:
            logger.error(f"Error generating template summary: {str(e)}")
            return "Professional video script"

    def _extract_name(self, text: str) -> str:
        """Extract name from resume text."""
        # Try to get name from the first line
        lines = text.split('\n')
        if lines:
            return lines[0].strip()
        return ""

    def _extract_email(self, text: str) -> str:
        """Extract email from resume text."""
        import re
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        match = re.search(email_pattern, text)
        return match.group(0) if match else ""

    def _extract_key_points(self, text: str) -> dict:
        """Extract key information from resume text."""
        key_points = {
            'skills': [],
            'companies': [],
            'achievements': []
        }
        
        # Common professional skills to look for
        common_skills = [
            # Technical Skills
            'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'node.js',
            'aws', 'azure', 'docker', 'kubernetes', 'git', 'ci/cd', 'agile',
            
            # Business Skills
            'project management', 'team leadership', 'strategic planning',
            'budget management', 'risk management', 'stakeholder management',
            
            # HR Skills
            'recruitment', 'talent acquisition', 'employee relations', 'hr policies',
            'benefits administration', 'performance management', 'training',
            'workforce planning', 'compensation', 'labor relations', 'osha compliance',
            'affirmative action', 'diversity', 'inclusion', 'employee retention',
            
            # Soft Skills
            'communication', 'leadership', 'problem solving', 'decision making',
            'time management', 'conflict resolution', 'negotiation', 'teamwork'
        ]
        
        # Extract skills
        text_lower = text.lower()
        
        # Look for skills in the common skills list
        for skill in common_skills:
            if skill in text_lower:
                key_points['skills'].append(skill)
        
        # Look for skills after common indicators
        skill_indicators = [
            'proficient in', 'skilled in', 'expertise in', 'experience with',
            'knowledge of', 'certified in', 'specializing in', 'trained in'
        ]
        
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower()
            
            # Check for skill indicators
            for indicator in skill_indicators:
                if indicator in line_lower:
                    # Extract skills after the indicator
                    skills_part = line_lower.split(indicator)[-1]
                    # Split by common separators
                    skills = [
                        s.strip(' .,;()') 
                        for s in re.split(r'[,;/|]', skills_part)
                    ]
                    # Add non-empty skills
                    key_points['skills'].extend(
                        [s for s in skills if s and len(s) > 2]
                    )
        
        # Extract companies
        company_indicators = [
            'worked at', 'employed by', 'company:', 'employer:',
            'corporation', 'inc.', 'ltd.', 'llc', 'group'
        ]
        
        for line in lines:
            line_lower = line.lower()
            
            # Check for company indicators
            for indicator in company_indicators:
                if indicator in line_lower:
                    # Try to extract company name
                    parts = re.split(r'[-|]', line)
                    for part in parts:
                        # Look for company-like strings
                        company_matches = re.findall(
                            r'([A-Z][A-Za-z\s&]+(?:Inc\.|LLC|Ltd\.|Corporation|Group|Company)?)',
                            part
                        )
                        for company in company_matches:
                            if len(company) > 2 and company.strip() not in key_points['companies']:
                                key_points['companies'].append(company.strip())
        
        # Extract achievements
        achievement_indicators = [
            'achieved', 'accomplished', 'led', 'developed', 'increased',
            'reduced', 'improved', 'created', 'implemented', 'managed',
            'launched', 'established', 'generated', 'saved', 'awarded'
        ]
        
        for line in lines:
            line_lower = line.lower()
            # Check for achievement indicators at the start of the line
            if any(line_lower.strip().startswith(indicator) for indicator in achievement_indicators):
                achievement = line.strip()
                if achievement and len(achievement) > 20:  # Minimum meaningful length
                    key_points['achievements'].append(achievement)
            # Also look for quantifiable achievements
            elif any(char.isdigit() for char in line):
                if any(word in line_lower for word in ['increase', 'decrease', 'reduce', 'improve', 'save']):
                    achievement = line.strip()
                    if achievement and len(achievement) > 20:
                        key_points['achievements'].append(achievement)
        
        # Remove duplicates while preserving order
        key_points['skills'] = list(dict.fromkeys(key_points['skills']))
        key_points['companies'] = list(dict.fromkeys(key_points['companies']))
        key_points['achievements'] = list(dict.fromkeys(key_points['achievements']))
        
        return key_points

    def _validate_summary(self, summary: str, resume_text: str) -> bool:
        """Validate that the summary meets our requirements."""
        try:
            # Basic validation
            if not summary:
                logger.error("Empty summary")
                return False
                
            # Check for required sections
            required_sections = ['1.', '2.', '3.', '4.', '5.', '6.']
            for section in required_sections:
                if section not in summary:
                    logger.error(f"Missing section {section}")
                    return False
            
            # Check for required components in each section
            required_components = ['Caption:', 'Audio:', 'Visual:']
            for component in required_components:
                if component not in summary:
                    logger.error(f"Missing component {component}")
                    return False
            
            # Extract key information from the resume
            resume_info = self._extract_key_points(resume_text)
            
            # Validate against extracted information
            summary_lower = summary.lower()
            
            # Check if skills are mentioned
            skills_mentioned = any(
                skill.lower() in summary_lower 
                for skill in resume_info['skills']
            )
            
            # Check if companies are mentioned
            companies_mentioned = any(
                company.lower() in summary_lower 
                for company in resume_info['companies']
            )
            
            # Check if achievements are referenced
            achievements_mentioned = any(
                achievement.lower() in summary_lower 
                for achievement in resume_info['achievements']
            )
            
            # Check for name and email
            name = self._extract_name(resume_text)
            email = self._extract_email(resume_text)
            
            name_mentioned = name.lower() in summary_lower if name else True
            email_mentioned = email.lower() in summary_lower if email else True
            
            # Log validation results
            logger.info(f"Validation results:")
            logger.info(f"Skills mentioned: {skills_mentioned}")
            logger.info(f"Companies mentioned: {companies_mentioned}")
            logger.info(f"Achievements mentioned: {achievements_mentioned}")
            logger.info(f"Name mentioned: {name_mentioned}")
            logger.info(f"Email mentioned: {email_mentioned}")
            
            # Return True only if all validations pass
            return all([
                skills_mentioned,
                companies_mentioned,
                achievements_mentioned,
                name_mentioned,
                email_mentioned
            ])
            
        except Exception as e:
            logger.error(f"Error in summary validation: {str(e)}")
            return False

    def _clean_summary(self, summary: str) -> str:
        """Clean the generated summary."""
        if not summary:
            return ""
            
        # Remove any leading/trailing whitespace
        summary = summary.strip()
        
        # Remove any extra newlines
        summary = '\n'.join(line.strip() for line in summary.split('\n') if line.strip())
        
        return summary
