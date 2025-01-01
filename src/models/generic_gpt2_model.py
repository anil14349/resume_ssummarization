from typing import Dict, Any
import logging
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from .base_model import BaseModel
import re
import torch

logger = logging.getLogger(__name__)

class GenericGPT2Model(BaseModel):
    """A GPT-Neo model that can generate video scripts from resume data."""
    
    def __init__(self):
        """Initialize the model."""
        super().__init__()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Use base GPT2 for more stable generation
            logger.info("Loading model and tokenizer...")
            model_name = "gpt2"  # Base GPT2 model
            
            # Determine device (GPU/CPU)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load tokenizer and model with caching
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=".model_cache")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=".model_cache")
            
            # Move model to appropriate device
            self.model = self.model.to(device)
            
            # Initialize the generator pipeline
            self.generator = pipeline(
                'text-generation',
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device == "cuda" else -1,
                max_length=800,    # Balanced length
                min_length=300,    # Ensure substantial content
                num_return_sequences=1,
                temperature=0.7,   # Balanced creativity
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
            
            # Set generation parameters
            self.max_length = 800
            self.min_length = 300
            self.num_return_sequences = 1
            self.temperature = 0.7
            self.top_p = 0.9
            self.top_k = 50
            self.repetition_penalty = 1.2
            
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
            
    def _create_section_prompt(self, section_num: int, title: str) -> str:
        """Create a prompt for a specific section."""
        return f"{section_num}. {title}\n- Caption: [Title for {title}]\n- Audio: [Script for {title}]\n- Visual: [Visuals for {title}]\n\n"
            
    def generate_summary(self, resume_data: Dict[str, Any]) -> str:
        """Generate a video script summary from resume data."""
        try:
            name = resume_data.get('name', '')
            current_role = resume_data.get('current_role', '')
            years = resume_data.get('years_experience', 0)
            companies = resume_data.get('companies', [])
            skills = resume_data.get('skills', [])
            achievement_text = resume_data.get('achievements', [''])[0] if resume_data.get('achievements') else ''
            education_str = resume_data.get('education', '')
            email = resume_data.get('email', '')
            
            # Detect if this is a technical resume based on skills
            tech_skills = {'python', 'java', 'javascript', 'react', 'node', 'aws', 'docker', 'kubernetes', 
                         'git', 'ci/cd', 'sql', 'mongodb', 'jenkins', 'jira', 'terraform'}
            is_technical = any(skill.lower() in tech_skills for skill in skills)
            
            logger.info(f"Resume type detected: {'Technical' if is_technical else 'Non-technical'}")
            
            # Create appropriate prompt based on resume type
            if is_technical:
                prompt = (
                    "Create a professional video script using ONLY the information provided below. Do not add any information that is not in the resume data.\n\n"
                    "Here is an example format for a technical professional (DO NOT copy the content, only the structure):\n\n"
                    "1. Introduction\n"
                    "- Caption: Senior Software Engineer\n"
                    "- Audio: Meet Alex, a Software Engineer with 8 years of experience in full-stack development and cloud architecture.\n"
                    "- Visual: Professional headshot transitioning to coding environment showing live development.\n\n"
                    "2. Experience\n"
                    "- Caption: Technical Excellence\n"
                    "- Audio: At leading tech companies, Alex has architected scalable solutions and led development teams to success.\n"
                    "- Visual: Split screen showing code deployment, system architecture diagrams, and team collaboration.\n\n"
                    "3. Skills\n"
                    "- Caption: Technical Expertise\n"
                    "- Audio: Mastery in Python, Java, and cloud technologies, with proven experience in system design and optimization.\n"
                    "- Visual: Animated tech stack showcase with programming languages, frameworks, and tools.\n\n"
                    "4. Achievement\n"
                    "- Caption: Impact & Innovation\n"
                    "- Audio: Reduced system latency by 40% through innovative architecture redesign.\n"
                    "- Visual: Performance metrics dashboard showing before/after improvements.\n\n"
                    "5. Goals\n"
                    "- Caption: Future Vision\n"
                    "- Audio: Passionate about leveraging technology to solve complex problems and mentor future engineers.\n"
                    "- Visual: Modern development environment with emerging technology icons.\n\n"
                    "6. Contact\n"
                    "- Caption: Let's Connect\n"
                    "- Audio: Ready to bring technical expertise to your next challenging project.\n"
                    "- Visual: Professional contact details with tech-themed background.\n\n"
                )
            else:
                prompt = (
                    "Create a professional video script using ONLY the information provided below. Do not add any information that is not in the resume data.\n\n"
                    "Here is an example format for a restaurant manager (DO NOT copy the content, only the structure):\n\n"
                    "1. Introduction\n"
                    "- Caption: Seasoned Restaurant Professional\n"
                    "- Audio: Meet John, a Restaurant Manager with 15 years of experience in food service operations.\n"
                    "- Visual: Professional footage showing manager interacting with staff and customers.\n\n"
                    "2. Experience\n"
                    "- Caption: Restaurant Management Excellence\n"
                    "- Audio: At top dining establishments, led teams and improved customer satisfaction.\n"
                    "- Visual: Dynamic montage of restaurant operations and team training.\n\n"
                    "3. Skills\n"
                    "- Caption: Operational Expertise\n"
                    "- Audio: Core competencies include staff training and inventory management.\n"
                    "- Visual: Animated showcase of key skills with icons.\n\n"
                    "4. Achievement\n"
                    "- Caption: Performance & Growth\n"
                    "- Audio: Increased revenue through strategic initiatives.\n"
                    "- Visual: Performance dashboard showing key metrics.\n\n"
                    "5. Goals\n"
                    "- Caption: Vision for Excellence\n"
                    "- Audio: Committed to operational excellence and team development.\n"
                    "- Visual: Forward-looking imagery of modern operations.\n\n"
                    "6. Contact\n"
                    "- Caption: Let's Connect\n"
                    "- Audio: Ready to bring expertise to your establishment.\n"
                    "- Visual: Professional contact details with themed background.\n\n"
                )
            
            # Add resume data and instructions
            prompt += (
                "RESUME DATA (Use ONLY this information):\n"
                "------------------------\n"
                f"Full Name: {name}\n"
                f"Current Position: {current_role}\n"
                f"Years of Experience: {years}\n"
                f"Core Skills: {', '.join(skills[:5])}\n"
                f"Companies Worked At: {', '.join(companies[:2])}\n"
                f"Key Achievement: {achievement_text}\n"
                f"Education: {education_str}\n"
                "------------------------\n\n"
                "INSTRUCTIONS:\n"
                "1. Create exactly 6 sections: Introduction, Experience, Skills, Achievement, Goals, Contact\n"
                "2. Each section must start with a number (1. Introduction, etc.)\n"
                "3. Each section must have Caption, Audio, and Visual with dashes\n"
                "4. Use ONLY the information provided above\n"
                "5. Keep the tone professional and appropriate for the role\n"
                "6. Highlight relevant achievements and expertise\n"
                "7. Focus on core competencies and impact\n"
                "8. Use relevant metrics and visuals\n"
                "9. Maintain a focus on professional excellence\n\n"
                "Begin the script here:\n\n"
                "1. Introduction\n"
            )
            
            logger.info("Generating text with GPT-Neo...")
            logger.info(f"Prompt length: {len(prompt)} characters")
            logger.info("Starting generation pipeline...")
            
            def validate_and_clean_script(script: str) -> str:
                """Validate and clean the generated script."""
                # Find the end of the script (last section)
                end_markers = ["==", "RESUME DATA", "INSTRUCTIONS", "Overview", "..."]
                script_end = len(script)
                for marker in end_markers:
                    pos = script.find(marker)
                    if pos != -1 and pos < script_end:
                        script_end = pos
                
                # Extract just the script content
                script = script[:script_end].strip()
                
                # Ensure all sections are present
                required_sections = ["1. Introduction", "2. Experience", "3. Skills", 
                                  "4. Achievement", "5. Goals", "6. Contact"]
                if not all(section in script for section in required_sections):
                    logger.warning("Missing required sections")
                    return None
                
                # Ensure each section has required elements
                required_elements = ["- Caption:", "- Audio:", "- Visual:"]
                section_count = 0
                for section in required_sections:
                    start = script.find(section)
                    if start == -1:
                        continue
                    next_section = section_count + 2 if section_count < 5 else len(script)
                    next_pos = script.find(f"{next_section}. ", start) if next_section <= 6 else len(script)
                    section_content = script[start:next_pos] if next_pos != -1 else script[start:]
                    
                    if not all(element in section_content for element in required_elements):
                        logger.warning(f"Missing required elements in section {section}")
                        return None
                    
                    if len(section_content.strip()) < 50:
                        logger.warning(f"Section {section} content too short")
                        return None
                    
                    section_count += 1
                
                return script
            
            # Generate text with more conservative parameters
            outputs = self.generator(
                prompt,
                max_length=1024,
                min_length=400,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.85,
                top_k=40,
                repetition_penalty=1.5,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                no_repeat_ngram_size=3
            )
            
            logger.info("Generation completed")
            generated_text = outputs[0]['generated_text']
            logger.info(f"Generated text length: {len(generated_text)} characters")
            
            # Extract and validate the script
            script_start = generated_text.find("1. Introduction")
            if script_start == -1:
                logger.warning("Could not find script start marker")
                return self._get_default_script(name, email, resume_data)
            
            script = generated_text[script_start:]
            logger.info(f"Extracted script length: {len(script)} characters")
            
            # Validate and clean the script
            script = validate_and_clean_script(script)
            if script is None:
                return self._get_default_script(name, email, resume_data)
            
            # Post-process the script with role awareness
            def post_process_script(script: str, achievement: str, role: str) -> str:
                """Post-process the script to ensure accuracy and proper formatting."""
                # Ensure achievement is accurately represented
                achievement_section_start = script.find("4. Achievement")
                if achievement_section_start != -1:
                    achievement_section_end = script.find("5. Goals", achievement_section_start)
                    if achievement_section_end == -1:
                        achievement_section_end = len(script)
                    
                    # Create appropriate visual description based on role type
                    role_lower = role.lower()
                    
                    if any(industry_term in role_lower for industry_term in ['operations', 'manager', 'supervisor', 'hospitality']):
                        visual_desc = "Dynamic performance dashboard showing operational metrics, team efficiency, and customer satisfaction scores"
                    elif any(hr_term in role_lower for hr_term in ['hr', 'human resources', 'talent', 'recruiting']):
                        visual_desc = "Interactive dashboard showing employee engagement metrics, retention rates, and program success indicators"
                    elif any(tech_term in role_lower for tech_term in ['engineer', 'developer', 'architect', 'technical']):
                        visual_desc = "Data visualization showing technical metrics and system improvements"
                    elif any(creative_term in role_lower for creative_term in ['designer', 'artist', 'creative', 'marketing']):
                        visual_desc = "Portfolio showcase highlighting creative work and campaign results"
                    elif any(business_term in role_lower for business_term in ['business', 'sales', 'finance', 'analyst']):
                        visual_desc = "Business performance dashboard with key growth metrics and market indicators"
                    else:
                        visual_desc = "Professional achievement visualization with key metrics and results"
                    
                    # Create role-specific caption
                    if any(industry_term in role_lower for industry_term in ['operations', 'manager', 'supervisor']):
                        caption = "Operational Excellence & Leadership"
                    elif 'hr' in role_lower or 'human resources' in role_lower:
                        caption = "HR Excellence & Impact"
                    elif 'manager' in role_lower or 'director' in role_lower:
                        caption = "Leadership Achievement"
                    else:
                        caption = "Key Professional Achievement"
                    
                    # Format achievement text with proper capitalization
                    if achievement and not achievement[0].isupper():
                        achievement = achievement[0].upper() + achievement[1:]
                    
                    before_achievement = script[:achievement_section_start]
                    after_achievement = script[achievement_section_end:]
                    
                    achievement_section = (
                        "4. Achievement\n"
                        f"- Caption: {caption}\n"
                        f"- Audio: {achievement}\n"
                        f"- Visual: {visual_desc}\n\n"
                    )
                    script = before_achievement + achievement_section + after_achievement
                
                return script
            
            script = post_process_script(script, achievement_text, current_role)
            
            return script
            
        except Exception as e:
            logger.error(f"Error in generate_summary: {str(e)}")
            return self._get_default_script(name, email, resume_data)
            
    def _get_default_script(self, name: str, email: str, resume_data: Dict[str, Any]) -> str:
        """Get a default script when generation fails, utilizing resume data effectively."""
        logger.info("Falling back to default script generation")
        
        # Extract additional information
        current_role = resume_data.get('current_role', '')
        years = resume_data.get('years_experience', 0)
        companies = resume_data.get('companies', [])
        skills = resume_data.get('skills', [])
        achievements = resume_data.get('achievements', [])
        education = resume_data.get('education', [])
        
        # Format education
        education_str = ''
        if education and isinstance(education[0], dict):
            degree = education[0].get('degree', '')
            institution = education[0].get('institution', '')
            if degree and institution:
                education_str = f"{degree} from {institution}"
        
        # Format skills by category
        technical_skills = [s for s in skills if any(tech in s.lower() for tech in ['python', 'java', 'cloud', 'aws', 'ml', 'ai'])]
        business_skills = [s for s in skills if any(biz in s.lower() for biz in ['management', 'leadership', 'strategy', 'analysis'])]
        
        # Format achievements with metrics
        achievement_str = ''
        if achievements:
            achievement = achievements[0]
            # Extract metrics using regex
            metrics = re.findall(r'(\d+(?:\.\d+)?%|\$\d+(?:,\d+)*(?:\.\d+)?|\d+(?:,\d+)*)', achievement)
            if metrics:
                achievement_str = f"achieved {metrics[0]} improvement through {achievement.split('through')[-1].strip()}" if 'through' in achievement else f"delivered {metrics[0]} in results through strategic initiatives"
            else:
                achievement_str = achievement
        
        return f"""1. Introduction
- Caption: {name} | {current_role}
- Audio: Hello, I'm {name}, a {current_role} with {years} years of experience in {', '.join(technical_skills[:2])}. {education_str} has equipped me with a strong foundation in {', '.join(business_skills[:2])}.
- Visual: Professional headshot transitioning to a dynamic showcase of {current_role} responsibilities

2. Experience
- Caption: {years} Years of Industry Excellence
- Audio: Throughout my career at {' and '.join(companies)}, I've consistently delivered impactful solutions. My expertise spans {', '.join(technical_skills)} with a focus on {technical_skills[0] if technical_skills else 'technical excellence'}.
- Visual: Animated timeline highlighting key roles and companies, with emphasis on growth trajectory

3. Skills
- Caption: Technical Mastery & Leadership
- Audio: My technical expertise in {', '.join(technical_skills[:3])} is complemented by strong {', '.join(business_skills[:2])}. This unique combination enables me to bridge technical solutions with business objectives.
- Visual: Interactive skill matrix showing technical and business competencies with proficiency levels

4. Achievement
- Caption: Driving Transformative Results
- Audio: {achievement_str}
- Visual: Data visualization showcasing the impact metrics with supporting graphics

5. Goals
- Caption: Future Vision & Innovation
- Audio: Looking ahead, I'm passionate about leveraging {technical_skills[0] if technical_skills else 'technology'} to drive innovation. My goal is to lead transformative projects that combine {' and '.join(technical_skills[:2] + business_skills[:1])}.
- Visual: Forward-looking imagery representing innovation and growth in {current_role}

6. Contact
- Caption: Let's Connect
- Audio: I'm excited to discuss how my expertise in {technical_skills[0] if technical_skills else 'technology'} and {business_skills[0] if business_skills else 'business'} can add value to your organization. Reach out at {email}.
- Visual: Professional contact display with animated social media links and QR code"""

    def _post_process_script(self, script: str, name: str, email: str) -> str:
        """Clean and format the generated script."""
        try:
            # Split into sections
            sections = script.split('\n\n')
            cleaned_sections = []
            
            for section in sections:
                if not section.strip():
                    continue
                    
                # Clean the section
                cleaned_section = self._clean_section(section, name, email)
                if cleaned_section:
                    cleaned_sections.append(cleaned_section)
                    
            # Join sections back together
            return '\n\n'.join(cleaned_sections)
            
        except Exception as e:
            logger.error(f"Error in post-processing: {str(e)}")
            return script
            
    def _clean_section(self, section: str, name: str, email: str) -> str:
        """Clean an individual section."""
        try:
            # Split into lines
            lines = section.strip().split('\n')
            if not lines:
                return ""
                
            # Get section number and title
            if not lines[0][0].isdigit():
                return ""
            section_num = lines[0].split('.')[0]
            
            # Process components
            components = {}
            current_component = None
            
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                    
                # Check for component marker
                if line.startswith('- '):
                    component_type = line[2:].split(':')[0].lower()
                    component_text = ':'.join(line[2:].split(':')[1:]).strip()
                    components[component_type] = component_text
                    
            # Clean components
            components = self._clean_components(components, section_num, name, email)
            
            # Format section
            formatted_lines = [
                f"{section_num}. {self._get_section_title(section_num)}",
                f"- Caption: {components.get('caption', self._get_default_caption(section_num, name))}",
                f"- Audio: {components.get('audio', self._get_default_audio(section_num, name, email))}",
                f"- Visual: {components.get('visual', self._get_default_visual(section_num))}"
            ]
            
            return '\n'.join(formatted_lines)
            
        except Exception as e:
            logger.error(f"Error cleaning section: {str(e)}")
            return section
            
    def _get_section_title(self, section_num: str) -> str:
        """Get the title for a section."""
        titles = {
            '1': 'Introduction',
            '2': 'Experience',
            '3': 'Skills',
            '4': 'Achievement',
            '5': 'Goals',
            '6': 'Contact'
        }
        return titles.get(section_num, 'Section')
        
    def _clean_components(self, components: Dict[str, str], section_num: str, name: str, email: str) -> Dict[str, str]:
        """Clean and validate section components."""
        try:
            section_num = int(section_num)
            cleaned = {}
            
            # Clean caption
            if 'caption' in components:
                caption = components['caption'].strip()
                if len(caption) < 5:  # Too short, use default
                    caption = self._get_default_caption(section_num, name)
                cleaned['caption'] = caption
                
            # Clean audio
            if 'audio' in components:
                audio = components['audio'].strip()
                if len(audio) < 10:  # Too short, use default
                    audio = self._get_default_audio(section_num, name, email)
                # Ensure first-person perspective
                if not any(pronoun in audio.lower() for pronoun in ['i ', "i'm", "my", "me", "we"]):
                    audio = f"I {audio}" if not audio.lower().startswith('i ') else audio
                cleaned['audio'] = audio
                
            # Clean visual
            if 'visual' in components:
                visual = components['visual'].strip()
                if len(visual) < 5:  # Too short, use default
                    visual = self._get_default_visual(section_num)
                cleaned['visual'] = visual
                
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning components: {str(e)}")
            return components
            
    def _get_default_caption(self, section_num: int, name: str) -> str:
        """Get default caption for a section."""
        captions = {
            1: f"{name} | Professional Overview",
            2: "Proven Track Record",
            3: "Expert Skill Set",
            4: "Key Achievement Spotlight",
            5: "Vision & Aspirations",
            6: "Let's Connect"
        }
        return captions.get(section_num, "Professional Profile")
        
    def _get_default_audio(self, section_num: int, name: str, email: str) -> str:
        """Get default audio for a section."""
        audio = {
            1: f"Hello, I'm {name}. I bring expertise and innovation to every project I undertake.",
            2: "My career journey has been marked by continuous growth and impactful contributions.",
            3: "I've developed a diverse skill set that enables me to tackle complex challenges effectively.",
            4: "One of my proudest achievements demonstrates my ability to drive results.",
            5: "Looking ahead, I'm excited to take on new challenges and contribute to innovative projects.",
            6: f"I'm always open to discussing new opportunities. Feel free to reach out at {email}."
        }
        return audio.get(section_num, "")
        
    def _get_default_visual(self, section_num: int) -> str:
        """Get default visual for a section."""
        visuals = {
            1: "Professional headshot with modern office background",
            2: "Animated timeline showcasing career progression",
            3: "Interactive 3D visualization of interconnected skills",
            4: "Dynamic infographic highlighting key achievements",
            5: "Inspiring imagery of innovation and growth",
            6: "Clean, modern contact information display with social media icons"
        }
        return visuals.get(section_num, "Professional imagery")
