"""
GPT2-based resume summarization model.
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .base_model import BaseResumeModel
from config.model_config import GPT2_CONFIG
from config.model_prompts import GPT2_PROMPT, SUMMARY_TEMPLATES
from config.text_config import TEXT_CLEAN_CONFIG
import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

class GPT2ResumeModel(BaseResumeModel):
    def __init__(self, model_name="gpt2", device="cpu"):
        """Initialize GPT2 model with specified configuration."""
        logger.info("Initializing GPT2ResumeModel")
        super().__init__()
        self.model_name = model_name
        self.device = device
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            logger.info(f"Successfully loaded model and tokenizer from {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {e}")
            raise
        
        # Set pad token to eos token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        if device == "cuda":
            try:
                self.model = self.model.to(device)
            except RuntimeError:
                logger.error("CUDA not available, using CPU instead")
                self.device = "cpu"
            
        # Update generation params
        self.config = {
            'model': {
                'generation_params': {
                    'max_length': None,  # Let max_new_tokens control length
                    'max_new_tokens': 150,  # Control output length
                    'min_length': 50,  # Reduced to allow shorter summaries
                    'num_beams': 5,  # Beam search for better coherence
                    'length_penalty': 1.0,  # Neutral length penalty
                    'no_repeat_ngram_size': 3,  # Prevent repetition of 3-grams
                    'early_stopping': True,
                    'repetition_penalty': 1.2,  # Mild repetition penalty
                    'do_sample': True,  # Enable sampling
                    'temperature': 0.7,  # Lower temperature for more focused output
                    'top_p': 0.9,  # Nucleus sampling threshold
                    'top_k': 50,  # Top-k sampling
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'eos_token_id': self.tokenizer.eos_token_id
                }
            }
        }

    def clean_output(self, text, prompt):
        """Clean up the GPT2 model output."""
        logger.debug(f"Starting output cleaning. Input text:\n{text}")
        
        # Extract name and role from prompt
        name = "Unknown"
        role = ""
        company = ""
        experience = ""
        achievements = ""
        skills = ""
        
        # Parse prompt sections
        current_section = ""
        for line in prompt.split('\n'):
            line = line.strip()
            if line.startswith('Name:'):
                name = line.replace('Name:', '').strip()
            elif line.startswith('Current Role:'):
                role = line.replace('Current Role:', '').strip()
            elif line.startswith('Company:'):
                company = line.replace('Company:', '').strip()
            elif line.startswith('Experience:'):
                experience = line.replace('Experience:', '').strip()
            elif line == 'Key Achievements:':
                current_section = 'achievements'
            elif line == 'Core Skills:':
                current_section = 'skills'
            elif line and current_section == 'achievements':
                achievements = line.strip()
            elif line and current_section == 'skills':
                skills = line.strip()
        
        logger.debug(f"Extracted data from prompt: name='{name}', role='{role}', company='{company}', experience='{experience}'")
        
        # Remove the prompt from the output
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
            logger.debug(f"After removing prompt:\n{text}")
            
        # Split into lines and clean
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Log skipped lines for debugging
            should_skip = False
            skip_reason = ""
            
            # Skip example sections
            if any(x in line.lower() for x in ['example', 'hi, i am sarah', 'input:', 'output:']):
                should_skip = True
                skip_reason = "example section"
                
            # Skip lines starting with special characters or keywords
            elif line.startswith(('-', '*', '#', 'Example', 'Input:', 'Output:', 'Role:', 'Skills:', 'Now,', 'Generate', '(')):
                should_skip = True
                skip_reason = "special character/keyword start"
                
            # Skip lines containing analysis keywords
            elif any(x in line.lower() for x in [
                'analysis:', 'structure:', 'indicators:', 'competencies:', 
                'accomplishments:', 'results:', 'responsibilities:', 'write a',
                'professional summary', 'concise and impactful', 'example',
                'your summary', 'summary:', 'output:', 'click here', 'submit',
                'we look forward', 'contact us', 'please', 'thank you',
                'provide', 'share', 'explain', 'describe'
            ]):
                should_skip = True
                skip_reason = "analysis keyword"
                
            # Skip numeric lists and number sequences
            elif re.match(r'^\d+[\.\s\)]', line):
                should_skip = True
                skip_reason = "numeric list"
                
            # Skip lines that are too short or look like labels
            elif len(line.split()) < 3 or line.isupper():
                should_skip = True
                skip_reason = "too short/label"
                
            # Skip duplicate name mentions
            elif name in line and line.lower().startswith(name.lower()):
                should_skip = True
                skip_reason = "duplicate name"
                
            # Skip lines with wrong role/experience
            elif 'software engineer' in line.lower() or 'developer' in line.lower():
                should_skip = True
                skip_reason = "wrong role"
            
            if should_skip:
                logger.debug(f"Skipping line due to {skip_reason}: {line}")
            else:
                cleaned_lines.append(line)
                logger.debug(f"Keeping line: {line}")
            
        # Join lines and clean up
        text = ' '.join(cleaned_lines)
        logger.debug(f"After line cleaning:\n{text}")
        
        # Remove any remaining artifacts
        artifacts = [
            'Summary:', 'Based on the above analysis,', 'create a professional summary that:',
            'Name:', 'Current Role:', 'Company:', 'Experience:', 'Achievements:', 'Skills:',
            'Core Team Player', 'Team Leadership', 'Team Building',
            'Provide', 'Share', 'Explain', 'Describe'
        ]
        for artifact in artifacts:
            if artifact in text:
                logger.debug(f"Removing artifact: {artifact}")
                text = text.replace(artifact, '').strip()
        
        logger.debug(f"After artifact removal:\n{text}")
        
        # Ensure proper sentence structure
        sentences = text.split('.')
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Log skipped sentences for debugging
                should_skip = False
                skip_reason = ""
                
                # Skip sentences that are too short
                if len(sentence.split()) < 4:
                    should_skip = True
                    skip_reason = "too short"
                # Skip sentences that are duplicates or very similar
                elif any(self._similarity_score(sentence, s) > 0.7 for s in cleaned_sentences):
                    should_skip = True
                    skip_reason = "duplicate/similar"
                # Skip sentences that don't start with a capital letter
                elif not sentence[0].isupper():
                    should_skip = True
                    skip_reason = "no capital start"
                # Skip sentences with wrong role/experience
                elif 'software engineer' in sentence.lower() or 'developer' in sentence.lower():
                    should_skip = True
                    skip_reason = "wrong role"
                
                if should_skip:
                    logger.debug(f"Skipping sentence due to {skip_reason}: {sentence}")
                else:
                    # Add missing period
                    if not sentence.endswith('.'):
                        sentence += '.'
                    cleaned_sentences.append(sentence)
                    logger.debug(f"Keeping sentence: {sentence}")
        
        # If we don't have any sentences after cleaning, create a summary from the data we have
        if not cleaned_sentences:
            logger.debug("No valid sentences found, creating default summary")
            # Start with name introduction
            cleaned_sentences.append(f"Hi, I am {name}.")
            
            # Add role and experience
            intro = f"I am a {role} at {company}"
            if experience:
                intro += f" with {experience} experience"
            intro += "."
            cleaned_sentences.append(intro)
            
            if achievements:
                cleaned_sentences.append(f"My key achievements include {achievements}.")
                
            if skills:
                cleaned_sentences.append(f"My expertise includes {skills}.")
            
        text = ' '.join(cleaned_sentences)
        logger.debug(f"After sentence cleaning:\n{text}")
        
        # Remove any double periods and spaces
        text = text.replace('..', '.').replace('  ', ' ').strip()
        
        # Ensure it starts with name introduction
        if not text.startswith(f"Hi, I am {name}"):
            text = f"Hi, I am {name}. " + text
            
        # Limit to 3-4 sentences for conciseness
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 4:
            logger.debug(f"Truncating from {len(sentences)} to 4 sentences")
            sentences = sentences[:4]
        text = '. '.join(sentences) + '.'
        
        logger.debug(f"Final output:\n{text}")    
        return text

    def format_input_data(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        """Format input data for GPT2 model."""
        logger.debug(f"Raw input data: {input_json}")
        
        # Extract and format fields
        formatted_data = {
            'name': input_json.get('name', 'Unknown'),
            'current_role': input_json.get('current_role', ''),
            'years_experience': input_json.get('years_experience', '0'),
            'skills': ', '.join(input_json.get('skills', [])) if isinstance(input_json.get('skills'), list) else input_json.get('skills', ''),
            'achievements': input_json.get('achievements', ''),
        }
        
        # Handle companies field
        companies = input_json.get('companies', [])
        if isinstance(companies, list):
            formatted_data['companies'] = companies[0] if companies else ''
        else:
            formatted_data['companies'] = companies.split(',')[0].strip() if companies else ''
            
        # Log formatted data
        logger.debug(f"Formatted data: {formatted_data}")
        
        return formatted_data

    def generate_prompt(self, input_data: Dict[str, Any]) -> str:
        """Generate a prompt for GPT2 model."""
        prompt = GPT2_PROMPT.format(
            name=input_data.get('name', 'Unknown'),
            role=input_data.get('current_role', ''),
            company=input_data.get('companies', ''),
            experience=input_data.get('years_experience', '0'),
            achievements=input_data.get('achievements', 'No achievements provided.'),
            skills=input_data.get('skills', 'No skills provided.')
        )
        
        logger.debug(f"Generated prompt:\n{prompt}")
        return prompt

    def generate_summary(self, input_json: Dict[str, Any]) -> str:
        """Generate a summary using GPT2 model."""
        logger.info("Starting summary generation")
        logger.debug(f"Input data: {input_json}")
        
        try:
            # Format input data
            input_data = self.format_input_data(input_json)
            logger.debug(f"Formatted input data: {input_data}")
            
            # Generate prompt
            prompt = self.generate_prompt(input_data)
            logger.debug(f"Generated prompt:\n{prompt}")
            
            # Encode prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            if self.device == "cuda":
                inputs = inputs.to(self.device)
            
            # Generate text
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **self.config['model']['generation_params']
            )
            
            # Decode output
            raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.debug(f"Raw model output:\n{raw_output}")
            
            # Clean output
            cleaned_output = self.clean_output(raw_output, prompt)
            logger.debug(f"Cleaned output:\n{cleaned_output}")
            
            return cleaned_output
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise

    def _similarity_score(self, s1: str, s2: str) -> float:
        """Calculate similarity score between two sentences."""
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0
