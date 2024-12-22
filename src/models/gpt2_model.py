"""GPT-2 model for resume summary generation."""
import logging
from typing import Dict, Any, List, Optional

from transformers import GPT2LMHeadModel, GPT2Tokenizer

logger = logging.getLogger(__name__)


class GPT2Model:
    """GPT-2 model for generating resume summaries."""
    
    def __init__(self):
        """Initialize GPT-2 model."""
        logger.info("Initializing GPT-2 model")
        
        try:
            # Load model and tokenizer
            self.model_name = "gpt2"
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            
            # Set generation parameters
            self.max_length = 200
            self.min_length = 100
            self.num_beams = 4
            self.temperature = 0.7
            self.top_p = 0.9
            self.top_k = 50
            
            logger.info("Successfully initialized GPT-2 model")
            
        except Exception as e:
            logger.error(f"Error initializing GPT-2 model: {e}")
            raise
    
    def generate_summary(self, resume_data: Dict[str, Any]) -> str:
        """Generate a summary from resume data.
        
        Args:
            resume_data: Dictionary containing parsed resume data
            
        Returns:
            Generated summary text
        """
        try:
            # Build prompt
            prompt = self._build_prompt(resume_data)
            
            # Generate summary
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                min_length=self.min_length,
                num_beams=self.num_beams,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            
            # Decode summary
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean and validate summary
            summary = self._clean_summary(summary)
            if not self._validate_summary(summary):
                summary = self._generate_fallback_summary(resume_data)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise
    
    def _build_prompt(self, resume_data: Dict[str, Any]) -> str:
        """Build a prompt for the model from resume data."""
        try:
            name = resume_data.get('name', '')
            current_role = resume_data.get('current_role', '')
            years_exp = resume_data.get('years_experience', 0)
            companies = resume_data.get('companies', [])
            skills = resume_data.get('skills', [])
            achievements = resume_data.get('achievements', [])
            
            prompt = f"Generate a professional summary for {name}, "
            prompt += f"a {current_role} with {years_exp} years of experience. "
            
            if companies:
                prompt += f"Has worked at {', '.join(companies)}. "
            
            if skills:
                prompt += f"Key skills include {', '.join(skills)}. "
            
            if achievements:
                prompt += f"Notable achievements: {'; '.join(achievements)}. "
            
            prompt += "Write a concise, professional summary highlighting their experience and achievements:"
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error building prompt: {e}")
            raise
    
    def _clean_summary(self, summary: str) -> str:
        """Clean and format the generated summary."""
        try:
            # Remove any prompt text that might have been copied
            if ":" in summary:
                summary = summary.split(":")[-1]
            
            # Clean whitespace
            summary = " ".join(summary.split())
            
            # Remove any trailing punctuation
            summary = summary.rstrip(".,")
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning summary: {e}")
            raise
    
    def _validate_summary(self, summary: str) -> bool:
        """Validate the generated summary."""
        try:
            # Check length
            if len(summary.split()) < 50 or len(summary.split()) > 200:
                return False
            
            # Check for incomplete sentences
            if not summary[-1] in ".!?":
                return False
            
            # Check for repetition
            words = summary.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Only check non-trivial words
                    word_freq[word] = word_freq.get(word, 0) + 1
                    if word_freq[word] > 3:  # Word appears too many times
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating summary: {e}")
            raise
    
    def _generate_fallback_summary(self, resume_data: Dict[str, Any]) -> str:
        """Generate a fallback summary when the main generation fails."""
        try:
            name = resume_data.get('name', 'The candidate')
            current_role = resume_data.get('current_role', 'professional')
            years_exp = resume_data.get('years_experience', 0)
            
            summary = f"{name} is a {current_role} with {years_exp} years of experience. "
            
            skills = resume_data.get('skills', [])
            if skills:
                summary += f"Their expertise includes {', '.join(skills[:5])}. "
            
            achievements = resume_data.get('achievements', [])
            if achievements:
                summary += f"Key achievements include {achievements[0]}"
                if len(achievements) > 1:
                    summary += f" and {achievements[1]}"
                summary += ". "
            
            companies = resume_data.get('companies', [])
            if companies:
                summary += f"Has valuable experience working at {', '.join(companies)}."
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error generating fallback summary: {e}")
            raise
