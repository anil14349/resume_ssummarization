"""
GPT-2 model implementation for resume summary generation.
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .base_model import BaseResumeModel
from config.model_config import GPT2_CONFIG
from config.model_prompts import GPT2_PROMPT, SUMMARY_TEMPLATES

class GPT2ResumeModel(BaseResumeModel):
    def __init__(self, model_name="gpt2-medium", cache_dir=None):
        super().__init__()
        self.config.update(GPT2_CONFIG)
        self.model_name = model_name
        
        # Initialize tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            local_files_only=cache_dir is not None
        )
        
        self.model = GPT2LMHeadModel.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            local_files_only=cache_dir is not None
        )
        
        # Set pad token ID
        self.model.config.pad_token_id = self.model.config.eos_token_id
        
    def generate_prompt(self, formatted_data):
        """Generate a prompt using the template strings."""
        templates = {k: v.format(**formatted_data) for k, v in SUMMARY_TEMPLATES.items()}
        return GPT2_PROMPT.format(**templates)
        
    def generate_summary(self, input_data):
        """Generate a summary using GPT-2."""
        try:
            # Format input data
            formatted_data = self.format_input_data(input_data)
            
            # Generate prompt
            prompt = self.generate_prompt(formatted_data)
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Generate summary
            outputs = self.model.generate(
                inputs,
                **self.get_generation_config()
            )
            
            # Decode and clean up the generated text
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Keep only the core summary information
            core_info = [
                "Hi, I am",
                "I am a",
                "I am skilled in",
                "I hold",
                "My key achievements include"
            ]
            
            # Find the first occurrence of core information
            start_idx = -1
            for phrase in core_info:
                idx = summary.find(phrase)
                if idx != -1 and (start_idx == -1 or idx < start_idx):
                    start_idx = idx
            
            if start_idx != -1:
                summary = summary[start_idx:]
            
            # Split into sentences and keep only the relevant ones
            sentences = summary.split(". ")
            relevant_sentences = []
            max_sentences = 5  # Limit to 5 sentences
            
            for sentence in sentences:
                # Skip sentences that look like they're starting new topics
                if any(phrase in sentence for phrase in [
                    "Please feel",
                    "For more information",
                    "Contact me",
                    "I am currently",
                    "I would like",
                    "In my spare time",
                    "References",
                    "She is",
                    "She has",
                    "He is",
                    "He has",
                    "In addition",
                    "Furthermore",
                    "Moreover",
                    "Also",
                    "Additionally"
                ]):
                    break
                    
                # Skip sentences that seem to be about membership or additional roles
                if any(word in sentence.lower() for word in [
                    "member of",
                    "board of",
                    "graduate of",
                    "professor",
                    "taught",
                    "served",
                    "university",
                    "college",
                    "school"
                ]) and len(relevant_sentences) > 0:  # Allow education in first sentence
                    continue
                
                relevant_sentences.append(sentence)
                if len(relevant_sentences) >= max_sentences:
                    break
            
            # Join sentences and clean up
            summary = ". ".join(relevant_sentences).strip()
            if not summary.endswith("."):
                summary = summary + "."
                
            # Fix common formatting issues
            summary = summary.replace("..", ".")
            summary = summary.replace("I'mplement", "implement")
            summary = summary.replace("['", "")
            summary = summary.replace("']", "")
            summary = summary.replace("',", ",")
            summary = summary.replace(" '", " ")
            
            return summary
            
        except Exception as e:
            raise RuntimeError(f"Error in GPT2 generation: {str(e)}")
            
    def get_generation_config(self):
        """Get the generation configuration."""
        return self.config['model']['generation_params']
