# models/gpt2_model.py
"""
GPT-2 model implementation for resume summary generation.
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .base_model import BaseResumeModel
from config.model_config import GPT2_CONFIG
from config.model_prompts import GPT2_PROMPT, SUMMARY_TEMPLATES
from config.app_config import SUMMARY_CONFIG
from config.cache_config import CACHE_CONFIG

class GPT2ResumeModel(BaseResumeModel):
    def __init__(self, model_name="gpt2-medium"):
        super().__init__()
        self.config.update(GPT2_CONFIG)
        self.model_name = model_name
        
        # Get cache directory from config
        self.cache_dir = CACHE_CONFIG['cache']['location']['base_dir']
        
        # Initialize tokenizer and model with custom cache location
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=False  # Allow downloading if not in cache
        )
        # Set pad token to eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = GPT2LMHeadModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=False  # Allow downloading if not in cache
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
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            
            # Generate summary
            #outputs = self.model.generate(
            #    inputs,
            #    **self.get_generation_config()
            #)

            # Pass attention mask and input IDs to the model
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # Generate summary
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,  # Pass the attention mask
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
            max_sentences = SUMMARY_CONFIG['output']['max_sentences']
            
            for sentence in sentences:
                # Skip sentences that look like they're starting new topics
                if any(phrase in sentence for phrase in SUMMARY_CONFIG['filtering']['stop_phrases']):
                    break
                    
                # Skip sentences that seem to be about membership or additional roles
                if any(word in sentence.lower() for word in SUMMARY_CONFIG['filtering']['membership_words']) and len(relevant_sentences) > 0:  # Allow education in first sentence
                    continue
                
                relevant_sentences.append(sentence)
                if len(relevant_sentences) >= max_sentences:
                    break
            
            # Join sentences and clean up
            summary = ". ".join(relevant_sentences).strip()
            if not summary.endswith("."):
                summary = summary + "."
                
            # Fix common formatting issues
            for old, new in SUMMARY_CONFIG['formatting']['cleanup_replacements'].items():
                summary = summary.replace(old, new)
            
            return summary
            
        except Exception as e:
            raise RuntimeError(f"Error in GPT2 generation: {str(e)}")
            
    def get_generation_config(self):
        """Get the generation configuration."""
        return self.config['model']['generation_params']
