# model/bart_model.py
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from .base_model import BaseResumeModel
from config.model_config import BART_CONFIG
from config.model_prompts import BART_PROMPT, SUMMARY_TEMPLATES
from config.cache_config import CACHE_CONFIG

class BartResumeModel(BaseResumeModel):
    def __init__(self, model_name="facebook/bart-base"):
        super().__init__()
        self.config.update(BART_CONFIG)
        if model_name != "facebook/bart-base":
            self.config['model']['name'] = model_name
        
        # Get cache directory from config
        self.cache_dir = CACHE_CONFIG['cache']['location']['base_dir']
        
        # Initialize tokenizer and model with caching
        self.tokenizer = BartTokenizer.from_pretrained(
            self.config['model']['name'],
            force_download=False,
            cache_dir=self.cache_dir,
            local_files_only=False
        )
        
        self.model = BartForConditionalGeneration.from_pretrained(
            self.config['model']['name'],
            force_download=False,
            cache_dir=self.cache_dir,
            local_files_only=False
        )
    
    def generate_prompt(self, formatted_data):
        """Generate a prompt using the template strings."""
        template_data = self.format_template_data(formatted_data)
        templates = {k: v.format(**template_data) for k, v in SUMMARY_TEMPLATES.items()}
        prompt = BART_PROMPT.format(**templates)
        return prompt.replace('\n', ' ').strip()
    
    def generate_summary(self, input_json):
        """Generate a professional summary using BART model."""
        try:
            # Format input data
            formatted_data = self.format_input_data(input_json)
            
            # Generate prompt
            input_text = self.generate_prompt(formatted_data)
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Generate summary
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=self.config['model']['generation_params']['max_length'],
                    min_length=self.config['model']['generation_params']['min_length'],
                    num_beams=self.config['model']['generation_params']['num_beams'],
                    length_penalty=self.config['model']['generation_params']['length_penalty'],
                    no_repeat_ngram_size=self.config['model']['generation_params']['no_repeat_ngram_size'],
                    early_stopping=self.config['model']['generation_params']['early_stopping'],
                    repetition_penalty=self.config['model']['generation_params']['repetition_penalty'],
                    do_sample=self.config['model']['generation_params']['do_sample']
                )
            
            # Decode and clean up the generated text
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the output if it appears
            if input_text in summary:
                summary = summary.replace(input_text, "").strip()
            
            return self.clean_output(summary)
            
        except Exception as e:
            raise RuntimeError(f"Error in BART generation: {e}")
