from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from .base_model import BaseResumeModel
from config.model_config import T5_CONFIG
from config.model_prompts import T5_PROMPT, SUMMARY_TEMPLATES

class T5ResumeModel(BaseResumeModel):
    def __init__(self, model_name="t5-base"):
        super().__init__()
        self.config.update(T5_CONFIG)
        if model_name != "t5-base":
            self.config['model']['name'] = model_name
        
        self.tokenizer = T5Tokenizer.from_pretrained(self.config['model']['name'])
        self.model = T5ForConditionalGeneration.from_pretrained(self.config['model']['name'])
    
    def generate_prompt(self, formatted_data):
        """Generate a prompt using the template strings."""
        template_data = self.format_template_data(formatted_data)
        templates = {k: v.format(**template_data) for k, v in SUMMARY_TEMPLATES.items()}
        return T5_PROMPT.format(**templates)
    
    def generate_summary(self, input_json):
        """Generate a professional summary using T5 model."""
        try:
            # Format input data
            formatted_data = self.format_input_data(input_json)
            
            # Generate prompt
            input_text = self.generate_prompt(formatted_data)
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                input_text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            )
            
            # Generate summary
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    **self.config['model']['generation_params']
                )
            
            # Decode and clean up the generated text
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self.clean_output(summary)
            
        except Exception as e:
            raise RuntimeError(f"Error in T5 generation: {e}")
