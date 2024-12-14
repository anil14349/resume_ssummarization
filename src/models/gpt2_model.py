from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from .base_model import BaseResumeModel
from config.model_config import GPT2_CONFIG
from config.model_prompts import GPT2_PROMPT, SUMMARY_TEMPLATES

class GPT2ResumeModel(BaseResumeModel):
    def __init__(self, model_name="gpt2-medium"):
        super().__init__()
        self.config.update(GPT2_CONFIG)
        if model_name != "gpt2-medium":
            self.config['model']['name'] = model_name
        
        # Initialize tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.config['model']['name'])
        self.model = GPT2LMHeadModel.from_pretrained(self.config['model']['name'])
        
        # GPT2 doesn't have a padding token by default
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config['model']['generation_params']['pad_token_id'] = self.model.config.eos_token_id
        
    def generate_prompt(self, formatted_data):
        """Generate a prompt using the template strings."""
        template_data = self.format_template_data(formatted_data)
        templates = {k: v.format(**template_data) for k, v in SUMMARY_TEMPLATES.items()}
        return GPT2_PROMPT.format(**templates)
    
    def generate_summary(self, input_json):
        """Generate a professional summary using GPT-2 model."""
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
            raise RuntimeError(f"Error in GPT2 generation: {e}")

    def clean_output(self, summary):
        """Clean the generated text."""
        summary = super().clean_output(summary)
        
        # Remove the prompt text
        if "Generate a professional summary:" in summary:
            summary = summary.split("Generate a professional summary:")[-1].strip()
        
        # Remove any additional content
        cutoff_phrases = [
            "I have worked in",
            "I would like to",
            "Thank you",
            "Sincerely",
            "I look forward",
            "I will be",
            "My ideal",
            "I am currently"
        ]
        
        for phrase in cutoff_phrases:
            if phrase in summary:
                summary = summary.split(phrase)[0]
        
        # Fix formatting issues
        summary = summary.replace("My is", "My name is")
        summary = summary.replace("Im", "I'm")
        summary = summary.replace("im", "I'm")
        summary = summary.replace("tI'me", "time")
        summary = summary.replace("POSition", "position")
        summary = summary.replace("  ", " ")
        
        # Fix role repetition
        summary = summary.replace("Restaurant Manager at Restaurant Manager", "Restaurant Manager")
        
        # Clean up
        summary = summary.strip()
        if not summary.endswith("."):
            summary = summary.rstrip(",.!? ") + "."
            
        return summary
