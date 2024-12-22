"""BART model implementation for resume summary generation."""
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from .base_model import BaseResumeModel
from config.model_config import BART_CONFIG
from config.model_prompts import BART_PROMPT, SUMMARY_TEMPLATES
from config.cache_config import CACHE_CONFIG
from config.text_config import TEXT_CLEAN_CONFIG

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
    
    def clean_output(self, text):
        """Clean up the generated text."""
        # Get name and role from stored data
        name = self.current_name if hasattr(self, 'current_name') else "Unknown"
        role = self.current_role if hasattr(self, 'current_role') else ""
        company = self.current_company if hasattr(self, 'current_company') else ""
        
        # Remove any generated prefixes
        prefixes = TEXT_CLEAN_CONFIG['prefixes_to_remove'] + [f"Hi, I am Hi, I am {name}"]  # Add dynamic prefix
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
            text = text.replace(prefix, "")
        
        # Fix formatting
        text = text.replace("  ", " ")
        text = text.replace(" .", ".")
        text = text.replace(" ,", ",")
        
        # Apply word replacements from config
        for old, new in TEXT_CLEAN_CONFIG['word_replacements'].items():
            text = text.replace(old, new)
        
        # Extract role and company if present in text
        role_company = ""
        if role and company:
            # Look for role and company in text
            role_patterns = [
                f"{role} at {company}",
                f"{role} with {company}",
                f"{role} at {company.split(',')[0]}"  # Try first company only
            ]
            for pattern in role_patterns:
                if pattern in text:
                    role_start = text.find(pattern)
                    role_end = text.find(".", role_start)
                    if role_end != -1:
                        role_company = text[role_start:role_end+1]
                        text = text[:role_start] + text[role_end+1:]
                    break
        
        # Add name introduction
        if not text.lower().startswith("hi, i am"):
            text = f"Hi, I am {name}. " + text
        
        # Add role and company if not present
        if role and company:
            if not any(pattern in text for pattern in [f"{role} at {company}", f"{role} with {company}"]):
                text = text.replace(f"Hi, I am {name}.", f"Hi, I am {name}. I am a {role} at {company}.")
            elif role_company:
                text = text.replace(f"Hi, I am {name}.", f"Hi, I am {name}. {role_company}")
        
        # Ensure proper sentence structure
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        cleaned_sentences = []
        for sentence in sentences[:TEXT_CLEAN_CONFIG['formatting']['max_sentences']]:
            # Remove duplicate role mentions
            if cleaned_sentences and role in sentence and role in cleaned_sentences[0]:
                continue
            if sentence:
                if not sentence.endswith("."):
                    sentence += "."
                cleaned_sentences.append(sentence)
        
        text = " ".join(cleaned_sentences)
        
        # Fix any remaining formatting issues
        text = text.replace("..", ".")
        text = text.replace("  ", " ")
        text = text.strip()
        
        return text
    
    def generate_prompt(self, formatted_data):
        """Generate a prompt using the template strings."""
        # Store data for clean_output
        self.current_name = formatted_data['name']
        self.current_role = formatted_data['current_role']
        self.current_company = formatted_data['companies'].split(',')[0].strip() if formatted_data.get('companies') else ""
        
        # Format template data
        template_data = self.format_template_data(formatted_data)
        
        # Get templates from config
        templates = {k: v.format(**template_data) for k, v in SUMMARY_TEMPLATES.items()}
        
        # Create prompt using BART_PROMPT from config
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
            raise RuntimeError(f"Error in BART generation: {str(e)}")
