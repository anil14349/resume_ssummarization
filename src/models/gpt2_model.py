"""
GPT2 model implementation for resume summary generation.
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .base_model import BaseResumeModel
from config.model_config import GPT2_CONFIG
from config.model_prompts import GPT2_PROMPT, SUMMARY_TEMPLATES
from config.text_config import TEXT_CLEAN_CONFIG

class GPT2ResumeModel(BaseResumeModel):
    def __init__(self, model_name="gpt2", device="cpu"):
        """Initialize GPT2 model with specified configuration."""
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Set pad token to eos token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        if device == "cuda":
            try:
                self.model = self.model.to(device)
            except RuntimeError:
                print("CUDA not available, using CPU instead")
                self.device = "cpu"
            
        # Update generation params
        self.config['model']['generation_params'].update({
            'max_length': None,  # Let max_new_tokens control length
            'max_new_tokens': 150,  # Control output length
            'min_length': 100,
            'num_beams': 5,  # Increased for better search
            'length_penalty': 1.5,  # Reduced to favor shorter outputs
            'no_repeat_ngram_size': 3,
            'early_stopping': True,
            'repetition_penalty': 1.5,  # Reduced to allow some repetition
            'do_sample': True,
            'temperature': 0.8,  # Slightly increased for more variety
            'top_p': 0.95,  # Increased for more options
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id
        })

    def clean_output(self, text):
        """Clean up the GPT2 model output."""
        # Get data from stored attributes
        name = self.current_name if hasattr(self, 'current_name') else "Unknown"
        role = self.current_role if hasattr(self, 'current_role') else ""
        company = self.current_company if hasattr(self, 'current_company') else ""
        experience = self.current_experience if hasattr(self, 'current_experience') else ""
        achievements = self.current_achievements if hasattr(self, 'current_achievements') else ""
        
        # Remove any generated prefixes
        prefixes = TEXT_CLEAN_CONFIG['prefixes_to_remove'] + [
            f"Hi, I am Hi, I am {name}",
            "True Story:",
            "Here's my story:",
            "Let me tell you about myself:",
            "Here's what I do:",
            "write a detailed",
            "create a summary",
            "generate a summary",
            "professional summary",
            "first-person summary",
            "Instructions:",
            "Generate a professional",
            "Begin with",
            "Focus on",
            "Keep it"
        ]
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
            text = text.replace(prefix, "")
        
        # Fix formatting
        text = text.replace("  ", " ")
        text = text.replace(" .", ".")
        text = text.replace(" ,", ",")
        text = text.replace(" at at ", " at ")
        text = text.replace("I am a I am", "I am")
        text = text.replace("I am an I am", "I am")
        text = text.replace(" : ", ": ")
        text = text.replace(": ", " ")
        
        # Apply word replacements from config
        for old, new in TEXT_CLEAN_CONFIG['word_replacements'].items():
            text = text.replace(old, new)
        
        # Add name introduction if missing
        if not text.lower().startswith("hi, i am"):
            text = f"Hi, I am {name}. " + text
        
        # Add role and experience if missing
        intro_text = f"Hi, I am {name}. I am a {role} at {company}"
        if experience:
            intro_text += f" with {experience} years of experience in HR"
        intro_text += "."
        
        if not any(pattern in text for pattern in [f"{role} at {company}", f"{role} with {company}"]):
            text = text.replace(f"Hi, I am {name}.", intro_text)
        
        # Add achievements if missing
        if achievements and not any(metric in text for metric in ["10%", "14%", "90%", "improved", "increased", "reduced"]):
            text = text.rstrip(".") + f". {achievements}"
        
        # Ensure proper sentence structure
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        cleaned_sentences = []
        for sentence in sentences[:TEXT_CLEAN_CONFIG['formatting']['max_sentences']]:
            # Remove duplicate role mentions
            if cleaned_sentences and role in sentence and role in cleaned_sentences[0]:
                continue
            # Remove sentences that are too similar to previous ones
            if cleaned_sentences and any(self._similarity_score(sentence, prev) > 0.7 for prev in cleaned_sentences):
                continue
            if sentence:
                if not sentence.endswith("."):
                    sentence += "."
                cleaned_sentences.append(sentence)
        
        text = " ".join(cleaned_sentences)
        
        # Fix any remaining formatting issues
        text = text.replace("..", ".")
        text = text.replace("  ", " ")
        text = text.replace('"', "")
        text = text.strip()
        
        return text
    
    def _similarity_score(self, s1, s2):
        """Simple similarity score between two sentences."""
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0
    
    def generate_prompt(self, formatted_data):
        """Generate a prompt using the template strings."""
        # Store data for clean_output
        self.current_name = formatted_data['name']
        self.current_role = formatted_data['current_role']
        self.current_company = formatted_data['companies'].split(',')[0].strip() if formatted_data.get('companies') else ""
        self.current_experience = formatted_data.get('years_experience', "")
        self.current_achievements = formatted_data.get('achievements', "")
        
        # Format template data
        template_data = self.format_template_data(formatted_data)
        
        # Get templates from config
        templates = {k: v.format(**template_data) for k, v in SUMMARY_TEMPLATES.items()}
        
        # Create prompt using GPT2_PROMPT from config
        prompt = GPT2_PROMPT.format(**templates)
        
        return prompt.replace('\n', ' ').strip()
    
    def generate_summary(self, input_json):
        """Generate a summary using GPT2 model."""
        try:
            # Format input data
            formatted_data = self.format_input_data(input_json)
            
            # Generate prompt
            prompt = self.generate_prompt(formatted_data)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate summary with attention mask
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **self.config['model']['generation_params']
            )
            
            # Decode and clean output
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self.clean_output(summary)
            
        except Exception as e:
            raise RuntimeError(f"Error in GPT2 generation: {str(e)}")
