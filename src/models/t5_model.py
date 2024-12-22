"""T5 model implementation for resume summary generation."""
from transformers import T5Tokenizer, T5ForConditionalGeneration
from .base_model import BaseResumeModel
from config.model_config import T5_CONFIG
from config.model_prompts import T5_PROMPT, SUMMARY_TEMPLATES
from config.text_config import TEXT_CLEAN_CONFIG

class T5ResumeModel(BaseResumeModel):
    def __init__(self, model_name="t5-base", device="cpu"):
        """Initialize T5 model with specified configuration."""
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        if device == "cuda":
            try:
                self.model = self.model.to(device)
            except RuntimeError:
                print("CUDA not available, using CPU instead")
                self.device = "cpu"
            
        # Update generation params based on BART success
        self.config['model']['generation_params'].update({
            'max_length': 150,
            'min_length': 75,
            'num_beams': 4,
            'length_penalty': 2.0,
            'no_repeat_ngram_size': 3,
            'early_stopping': True,
            'repetition_penalty': 2.0,
            'do_sample': False,  # Disable sampling for more consistent output
            'temperature': 1.0,  # Use default temperature
            'top_k': None,  # Disable top-k sampling
            'top_p': None  # Disable nucleus sampling
        })

    def clean_output(self, text):
        """Clean up the T5 model output."""
        # Get name and role from stored data
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
            "first-person summary"
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
    
    def format_template_data(self, formatted_data):
        """Format data for template, with additional preprocessing for T5."""
        template_data = super().format_template_data(formatted_data)
        
        # Deduplicate achievements by splitting and removing duplicates
        if 'achievements' in template_data:
            achievements = template_data['achievements'].split(', ')
            unique_achievements = []
            seen = set()
            for achievement in achievements:
                achievement_lower = achievement.lower()
                if achievement_lower not in seen:
                    seen.add(achievement_lower)
                    unique_achievements.append(achievement)
            template_data['achievements'] = ', '.join(unique_achievements)
        
        return template_data
    
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
        
        # Create prompt using T5_PROMPT from config
        prompt = T5_PROMPT.format(**templates)
        
        # T5 works better with explicit task prefixes
        prompt = "summarize professionally: " + prompt
        
        return prompt.replace('\n', ' ').strip()

    def generate_summary(self, input_json):
        """Generate a summary using T5 model."""
        try:
            # Format input data
            formatted_data = self.format_input_data(input_json)
            
            # Generate prompt
            prompt = self.generate_prompt(formatted_data)
            
            # Tokenize input
            input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).input_ids
            
            if self.device == "cuda":
                input_ids = input_ids.to(self.device)
            
            # Generate summary with focused parameters
            outputs = self.model.generate(
                input_ids,
                max_length=200,
                min_length=100,
                num_beams=4,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=2.0
            )
            
            # Decode and clean output
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self.clean_output(summary)
            
        except Exception as e:
            raise RuntimeError(f"Error in T5 generation: {str(e)}")
