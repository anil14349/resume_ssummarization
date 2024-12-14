from abc import ABC, abstractmethod
from config.model_prompts import SUMMARY_TEMPLATES

class BaseResumeModel(ABC):
    """Base class for resume summary generation models."""
    
    def __init__(self):
        self.config = {
            'formatting': {
                'max_achievements': 2,
                'max_responsibilities': 2,
                'max_skills': 3,
                'skill_exclusions': ['fun', 'energetic'],
                'cleanup_words': [
                    "Generate", "Write", "Create", "Summary:", "Profile:", "Experience:",
                    "Achievement 1:", "Achievement 2:", "Education:", "Skills:",
                    "first person", "starting with", "name", "write", "create",
                    "summary", "profile", "resume", "background", "expertise",
                    '"', "'", "says", "she says", "he says", "they say",
                    "Summarize this professional :", "a professional :", "Falk:",
                    "professional summary:", "professional profile:", "summary:",
                    "Here's a", "Here is a", "This is a", "Let me", "I can",
                    "Based on", "According to", "As per", "The following",
                    "Please find", "Please see", "Note that", "It should be noted",
                    "Summarize this professional into a first-person :",
                    "I am committed to driving operational excellence and team development.",
                    "Throughout My career,", "Throughout my career,",
                    "I have achieved significant results including",
                    ", and an M.S.", ", and a Ph.D.", ", and M.B.A.",
                    "Master's Degree", "M.S./M.B.A", "M.P.H.", "Food and Beverage",
                    ", respectively,", "in Hospital", "in Food"
                ],
                'word_replacements': {
                    "successfully ": "",
                    "through ": "via ",
                    "minimum of ": "least ",
                    "and by creating": "through",
                    " i ": " I ",
                    "i am": "I am",
                    "i have": "I have",
                    "my ": "My ",
                    "My is": "Hi, I am",
                    "b.s.": "B.S.",
                    "a.a.": "A.A.",
                    "..": ".",
                    "  ": " ",
                    " .": ".",
                    " ,": ",",
                    "foh": "FOH",
                    "pos": "POS",
                    "My My": "My",
                    "name name": "name",
                    "am am": "am",
                    "and and": "and",
                    "with with": "with",
                    "experience experience": "experience",
                    "at at": "at",
                    ".I": ". I",
                    ".My": ". My",
                    ".Throughout": ". Throughout",
                    "Hi, Hi,": "Hi,",
                    "ity Management": "Hospitality Management",
                    "Management/M.B.A/": "Management",
                    " in$": "",
                    ", .$": ".",
                    ", $": ".",
                    " in\\.?$": "."
                }
            }
        }

    def format_input_data(self, input_json):
        """Format input data for summary generation."""
        # Ensure all required fields are present
        required_fields = ['name', 'current_role', 'companies', 'achievements', 'skills', 'education']
        for field in required_fields:
            if field not in input_json:
                raise ValueError(f"Missing required field: {field}")
        
        # Format the data
        formatted_data = {
            'name': input_json['name'],
            'current_role': input_json['current_role'],
            'companies': input_json['companies'][:2],  # Limit to top 2 companies
            'achievements': input_json['achievements'][:self.config['formatting']['max_achievements']],
            'skills': [s for s in input_json['skills'][:self.config['formatting']['max_skills']] 
                      if s.lower() not in self.config['formatting']['skill_exclusions']],
            'education': input_json['education'][:2]  # Limit to top 2 education entries
        }
        
        return formatted_data

    def format_template_data(self, formatted_data):
        """Format data for template strings."""
        template_data = formatted_data.copy()
        template_data['companies'] = ', '.join(template_data['companies'])
        template_data['achievements'] = ', '.join(template_data['achievements'])
        template_data['skills'] = ', '.join(template_data['skills'])
        template_data['education'] = ', '.join(template_data['education'])
        return template_data
    
    def clean_output(self, summary):
        """Clean up the generated summary."""
        # Remove cleanup words
        for word in self.config['formatting']['cleanup_words']:
            summary = summary.replace(word, "")
        
        # Apply word replacements
        import re
        for old, new in self.config['formatting']['word_replacements'].items():
            if old.startswith('^') or old.endswith('$') or '\\.' in old:
                summary = re.sub(old, new, summary)
            else:
                summary = summary.replace(old, new)
        
        # Clean up whitespace
        summary = " ".join(summary.split())
        
        # Clean up trailing punctuation
        summary = re.sub(r'[,\s]+\.?$', '.', summary)
        summary = re.sub(r'\s+in\.?$', '.', summary)
        
        return summary.strip()
    
    @abstractmethod
    def generate_summary(self, input_json):
        """Generate a summary from input data."""
        pass

    def generate_prompt(self, formatted_data):
        """Generate a prompt using the template strings."""
        template_data = self.format_template_data(formatted_data)
        return {k: v.format(**template_data) for k, v in SUMMARY_TEMPLATES.items()}
