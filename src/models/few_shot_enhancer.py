"""Few-shot learning enhancement for resume summarization."""
from typing import Dict, List
import json
from pathlib import Path
import random

class FewShotEnhancer:
    def __init__(self):
        """Initialize few-shot enhancer."""
        self.examples_path = Path("data/few_shot_examples")
        self.examples_path.mkdir(parents=True, exist_ok=True)
        self.examples_file = self.examples_path / "examples.json"
        self.load_examples()

    def load_examples(self):
        """Load or initialize few-shot examples."""
        if self.examples_file.exists():
            with open(self.examples_file, "r") as f:
                self.examples = json.load(f)
        else:
            self.examples = {
                "hr": [],
                "tech": [],
                "management": [],
                "sales": [],
                "other": []
            }
            self.save_examples()

    def save_examples(self):
        """Save examples to disk."""
        with open(self.examples_file, "w") as f:
            json.dump(self.examples, f, indent=2)

    def add_example(self, category: str, example: Dict):
        """Add a new example to the database."""
        if category not in self.examples:
            self.examples[category] = []
        self.examples[category].append(example)
        self.save_examples()

    def get_examples(self, resume_data: Dict, num_examples: int = 3) -> str:
        """Get relevant few-shot examples based on resume data."""
        # Determine category based on role
        role = resume_data.get("current_role", "").lower()
        category = "other"
        for cat in ["hr", "tech", "management", "sales"]:
            if cat in role:
                category = cat
                break

        # Get examples for the category
        available_examples = self.examples.get(category, [])
        if not available_examples:
            available_examples = []
            for cat_examples in self.examples.values():
                available_examples.extend(cat_examples)

        # Select random examples
        selected_examples = random.sample(
            available_examples, 
            min(num_examples, len(available_examples))
        )

        if not selected_examples:
            return ""

        # Format examples
        examples_text = "Here are some example summaries:\n\n"
        for i, example in enumerate(selected_examples, 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"Input:\n"
            examples_text += f"Role: {example['role']}\n"
            examples_text += f"Experience: {example['experience']}\n"
            examples_text += f"Skills: {example['skills']}\n\n"
            examples_text += f"Output:\n{example['summary']}\n\n"

        examples_text += "Now, create a similar summary for the current profile.\n"
        return examples_text

    def add_default_examples(self):
        """Add some default examples to the database."""
        hr_example = {
            "role": "Senior HR Manager",
            "experience": "8 years",
            "skills": "Recruitment, Employee Relations, HRIS",
            "summary": "Hi, I am Sarah Johnson. I am a Senior HR Manager with 8 years of experience in talent acquisition and employee development. I have successfully implemented performance management systems that improved employee retention by 25% and reduced hiring costs by 30%. My expertise includes HRIS implementation, policy development, and strategic workforce planning."
        }
        self.add_example("hr", hr_example)

        tech_example = {
            "role": "Senior Software Engineer",
            "experience": "6 years",
            "skills": "Python, AWS, Machine Learning",
            "summary": "Hi, I am Michael Chen. I am a Senior Software Engineer with 6 years of experience in cloud computing and machine learning. I led the development of a microservices architecture that reduced system latency by 40% and improved scalability. My technical expertise includes Python, AWS, and implementing ML models in production."
        }
        self.add_example("tech", tech_example)
