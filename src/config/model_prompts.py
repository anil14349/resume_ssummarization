"""
This module contains the prompt templates used by the different models for generating summaries.
"""

# Templates for different sections of the summary
SUMMARY_TEMPLATES = {
    'greeting': "Hi, this is {name}",
    'role': "I am currently working as a {current_role} with {years_experience} years of experience",
    'experience': "Throughout my career, I have {achievements}",
    'skills': "My core expertise includes {skills}",
    'education': "I hold {education}",
    'recognition': ". {recognition}" if "{recognition}" else ""
}

# Model-specific prompts that use the templates
T5_PROMPT = """Generate a professional summary:
{greeting}. {role}. {experience}. {skills}. {education}{recognition}
"""

GPT2_PROMPT = """Create a professional summary:
{greeting}. {role}. {experience}. {skills}. {education}{recognition}
"""

BART_PROMPT = """Write a professional summary:
{greeting}. {role}. {experience}. {skills}. {education}{recognition}"""
