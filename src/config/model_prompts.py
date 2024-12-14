"""
This module contains the prompt templates used by the different models for generating summaries.
"""

# Templates for different sections of the summary
SUMMARY_TEMPLATES = {
    'greeting': "Hi, I am {name}",
    'role': "I am a {current_role} with {years_experience} years of experience at {companies}",
    'experience': "My key achievements include {achievements}",
    'skills': "I am skilled in {skills}",
    'education': "and hold {education}"
}

# Model-specific prompts that use the templates
T5_PROMPT = """
{role}. {skills}. {experience}.
"""

GPT2_PROMPT = """
{greeting} and {role}. {experience}. {skills} {education}.
"""

BART_PROMPT = """
{greeting} and {role}. {skills} {education}. {experience}. I have also been recognized as one of the top restaurant managers in our area.
"""
