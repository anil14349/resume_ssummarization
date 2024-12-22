"""
Model prompt templates.
"""

from typing import Dict

# Common templates for all models
SUMMARY_TEMPLATES = {
    'name': '{name}',
    'role': '{current_role}',
    'company': '{companies}',
    'experience': '{years_experience}',
    'achievements': '{achievements}',
    'skills': '{skills}',
    'education': '{education}',
    'recognition': '{recognition}' if '{recognition}' else ''
}

# T5-specific summary template
T5_SUMMARY_TEMPLATE = """Write a professional first-person summary:

Name: {name}
Current Role: {role} at {company}
Experience: {experience} in HR
Key Achievements: {achievements}
Core Skills: {skills}
Education: {education}

Start with "Hi, I am {name}" and describe your role, experience, and key achievements."""

# T5-specific prompt template
T5_PROMPT = """Write a detailed professional first-person summary:

Name: {name}
Current Role: {role} at {company}
Experience: {experience} in HR
Key Achievements:
- {achievements}

Core Skills: {skills}
Education: {education}
{recognition}

Create a summary that:
1. Starts with "Hi, I am {name}"
2. Describes your current role and experience
3. Highlights key achievements with metrics
4. Shows your expertise and impact"""

# GPT2-specific prompt template
GPT2_PROMPT = """Hi, I am {name}. I am a {role} at {company} with {experience} years of experience in HR. {achievements}

Technical Skills: {skills}
Education: {education}
{recognition}

Write a professional first-person summary that highlights my experience and achievements."""

# BART-specific prompt template
BART_PROMPT = """Generate a concise professional summary in first person:

Name: {name}
Current Role: {role} at {company}
Experience: {experience} in HR
Key Achievements:
- {achievements}

Core Skills: {skills}
Education: {education}
{recognition}

Begin with "Hi, I am {name}" and highlight your role, experience, and key achievements."""
