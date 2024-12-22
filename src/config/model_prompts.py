"""
Model prompt templates.
"""

from typing import Dict

# Common templates for all models
SUMMARY_TEMPLATES = {
    'name': "Hi, I am {name}",
    'role': "{current_role}",
    'company': "{companies}",
    'experience': "{years_experience} years of experience",
    'achievements': "{achievements}",
    'skills': "{skills}",
    'education': "{education}",
    'recognition': "{recognition}"
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
T5_PROMPT = """Summarize this professional profile:
Background: {role} with {experience}.
Key Achievements: {achievements}
Technical Skills: {skills}
Education: {education}
{recognition}

Generate a concise, first-person professional summary that highlights achievements and expertise.
"""

# GPT2-specific prompt template
GPT2_PROMPT = """Create a compelling professional profile:
Role: {role}
Experience: {experience}
Achievements: {achievements}
Skills: {skills}
Education: {education}
{recognition}

Write a concise, engaging first-person summary that emphasizes achievements and skills.
"""

# BART-specific prompt template
BART_PROMPT = """Generate a professional summary for this profile:
Professional Background:
- Current Role: {role}
- Experience: {experience}
- Achievements: {achievements}

Technical Expertise:
{skills}

Education:
{education}
{recognition}

Write a concise, professional first-person summary."""
