"""
Model prompts and templates.
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
GPT2_PROMPT = """Generate a professional first-person summary that highlights expertise and achievements.

Name: {name}
Current Role: {current_role}
Company: {company}
Experience: {years_experience} years
Key Achievements:
{achievements}
Core Skills: {skills}

Write a concise, professional summary that:
1. Starts with "I am {name}"
2. Highlights current role and experience
3. Emphasizes key achievements with metrics
4. Mentions relevant skills
5. Uses a professional tone
6. Focuses on concrete accomplishments
7. Avoids generic statements
8. Keeps to 3-4 sentences

Summary:
I am"""

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
