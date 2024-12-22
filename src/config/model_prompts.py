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
GPT2_PROMPT = '''Generate a professional resume summary for:

Name: {name}
Current Role: {role}
Company: {company}
Experience: {experience} years
Key Achievements: {achievements}
Core Skills: {skills}

Write a concise professional summary (3-4 sentences) that:
1. Introduces you by name
2. Describes your current role and experience
3. Highlights your key achievements with metrics
4. Showcases your most relevant skills

Example Output:
"Hi, I am Sarah Chen. I am a Senior HR Manager at TechCorp with 8 years of experience in talent acquisition and employee development. I have successfully implemented performance management systems that improved employee retention by 25% and reduced hiring costs by 30%. My expertise includes HRIS implementation, policy development, and strategic workforce planning."

Write a similar summary for the above person:'''

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
