# Template strings for each model
SUMMARY_TEMPLATES = {
    'greeting': "Hi, I am {name}",
    'role': "I am a {current_role}",
    'experience': "with experience at {companies}",
    'achievements': "My key achievements include {achievements}",
    'skills': "I am skilled in {skills}",
    'education': "and hold a {education}"
}

T5_PROMPT = """Write a professional summary: {greeting} and {role} {experience}. {achievements}. {skills}."""

GPT2_PROMPT = """Generate a professional summary:
{greeting} and {role} {experience}. {achievements}. {skills} {education}."""

BART_PROMPT = """Generate a professional summary:
{greeting} and {role} {experience}. {achievements}. {skills} {education}."""
