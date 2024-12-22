"""
Configuration for text cleaning and formatting.
"""

# Common text cleaning configuration
TEXT_CLEAN_CONFIG = {
    'prefixes_to_remove': [
        "Generate a professional summary for this profile:",
        "Professional Background:",
        "Technical Expertise:",
        "Current Role:",
        "Experience:",
        "Achievements:",
        "Write a professional summary:",
        "Summary:",
        "Profile:",
        "Write a professional first-person summary",
        "Start with",
        "Name:",
        "Company:",
        "Skills:",
        "Education:",
        "Key Achievements:",
        "- ",
        "based on this profile:",
        "Write a professional first-person summary based on this profile:",
        "Recognition:",
        "Generate a concise professional summary in first person:",
        "Key "
    ],
    'word_replacements': {
        " hr ": " HR ",
        " osha ": " OSHA ",
        "human resources": "Human Resources",
        "we raised": "I raised",
        "we increased": "I increased",
        "we developed": "I developed",
        "we reduced": "I reduced",
        "we built": "I built",
        "we deployed": "I deployed",
        "we led": "I led",
        "  ": " ",
        " .": ".",
        " ,": ","
    },
    'formatting': {
        'max_sentences': 4,
        'min_words': 50
    }
}
