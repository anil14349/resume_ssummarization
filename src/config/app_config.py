"""
Application-wide configuration settings.
"""

# UI Configuration
UI_CONFIG = {
    'layout': {
        'column_ratios': {
            'main_content': [2, 1],  # Ratio for main columns (upload/model selection)
            'action_buttons': [1, 1, 2]  # Ratio for action buttons
        },
        'text_area': {
            'default_height': 300,
            'min_height': 200,
            'font_family': 'Source Code Pro, monospace',
            'font_size': '1rem',
            'line_height': 1.5
        }
    },
    'models': {
        'GPT-2 (Professional)': {
            'type': 'gpt2',
            'size': 'base',
            'description': 'Generates concise, professional summaries with a focus on achievements and skills'
        },
        'T5 (Balanced)': {
            'type': 't5',
            'size': 'base',
            'description': 'Creates well-rounded summaries that balance experience and accomplishments'
        },
        'BART (Detailed)': {
            'type': 'bart',
            'size': 'base',
            'description': 'Produces detailed summaries with comprehensive coverage of skills and experience'
        }
    }
}

# Summary Generation Configuration
SUMMARY_CONFIG = {
    'output': {
        'max_sentences': 5,
        'min_sentences': 2
    },
    'filtering': {
        'stop_phrases': [
            'Please feel',
            'For more information',
            'Contact me',
            'I am currently',
            'I would like',
            'In my spare time',
            'References',
            'She is',
            'She has',
            'He is',
            'He has',
            'In addition',
            'Furthermore',
            'Moreover',
            'Also',
            'Additionally'
        ],
        'membership_words': [
            'member of',
            'board of',
            'graduate of',
            'professor',
            'taught',
            'served',
            'university',
            'college',
            'school'
        ]
    },
    'formatting': {
        'cleanup_replacements': {
            '..': '.',
            'I\'mplement': 'implement',
            '[\'': '',
            '\']': '',
            '\',': ',',
            ' \'': ' '
        }
    }
}
