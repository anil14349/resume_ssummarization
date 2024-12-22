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
        'T5 Model (Fast & Efficient)': {
            'type': 't5',
            'size': 'base',
            'description': 'Best for quick, concise summaries that focus on key points'
        },
        'GPT-2 Model (Creative)': {
            'type': 'gpt2',
            'size': 'medium',
            'description': 'Good for natural-sounding summaries with a professional tone'
        },
        'BART Model (Detailed)': {
            'type': 'bart',
            'size': 'large',
            'description': 'Ideal for comprehensive summaries with balanced detail and clarity'
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
