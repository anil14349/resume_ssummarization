"""
UI configuration settings.
"""

UI_CONFIG = {
    'page': {
        'title': 'Resume Parser',
        'layout': 'wide',
        'initial_sidebar_state': 'expanded'
    },
    'columns': {
        'input': 0.6,
        'output': 0.4
    },
    'max_file_size': 5 * 1024 * 1024,  # 5MB
    'supported_file_types': ['docx'],
    'text_area': {
        'height': 400
    },
    'model_options': {
        't5': 'T5',
        'gpt2': 'GPT-2',
        'bart': 'BART'
    },
    'parser_options': {
        'ats': 'ATS Parser',
        'industry': 'Industry Manager Parser'
    }
}
