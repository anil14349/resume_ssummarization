"""
This module contains model configurations.
"""

# BART model configuration
BART_CONFIG = {
    'model': {
        'name': 'facebook/bart-large',
        'generation_params': {
            'max_length': 150,          # Shorter length for more focused output
            'min_length': 75,           # Reasonable minimum length
            'num_beams': 4,             # Standard beam search
            'length_penalty': 1.0,      # Neutral length penalty
            'early_stopping': True,     # Stop when all beams are finished
            'no_repeat_ngram_size': 4,  # Prevent 4-gram repetitions
            'do_sample': False,         # Deterministic generation
            'repetition_penalty': 1.8   # Stronger repetition penalty
        }
    }
}

# T5 model configuration
T5_CONFIG = {
    'model': {
        'name': 't5-base',
        'generation_params': {
            'max_new_tokens': 150,
            'min_length': 75,
            'num_beams': 4,
            'length_penalty': 1.0,
            'early_stopping': True,
            'no_repeat_ngram_size': 3,
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.9,
        }
    }
}

# GPT2 model configuration
GPT2_CONFIG = {
    'model': {
        'name': 'gpt2-medium',
        'generation_params': {
            'max_new_tokens': 150,      # Reduced to prevent extra content
            'min_length': 100,          # Ensure reasonable length
            'num_beams': 5,             # More beams for better quality
            'length_penalty': 0.7,      # Stronger preference for shorter outputs
            'early_stopping': True,
            'no_repeat_ngram_size': 3,  # Prevent repetition
            'do_sample': True,
            'temperature': 0.5,         # Even more conservative sampling
            'top_p': 0.8,              # More focused sampling
            'pad_token_id': None,       # Will be set to eos_token_id in model
            'repetition_penalty': 1.4   # Stronger repetition penalty
        }
    }
}
