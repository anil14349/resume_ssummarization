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
        'name': 'gpt2',
        'generation_params': {
            'max_length': 200,
            'min_length': 50,
            'num_beams': 5,
            'no_repeat_ngram_size': 2,
            'early_stopping': True,
            'temperature': 0.7,  # Reduced from 0.9 to make output more focused
            'top_p': 0.85,      # Reduced from 0.9 to reduce randomness
            'top_k': 40,        # Reduced from 50 to limit vocabulary
            'repetition_penalty': 1.2,
            'length_penalty': 1.0,
            'pad_token_id': 50256,
            'do_sample': True,  # Enable sampling for more natural text
            'num_return_sequences': 1
        }
    },
    'tokenizer': {
        'name': 'gpt2',
        'padding': True,
        'truncation': True,
        'max_length': 512,
        'add_special_tokens': True
    }
}
