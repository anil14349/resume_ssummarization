"""Configuration file for model parameters."""

T5_CONFIG = {
    'model': {
        'name': 't5-base',
        'generation_params': {
            'max_length': 150,
            'min_length': 50,
            'num_beams': 4,
            'length_penalty': 1.0,
            'no_repeat_ngram_size': 3,
            'early_stopping': True
        }
    }
}

GPT2_CONFIG = {
    'model': {
        'name': 'gpt2-medium',
        'generation_params': {
            'max_length': 200,
            'min_length': 50,
            'num_beams': 4,
            'length_penalty': 1.0,
            'no_repeat_ngram_size': 3,
            'early_stopping': True,
            'pad_token_id': 50256
        }
    }
}

BART_CONFIG = {
    'model': {
        'name': 'facebook/bart-large',
        'generation_params': {
            'max_length': 150,
            'min_length': 50,
            'num_beams': 5,
            'length_penalty': 0.6,
            'no_repeat_ngram_size': 3,
            'early_stopping': True,
            'repetition_penalty': 1.3,
            'do_sample': False
        }
    }
}
