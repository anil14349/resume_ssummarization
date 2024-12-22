"""Configuration for model caching."""
import os
from pathlib import Path
import tempfile

# Default cache locations
DEFAULT_CACHE_LOCATIONS = {
    'system': str(Path.home() / '.cache' / 'tv3' / 'models'),  # System-wide cache
    'user': str(Path.home() / '.tv3' / 'models_cache'),  # User-specific cache
    'temp': str(Path(tempfile.gettempdir()) / 'tv3_models'),  # Temporary cache
    'project': str(Path(__file__).parent.parent.parent / 'models_cache')  # Project-specific cache
}

# Get environment variables with defaults
CACHE_CONFIG = {
    'cache': {
        # Cache location settings
        'location': {
            'type': os.getenv('TV3_CACHE_LOCATION_TYPE', 'user'),  # One of: system, user, temp, project, custom
            'custom_dir': os.getenv('TV3_CACHE_DIR', ''),  # Used if type is 'custom'
            'base_dir': '',  # Will be set based on type
            'create_parents': True,  # Create parent directories if they don't exist
            'cleanup_on_exit': os.getenv('TV3_CACHE_CLEANUP_ON_EXIT', 'false').lower() == 'true'
        },
        
        # Cache behavior settings
        'behavior': {
            'compression': {
                'enabled': os.getenv('TV3_CACHE_COMPRESSION', 'true').lower() == 'true',
                'algorithm': os.getenv('TV3_CACHE_COMPRESSION_ALGO', 'gzip'),  # gzip, bzip2, or lzma
                'level': int(os.getenv('TV3_CACHE_COMPRESSION_LEVEL', '6'))  # 1-9
            },
            'validation': {
                'verify_hashes': os.getenv('TV3_CACHE_VERIFY_HASHES', 'true').lower() == 'true',
                'check_integrity': os.getenv('TV3_CACHE_CHECK_INTEGRITY', 'true').lower() == 'true'
            },
            'retry': {
                'max_attempts': int(os.getenv('TV3_CACHE_MAX_RETRY', '3')),
                'delay_seconds': float(os.getenv('TV3_CACHE_RETRY_DELAY', '1.0'))
            }
        },
        
        # Cache limits and thresholds
        'limits': {
            'max_size_gb': float(os.getenv('TV3_CACHE_MAX_SIZE', '10')),
            'min_free_space_gb': float(os.getenv('TV3_CACHE_MIN_FREE_SPACE', '5')),
            'max_files': int(os.getenv('TV3_CACHE_MAX_FILES', '1000')),
            'cleanup_threshold': float(os.getenv('TV3_CACHE_CLEANUP_THRESHOLD', '0.9')),
            'file_size_limit_mb': float(os.getenv('TV3_CACHE_FILE_SIZE_LIMIT', '1000'))
        },
        
        # Logging and monitoring
        'monitoring': {
            'enabled': os.getenv('TV3_CACHE_MONITORING', 'true').lower() == 'true',
            'log_level': os.getenv('TV3_CACHE_LOG_LEVEL', 'INFO'),
            'stats_interval_mins': float(os.getenv('TV3_CACHE_STATS_INTERVAL', '60')),
            'alert_threshold_gb': float(os.getenv('TV3_CACHE_ALERT_THRESHOLD', '8'))
        },
        
        # Model-specific cache settings
        'models': {
            't5': {
                'cache_dir': 't5_models',
                'max_models': int(os.getenv('TV3_T5_MAX_MODELS', '2')),
                'compression_level': int(os.getenv('TV3_T5_COMPRESSION_LEVEL', '6')),
                'priority': int(os.getenv('TV3_T5_PRIORITY', '1')),  # Higher number = higher priority
                'ttl_days': float(os.getenv('TV3_T5_TTL_DAYS', '30'))  # Time-to-live in days
            },
            'gpt2': {
                'cache_dir': 'gpt2_models',
                'max_models': int(os.getenv('TV3_GPT2_MAX_MODELS', '2')),
                'compression_level': int(os.getenv('TV3_GPT2_COMPRESSION_LEVEL', '6')),
                'priority': int(os.getenv('TV3_GPT2_PRIORITY', '1')),
                'ttl_days': float(os.getenv('TV3_GPT2_TTL_DAYS', '30'))
            },
            'bart': {
                'cache_dir': 'bart_models',
                'max_models': int(os.getenv('TV3_BART_MAX_MODELS', '2')),
                'compression_level': int(os.getenv('TV3_BART_COMPRESSION_LEVEL', '6')),
                'priority': int(os.getenv('TV3_BART_PRIORITY', '1')),
                'ttl_days': float(os.getenv('TV3_BART_TTL_DAYS', '30'))
            }
        }
    }
}

# Set base_dir based on location type
location_type = CACHE_CONFIG['cache']['location']['type']
if location_type == 'custom' and CACHE_CONFIG['cache']['location']['custom_dir']:
    CACHE_CONFIG['cache']['location']['base_dir'] = CACHE_CONFIG['cache']['location']['custom_dir']
else:
    CACHE_CONFIG['cache']['location']['base_dir'] = DEFAULT_CACHE_LOCATIONS.get(location_type, DEFAULT_CACHE_LOCATIONS['user'])
