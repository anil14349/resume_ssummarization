# models/model_cache.py
"""
This module handles caching of transformer models to avoid repeated downloads.
"""
import json
import shutil
import logging
import gzip
import bz2
import lzma
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
import psutil
from config.cache_config import CACHE_CONFIG

logger = logging.getLogger(__name__)

class ModelCache:
    """Manages caching of transformer models."""
    
    def __init__(self, cache_dir=None):
        """Initialize the model cache.
        
        Args:
            cache_dir (str, optional): Directory to store cached models.
                If not provided, uses the directory from cache_config.py.
                Can be overridden by TV3_CACHE_DIR environment variable.
        """
        # Set up logging
        log_level = getattr(logging, CACHE_CONFIG['cache']['monitoring']['log_level'].upper())
        logger.setLevel(log_level)
        
        # Initialize cache directory
        if cache_dir is None:
            cache_dir = CACHE_CONFIG['cache']['location']['base_dir']
        
        self.cache_dir = Path(cache_dir)
        if CACHE_CONFIG['cache']['location']['create_parents']:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata file
        self.metadata_file = self.cache_dir / "metadata.json"
        if not self.metadata_file.exists():
            self._init_metadata()
        
        # Load settings
        self.model_settings = CACHE_CONFIG['cache']['models']
        self.limits = CACHE_CONFIG['cache']['limits']
        self.behavior = CACHE_CONFIG['cache']['behavior']
        self.monitoring = CACHE_CONFIG['cache']['monitoring']
        
        # Set up compression
        self.compression = self.behavior['compression']
        self.compressors = {
            'gzip': gzip,
            'bzip2': bz2,
            'lzma': lzma
        }
        
        # Ensure model-specific directories exist
        for model_type, settings in self.model_settings.items():
            model_dir = self.cache_dir / settings['cache_dir']
            model_dir.mkdir(parents=True, exist_ok=True)
        
        # Start monitoring if enabled
        if self.monitoring['enabled']:
            self._start_monitoring()
    
    def _compress_file(self, file_path):
        """Compress a file using the configured compression algorithm."""
        if not self.compression['enabled']:
            return file_path
        
        compressor = self.compressors[self.compression['algorithm']]
        compressed_path = file_path.with_suffix(f".{self.compression['algorithm']}")
        
        with open(file_path, 'rb') as f_in:
            with compressor.open(compressed_path, 'wb', compresslevel=self.compression['level']) as f_out:
                f_out.write(f_in.read())
        
        file_path.unlink()  # Remove original file
        return compressed_path
    
    def _decompress_file(self, file_path):
        """Decompress a file using its extension to determine the algorithm."""
        if not self.compression['enabled']:
            return file_path
            
        algo = file_path.suffix[1:]  # Remove the dot
        if algo not in self.compressors:
            return file_path
            
        compressor = self.compressors[algo]
        decompressed_path = file_path.with_suffix('')
        
        with compressor.open(file_path, 'rb') as f_in:
            with open(decompressed_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        return decompressed_path
    
    def _calculate_hash(self, file_path):
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _verify_integrity(self, file_path, stored_hash):
        """Verify file integrity using stored hash."""
        if not self.behavior['validation']['verify_hashes']:
            return True
        return self._calculate_hash(file_path) == stored_hash
    
    def _init_metadata(self):
        """Initialize metadata file with default values."""
        metadata = {
            'models': {},
            'last_cleanup': datetime.now().isoformat(),
            'total_size': 0,
            'stats': {
                'hits': 0,
                'misses': 0,
                'cleanups': 0
            }
        }
        self._save_metadata(metadata)
    
    def _get_model_dir(self, model_type, model_name):
        """Get the directory for a specific model type."""
        if model_type not in self.model_settings:
            raise ValueError(f"Unknown model type: {model_type}")
        return self.cache_dir / self.model_settings[model_type]['cache_dir'] / model_name
    
    def _check_cache_size(self):
        """Check if cache cleanup is needed based on size and free space."""
        total_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
        total_size_gb = total_size / (1024 ** 3)  # Convert to GB
        
        # Check various limits
        if total_size_gb > self.limits['max_size_gb']:
            logger.warning(f"Cache size ({total_size_gb:.2f}GB) exceeds limit ({self.limits['max_size_gb']}GB)")
            return True
        
        free_space_gb = psutil.disk_usage(self.cache_dir).free / (1024 ** 3)
        if free_space_gb < self.limits['min_free_space_gb']:
            logger.warning(f"Low disk space: {free_space_gb:.2f}GB free")
            return True
        
        file_count = sum(1 for _ in self.cache_dir.rglob('*') if _.is_file())
        if file_count > self.limits['max_files']:
            logger.warning(f"Too many files in cache: {file_count}")
            return True
        
        return False
    
    def _cleanup_cache(self):
        """Remove least recently used models to free up space."""
        metadata = self._load_metadata()
        
        # Update stats
        metadata['stats']['cleanups'] += 1
        self._save_metadata(metadata)
        
        # Get all models with their priorities and TTL
        all_models = []
        for model_type, models in metadata['models'].items():
            settings = self.model_settings[model_type]
            for model_name, info in models.items():
                # Check if model has expired
                last_access = datetime.fromisoformat(info['last_access'])
                ttl = timedelta(days=settings['ttl_days'])
                expired = datetime.now() - last_access > ttl
                
                all_models.append({
                    'type': model_type,
                    'name': model_name,
                    'last_access': last_access,
                    'size': info['size'],
                    'priority': settings['priority'],
                    'expired': expired
                })
        
        # Sort by expired (True first), then priority (lower first), then last access time
        all_models.sort(key=lambda x: (not x['expired'], x['priority'], x['last_access']))
        
        # Remove models until we're under all thresholds
        while self._check_cache_size() and all_models:
            model = all_models.pop(0)
            self.remove_model(model['type'], model['name'])
            logger.info(f"Removed {model['type']}-{model['name']} from cache (expired={model['expired']})")
    
    def _start_monitoring(self):
        """Start cache monitoring if enabled."""
        if not self.monitoring['enabled']:
            return
            
        def log_stats():
            metadata = self._load_metadata()
            total_size_gb = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file()) / (1024 ** 3)
            free_space_gb = psutil.disk_usage(self.cache_dir).free / (1024 ** 3)
            
            logger.info(f"Cache stats: size={total_size_gb:.2f}GB, free={free_space_gb:.2f}GB")
            logger.info(f"Cache hits/misses: {metadata['stats']['hits']}/{metadata['stats']['misses']}")
            
            # Check if we're approaching limits
            if total_size_gb > self.monitoring['alert_threshold_gb']:
                logger.warning(f"Cache size ({total_size_gb:.2f}GB) approaching limit ({self.limits['max_size_gb']}GB)")
        
        # Log initial stats
        log_stats()
    
    def get_cached_model_path(self, model_type, model_name):
        """Get the path to a cached model if it exists."""
        model_dir = self._get_model_dir(model_type, model_name)
        if model_dir.exists():
            # Update metadata
            metadata = self._load_metadata()
            if model_type in metadata['models'] and model_name in metadata['models'][model_type]:
                metadata['stats']['hits'] += 1
                metadata['models'][model_type][model_name]['last_access'] = datetime.now().isoformat()
                self._save_metadata(metadata)
                
                # Verify integrity if enabled
                if self.behavior['validation']['check_integrity']:
                    stored_hash = metadata['models'][model_type][model_name].get('hash')
                    if stored_hash and not self._verify_integrity(model_dir, stored_hash):
                        logger.error(f"Integrity check failed for {model_type}-{model_name}")
                        self.remove_model(model_type, model_name)
                        return None
                
                return str(model_dir)
            
        metadata = self._load_metadata()
        metadata['stats']['misses'] += 1
        self._save_metadata(metadata)
        return None
    
    def cache_model(self, model_type, model_name, model_path):
        """Cache a model and update metadata."""
        # Check cache limits
        if self._check_cache_size():
            self._cleanup_cache()
        
        target_dir = self._get_model_dir(model_type, model_name)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        
        # Retry logic for copying files
        max_attempts = self.behavior['retry']['max_attempts']
        delay = self.behavior['retry']['delay_seconds']
        
        for attempt in range(max_attempts):
            try:
                # Copy and compress model files
                shutil.copytree(model_path, target_dir)
                if self.compression['enabled']:
                    for file in target_dir.rglob('*'):
                        if file.is_file() and file.suffix not in ['.gz', '.bz2', '.xz']:
                            self._compress_file(file)
                
                # Calculate hash if validation is enabled
                model_hash = None
                if self.behavior['validation']['verify_hashes']:
                    model_hash = self._calculate_hash(target_dir)
                
                # Update metadata
                metadata = self._load_metadata()
                if model_type not in metadata['models']:
                    metadata['models'][model_type] = {}
                
                metadata['models'][model_type][model_name] = {
                    'last_access': datetime.now().isoformat(),
                    'size': sum(f.stat().st_size for f in target_dir.rglob('*') if f.is_file()),
                    'hash': model_hash
                }
                
                self._save_metadata(metadata)
                break
                
            except Exception as e:
                if attempt == max_attempts - 1:
                    logger.error(f"Failed to cache model after {max_attempts} attempts: {e}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay} seconds")
                time.sleep(delay)
    
    def remove_model(self, model_type, model_name):
        """Remove a model from the cache."""
        model_dir = self._get_model_dir(model_type, model_name)
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Update metadata
        metadata = self._load_metadata()
        if model_type in metadata['models'] and model_name in metadata['models'][model_type]:
            del metadata['models'][model_type][model_name]
            self._save_metadata(metadata)
    
    def clear_cache(self):
        """Clear all cached models."""
        try:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True)
            self._init_metadata()
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            raise
    
    def _load_metadata(self):
        """Load cache metadata from file."""
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return self._init_metadata()
    
    def _save_metadata(self, metadata):
        """Save cache metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            raise
    
    def __del__(self):
        """Cleanup on object destruction."""
        if CACHE_CONFIG['cache']['location']['cleanup_on_exit']:
            self.clear_cache()
