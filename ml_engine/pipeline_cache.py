"""
Pipeline Cache — Content-addressable caching for pipeline steps.

Fingerprints (dataset hash + config) are used to skip redundant computation.
Cache is stored as Parquet files for fast I/O.

Usage:
    cache = PipelineCache('outputs/.cache')
    fp = cache.fingerprint(df, config)
    
    cached = cache.get(fp, 'clean')
    if cached:
        cleaned_df, clean_report = cached['df'], cached['report']
    else:
        cleaned_df, clean_report = clean_dataset(df, profile)
        cache.put(fp, 'clean', {'df': cleaned_df, 'report': clean_report})
"""

import os
import json
import time
import hashlib
import logging
import pickle

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PipelineCache:
    """Content-addressable cache for pipeline intermediate results."""
    
    def __init__(self, cache_dir='outputs/.cache', max_entries=10):
        """
        Args:
            cache_dir: Directory to store cached results.
            max_entries: Maximum number of cached fingerprints to keep.
        """
        self.cache_dir = cache_dir
        self.max_entries = max_entries
        os.makedirs(cache_dir, exist_ok=True)
        self._index_path = os.path.join(cache_dir, '_index.json')
        self._index = self._load_index()
    
    def fingerprint(self, df, config=None):
        """
        Generate a content-based fingerprint for a dataset + config.
        
        Uses: shape, dtypes, first/last 5 rows, column names, and config hash.
        This is fast even on million-row DataFrames (~5ms).
        
        Args:
            df: pandas DataFrame.
            config: Optional dict of pipeline configuration.
        
        Returns:
            str: SHA-256 hex digest (64 chars).
        """
        hasher = hashlib.sha256()
        
        # Shape
        hasher.update(f"shape:{df.shape}".encode())
        
        # Column names and dtypes
        col_info = "|".join(f"{c}:{df[c].dtype}" for c in df.columns)
        hasher.update(col_info.encode())
        
        # Sample rows (first 5 + last 5) for content fingerprint
        n = min(5, len(df))
        if n > 0:
            head_bytes = df.head(n).to_csv(index=False).encode()
            tail_bytes = df.tail(n).to_csv(index=False).encode()
            hasher.update(head_bytes)
            hasher.update(tail_bytes)
        
        # Null counts per column (fast summary)
        null_counts = df.isnull().sum().to_dict()
        hasher.update(json.dumps(null_counts, sort_keys=True).encode())
        
        # Config
        if config:
            # Filter to serializable keys only
            safe_config = {}
            for k, v in config.items():
                if isinstance(v, (str, int, float, bool, list, type(None))):
                    safe_config[k] = v
            hasher.update(json.dumps(safe_config, sort_keys=True).encode())
        
        return hasher.hexdigest()
    
    def get(self, fingerprint, step):
        """
        Retrieve cached result for a pipeline step.
        
        Args:
            fingerprint: Dataset fingerprint from self.fingerprint().
            step: Pipeline step name ('profile', 'clean', 'transform', 'feature_eng').
        
        Returns:
            dict with cached data, or None if cache miss.
        """
        cache_key = f"{fingerprint}_{step}"
        entry = self._index.get(cache_key)
        
        if entry is None:
            logger.debug("Cache MISS: %s/%s", step, fingerprint[:12])
            return None
        
        # Check if files still exist
        data_path = entry.get('data_path')
        report_path = entry.get('report_path')
        
        if data_path and not os.path.exists(data_path):
            logger.debug("Cache MISS (file deleted): %s/%s", step, fingerprint[:12])
            del self._index[cache_key]
            self._save_index()
            return None
        
        try:
            result = {}
            
            # Load DataFrame (Parquet)
            if data_path and os.path.exists(data_path):
                result['df'] = pd.read_parquet(data_path)
            
            # Load report (pickle)
            if report_path and os.path.exists(report_path):
                with open(report_path, 'rb') as f:
                    result['report'] = pickle.load(f)
            
            logger.info("Cache HIT: %s/%s (saved %.1fs)",
                        step, fingerprint[:12], entry.get('original_time', 0))
            
            # Update access time
            entry['last_accessed'] = time.time()
            self._save_index()
            
            return result
            
        except Exception as e:
            logger.warning("Cache read error for %s/%s: %s", step, fingerprint[:12], e)
            return None
    
    def put(self, fingerprint, step, result, elapsed_time=0):
        """
        Store a pipeline step result in the cache.
        
        Args:
            fingerprint: Dataset fingerprint.
            step: Pipeline step name.
            result: dict with 'df' (DataFrame) and/or 'report' (dict).
            elapsed_time: How long the step took (for logging on cache hits).
        """
        cache_key = f"{fingerprint}_{step}"
        step_dir = os.path.join(self.cache_dir, fingerprint[:16], step)
        os.makedirs(step_dir, exist_ok=True)
        
        entry = {
            'created': time.time(),
            'last_accessed': time.time(),
            'original_time': elapsed_time,
            'fingerprint': fingerprint,
            'step': step,
        }
        
        try:
            # Save DataFrame as Parquet
            if 'df' in result and result['df'] is not None:
                data_path = os.path.join(step_dir, 'data.parquet')
                result['df'].to_parquet(data_path, index=False)
                entry['data_path'] = data_path
            
            # Save report as pickle
            if 'report' in result and result['report'] is not None:
                report_path = os.path.join(step_dir, 'report.pkl')
                with open(report_path, 'wb') as f:
                    pickle.dump(result['report'], f)
                entry['report_path'] = report_path
            
            self._index[cache_key] = entry
            self._evict_if_needed()
            self._save_index()
            
            logger.info("Cache STORE: %s/%s (%.1fs computation cached)",
                        step, fingerprint[:12], elapsed_time)
            
        except Exception as e:
            logger.warning("Cache write error for %s/%s: %s", step, fingerprint[:12], e)
    
    def invalidate(self, fingerprint=None):
        """
        Invalidate cache entries.
        
        Args:
            fingerprint: If provided, invalidate only entries for this fingerprint.
                         If None, clear all cache.
        """
        if fingerprint is None:
            self._index = {}
        else:
            keys_to_remove = [k for k in self._index if k.startswith(fingerprint)]
            for k in keys_to_remove:
                del self._index[k]
        
        self._save_index()
        logger.info("Cache invalidated: %s",
                     fingerprint[:12] if fingerprint else "ALL")
    
    def stats(self):
        """Return cache statistics."""
        total_entries = len(self._index)
        total_time_saved = sum(e.get('original_time', 0) for e in self._index.values())
        unique_fingerprints = len(set(e.get('fingerprint', '') for e in self._index.values()))
        
        return {
            'total_entries': total_entries,
            'unique_datasets': unique_fingerprints,
            'total_time_saved_seconds': round(total_time_saved, 1),
            'cache_dir': self.cache_dir,
        }
    
    # ── Internal ────────────────────────────────────────────
    
    def _load_index(self):
        """Load cache index from disk."""
        if os.path.exists(self._index_path):
            try:
                with open(self._index_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_index(self):
        """Persist cache index to disk."""
        try:
            with open(self._index_path, 'w') as f:
                json.dump(self._index, f, indent=2)
        except IOError as e:
            logger.warning("Failed to save cache index: %s", e)
    
    def _evict_if_needed(self):
        """Remove oldest entries if cache exceeds max_entries unique fingerprints."""
        fingerprints = {}
        for key, entry in self._index.items():
            fp = entry.get('fingerprint', '')
            if fp not in fingerprints:
                fingerprints[fp] = []
            fingerprints[fp].append((key, entry.get('last_accessed', 0)))
        
        if len(fingerprints) <= self.max_entries:
            return
        
        # Sort fingerprints by most recent access
        fp_access = [
            (fp, max(t for _, t in keys))
            for fp, keys in fingerprints.items()
        ]
        fp_access.sort(key=lambda x: x[1])
        
        # Remove oldest fingerprints until within limit
        to_remove = len(fingerprints) - self.max_entries
        for fp, _ in fp_access[:to_remove]:
            keys = [k for k, e in self._index.items() if e.get('fingerprint') == fp]
            for k in keys:
                # Delete cached files
                entry = self._index[k]
                for path_key in ('data_path', 'report_path'):
                    path = entry.get(path_key)
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                del self._index[k]
            
            logger.debug("Cache evicted fingerprint %s", fp[:12])
