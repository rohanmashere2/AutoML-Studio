"""
AutoML Studio — Thread-Safe Session Manager
Replaces the unbounded plain dict with a TTL-based cache and threading locks.
"""

import gc
import uuid
import threading
from cachetools import TTLCache


class SessionManager:
    """Thread-safe session store with automatic TTL expiry and memory cleanup."""

    def __init__(self, maxsize=1000, ttl=7200):
        """
        Args:
            maxsize: Maximum number of concurrent sessions.
            ttl: Time-to-live in seconds (default 2 hours).
        """
        self._cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._lock = threading.Lock()

    def __contains__(self, session_id):
        with self._lock:
            return session_id in self._cache

    def get(self, session_id, default=None):
        """Retrieve a session by ID. Returns *default* if missing or expired."""
        with self._lock:
            return self._cache.get(session_id, default)

    def set(self, session_id, session):
        """Store or update a session."""
        with self._lock:
            self._cache[session_id] = session

    def delete(self, session_id):
        """Remove a session and explicitly free large objects."""
        with self._lock:
            session = self._cache.pop(session_id, None)
        if session is not None:
            # Explicitly clear heavy attributes to help GC
            for attr in ('original_df', 'cleaned_df', 'transformed_df',
                         'X_train', 'X_test', 'y_train', 'y_test',
                         'trained_models', 'best_model', 'explainability',
                         'diagnostics'):
                try:
                    setattr(session, attr, None)
                except Exception:
                    pass
            del session
            gc.collect()

    def keys(self):
        with self._lock:
            return list(self._cache.keys())

    def values(self):
        with self._lock:
            return list(self._cache.values())

    def items(self):
        with self._lock:
            return list(self._cache.items())

    def __len__(self):
        with self._lock:
            return len(self._cache)

    @staticmethod
    def generate_session_id():
        """Generate a full 128-bit UUID (not truncated)."""
        return str(uuid.uuid4())
