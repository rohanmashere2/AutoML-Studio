"""
Tests for session_manager.py — TTL, thread safety, cleanup.
"""

import time
import threading
import pytest

from session_manager import SessionManager


class TestSessionManager:

    def test_basic_set_get(self):
        sm = SessionManager(maxsize=10, ttl=60)
        sm.set('abc', {'data': 42})
        assert sm.get('abc') == {'data': 42}

    def test_missing_key_returns_default(self):
        sm = SessionManager(maxsize=10, ttl=60)
        assert sm.get('nonexistent') is None
        assert sm.get('nonexistent', 'fallback') == 'fallback'

    def test_contains(self):
        sm = SessionManager(maxsize=10, ttl=60)
        sm.set('key1', 'val1')
        assert 'key1' in sm
        assert 'key2' not in sm

    def test_delete(self):
        sm = SessionManager(maxsize=10, ttl=60)
        sm.set('del_me', 'value')
        sm.delete('del_me')
        assert sm.get('del_me') is None

    def test_ttl_expiry(self):
        sm = SessionManager(maxsize=10, ttl=1)  # 1 second TTL
        sm.set('expires', 'soon')
        assert sm.get('expires') == 'soon'
        time.sleep(1.5)
        assert sm.get('expires') is None

    def test_maxsize_eviction(self):
        sm = SessionManager(maxsize=3, ttl=60)
        sm.set('a', 1)
        sm.set('b', 2)
        sm.set('c', 3)
        sm.set('d', 4)  # Should evict oldest
        assert len(sm) <= 3

    def test_thread_safety(self):
        sm = SessionManager(maxsize=1000, ttl=60)
        errors = []

        def writer(thread_id):
            try:
                for i in range(100):
                    sm.set(f't{thread_id}_{i}', i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_generate_session_id_is_full_uuid(self):
        sid = SessionManager.generate_session_id()
        assert len(sid) == 36  # Full UUID with hyphens
        assert sid.count('-') == 4
