"""Tests for auth.py — Firebase token verification and login_required decorator."""

import pytest
from unittest.mock import patch, MagicMock
from flask import Flask, g


def create_test_app():
    """Create a minimal Flask app with auth wired up."""
    app = Flask(__name__)
    app.secret_key = 'test-secret'
    app.config['TESTING'] = True

    from auth import login_required, verify_firebase_token, _get_firebase_app

    # Public route
    @app.route('/health')
    def health():
        return {'status': 'ok'}

    # Protected route using decorator
    @app.route('/api/test')
    @login_required
    def protected():
        return {'user': g.user.get('uid', 'unknown')}

    # before_request hook (same as app.py uses)
    @app.before_request
    def enforce_auth():
        from flask import request, jsonify
        path = request.path
        if path in {'/', '/health'}:
            return None
        if not path.startswith('/api/'):
            return None
        if _get_firebase_app() is None:
            g.user = {'uid': 'anonymous', 'email': 'anonymous@local'}
            return None
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authentication required'}), 401
        id_token = auth_header[7:]
        if not id_token:
            return jsonify({'error': 'Empty token'}), 401
        claims = verify_firebase_token(id_token)
        if claims is None:
            return jsonify({'error': 'Invalid token'}), 401
        g.user = claims

    @app.route('/api/data')
    def data_route():
        return {'uid': g.user.get('uid', 'none')}

    return app


class TestAuthNoFirebase:
    """Test behavior when Firebase Admin SDK is NOT configured."""

    def test_public_route_accessible(self):
        app = create_test_app()
        with app.test_client() as c:
            resp = c.get('/health')
            assert resp.status_code == 200
            assert resp.get_json()['status'] == 'ok'

    @patch('auth._get_firebase_app', return_value=None)
    def test_api_allowed_anonymous_when_firebase_not_configured(self, mock_fb):
        app = create_test_app()
        with app.test_client() as c:
            resp = c.get('/api/data')
            assert resp.status_code == 200
            data = resp.get_json()
            assert data['uid'] == 'anonymous'

    @patch('auth._get_firebase_app', return_value=None)
    def test_decorator_allows_anonymous_when_firebase_not_configured(self, mock_fb):
        app = create_test_app()
        with app.test_client() as c:
            resp = c.get('/api/test')
            assert resp.status_code == 200
            data = resp.get_json()
            assert data['user'] == 'anonymous'


class TestAuthWithFirebase:
    """Test behavior when Firebase Admin SDK IS configured."""

    @patch('auth._get_firebase_app', return_value=MagicMock())
    def test_missing_auth_header_returns_401(self, mock_fb):
        app = create_test_app()
        with app.test_client() as c:
            resp = c.get('/api/data')
            assert resp.status_code == 401
            assert 'Authentication required' in resp.get_json()['error']

    @patch('auth._get_firebase_app', return_value=MagicMock())
    def test_malformed_auth_header_returns_401(self, mock_fb):
        app = create_test_app()
        with app.test_client() as c:
            resp = c.get('/api/data', headers={'Authorization': 'Basic abc'})
            assert resp.status_code == 401

    @patch('auth._get_firebase_app', return_value=MagicMock())
    def test_empty_token_returns_401(self, mock_fb):
        app = create_test_app()
        with app.test_client() as c:
            resp = c.get('/api/data', headers={'Authorization': 'Bearer '})
            assert resp.status_code == 401

    @patch('auth.verify_firebase_token', return_value=None)
    @patch('auth._get_firebase_app', return_value=MagicMock())
    def test_invalid_token_returns_401(self, mock_fb, mock_verify):
        app = create_test_app()
        with app.test_client() as c:
            resp = c.get('/api/data', headers={'Authorization': 'Bearer bad-token'})
            assert resp.status_code == 401

    @patch('auth.verify_firebase_token', return_value={'uid': 'user123', 'email': 'test@example.com'})
    @patch('auth._get_firebase_app', return_value=MagicMock())
    def test_valid_token_allows_access(self, mock_fb, mock_verify):
        app = create_test_app()
        with app.test_client() as c:
            resp = c.get('/api/data', headers={'Authorization': 'Bearer valid-token-xyz'})
            assert resp.status_code == 200
            data = resp.get_json()
            assert data['uid'] == 'user123'

    @patch('auth.verify_firebase_token', return_value={'uid': 'user456', 'email': 'user@test.com'})
    @patch('auth._get_firebase_app', return_value=MagicMock())
    def test_decorator_with_valid_token(self, mock_fb, mock_verify):
        app = create_test_app()
        with app.test_client() as c:
            resp = c.get('/api/test', headers={'Authorization': 'Bearer valid-token'})
            assert resp.status_code == 200
            assert resp.get_json()['user'] == 'user456'

    @patch('auth._get_firebase_app', return_value=MagicMock())
    def test_health_endpoint_skips_auth(self, mock_fb):
        """Public routes should never require auth."""
        app = create_test_app()
        with app.test_client() as c:
            resp = c.get('/health')
            assert resp.status_code == 200
