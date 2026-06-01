"""
AutoML Studio — Firebase Authentication Middleware
Verifies Firebase ID tokens on all protected API routes.
"""

import os
import functools
import logging
from flask import request, jsonify, g

logger = logging.getLogger(__name__)

# Lazy-init Firebase Admin SDK
_firebase_app = None


def _get_firebase_app():
    """Initialize Firebase Admin SDK once (lazy singleton)."""
    global _firebase_app
    if _firebase_app is not None:
        return _firebase_app

    try:
        import firebase_admin
        from firebase_admin import credentials

        # Check if already initialized
        try:
            _firebase_app = firebase_admin.get_app()
            return _firebase_app
        except ValueError:
            pass

        # Try environment-based credentials first
        project_id = os.getenv('FIREBASE_PROJECT_ID')
        private_key = os.getenv('FIREBASE_PRIVATE_KEY')
        client_email = os.getenv('FIREBASE_CLIENT_EMAIL')

        if project_id and private_key and client_email:
            # Make parsing robust against Windows CMD quotes and literal newlines
            clean_key = private_key.strip().strip('"').strip("'").replace('\\n', '\n')
            
            cred = credentials.Certificate({
                "type": "service_account",
                "project_id": project_id.strip().strip('"').strip("'"),
                "private_key": clean_key,
                "client_email": client_email,
                "token_uri": "https://oauth2.googleapis.com/token",
            })
            _firebase_app = firebase_admin.initialize_app(cred)
        elif os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            # Use default credentials file
            _firebase_app = firebase_admin.initialize_app()
        else:
            # Fallback: try default init (works on GCP)
            _firebase_app = firebase_admin.initialize_app()

        logger.info("Firebase Admin SDK initialized successfully")
        return _firebase_app

    except Exception as e:
        logger.warning(f"Firebase Admin SDK init failed: {e}. Auth will be disabled.")
        return None


def verify_firebase_token(id_token):
    """Verify a Firebase ID token and return the decoded claims.

    Args:
        id_token: The Firebase ID token string from the client.

    Returns:
        dict with user claims (uid, email, etc.) or None if invalid.
    """
    app = _get_firebase_app()
    if app is None:
        return None

    try:
        from firebase_admin import auth
        decoded = auth.verify_id_token(id_token, app)
        return decoded
    except Exception as e:
        logger.debug(f"Token verification failed: {e}")
        return None


def login_required(f):
    """Decorator that enforces Firebase ID token authentication.

    Expects the client to send:
        Authorization: Bearer <firebase_id_token>

    On success, sets g.user with the decoded token claims (uid, email, etc.).
    On failure, returns 401.
    """
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        # Check if auth is enabled (Firebase Admin SDK available)
        if _get_firebase_app() is None:
            # Firebase not configured — allow request but log warning
            g.user = {'uid': 'anonymous', 'email': 'anonymous@local'}
            return f(*args, **kwargs)

        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({
                'error': 'Authentication required',
                'message': 'Missing or malformed Authorization header. Expected: Bearer <token>'
            }), 401

        id_token = auth_header[7:]  # Strip "Bearer "
        if not id_token:
            return jsonify({'error': 'Authentication required', 'message': 'Empty token'}), 401

        claims = verify_firebase_token(id_token)
        if claims is None:
            return jsonify({
                'error': 'Invalid or expired token',
                'message': 'Please sign in again'
            }), 401

        # Store user info for downstream handlers
        g.user = claims
        return f(*args, **kwargs)

    return decorated


def get_current_user_uid():
    """Get the current authenticated user's UID from g.user."""
    user = getattr(g, 'user', None)
    if user:
        return user.get('uid')
    return None
