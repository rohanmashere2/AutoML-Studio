"""
Optional Firebase sync helpers for experiment metadata.

Supports either a service account JSON path or an inline JSON string via
environment variables:
- FIREBASE_CREDENTIALS_PATH
- FIREBASE_CREDENTIALS_JSON

If FIREBASE_DATABASE_URL is set, best scores are written to the Realtime Database
under /experiments/{experiment_id}.
"""

from __future__ import annotations

import json
import os
import logging
from functools import lru_cache
from typing import Any, Optional

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_db():
    credentials_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
    credentials_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
    database_url = os.getenv("FIREBASE_DATABASE_URL")

    if not database_url:
        logger.debug("Firebase disabled: FIREBASE_DATABASE_URL not set")
        return None

    try:
        import firebase_admin
        from firebase_admin import credentials, db
    except ImportError as e:
        logger.error(f"Firebase import failed: {e}")
        return None

    try:
        if not firebase_admin._apps:
            cred = None
            if credentials_path and os.path.exists(credentials_path):
                logger.info(f"Using Firebase service account from {credentials_path}")
                cred = credentials.Certificate(credentials_path)
            elif credentials_json:
                logger.info("Using Firebase service account from FIREBASE_CREDENTIALS_JSON env var")
                cred = credentials.Certificate(json.loads(credentials_json))
            else:
                logger.warning("Firebase configured but no credentials found (FIREBASE_CREDENTIALS_PATH or FIREBASE_CREDENTIALS_JSON)")
                return None

            firebase_admin.initialize_app(cred, {"databaseURL": database_url})
            logger.info("Firebase initialized successfully")

        return db
    except Exception as e:
        logger.error(f"Firebase initialization failed: {e}")
        return None


def save_best_score(experiment_id: str, best_score: Optional[float], best_model: Optional[str] = None) -> bool:
    if not experiment_id:
        return False

    db = _get_db()
    if db is None:
        logger.debug(f"Firebase not available, skipping save for experiment {experiment_id}")
        return False

    payload: dict[str, Any] = {
        "best_score": best_score,
        "best_model": best_model,
    }
    try:
        logger.info(f"Saving to Firebase: experiments/{experiment_id} = {payload}")
        db.reference(f"experiments/{experiment_id}").update(payload)
        logger.info(f"Successfully saved to Firebase: {experiment_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to save to Firebase for {experiment_id}: {e}")
        return False


def get_best_score(experiment_id: str) -> Optional[float]:
    if not experiment_id:
        return None

    db = _get_db()
    if db is None:
        logger.debug(f"Firebase not available, skipping get for experiment {experiment_id}")
        return None

    try:
        logger.debug(f"Fetching from Firebase: experiments/{experiment_id}")
        data = db.reference(f"experiments/{experiment_id}").get() or {}
        score = data.get("best_score")
        if score is None:
            logger.debug(f"No best_score found in Firebase for {experiment_id}")
            return None
        logger.debug(f"Retrieved best_score from Firebase: {experiment_id} = {score}")
        return float(score)
    except Exception as e:
        logger.error(f"Failed to get from Firebase for {experiment_id}: {e}")
        return None
