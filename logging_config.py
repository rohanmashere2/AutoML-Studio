"""
AutoML Studio — Structured Logging Configuration
JSON-formatted logs with request IDs for traceability.
"""

import logging
import uuid
from flask import request, g


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record):
        request_id = getattr(record, 'request_id', 'N/A')
        msg = record.getMessage()
        return (
            f'{{"time":"{self.formatTime(record)}",'
            f'"level":"{record.levelname}",'
            f'"request_id":"{request_id}",'
            f'"logger":"{record.name}",'
            f'"message":"{msg}"}}'
        )


class RequestIDFilter(logging.Filter):
    """Inject the current request ID into every log record."""

    def filter(self, record):
        try:
            record.request_id = getattr(g, 'request_id', 'N/A')
        except RuntimeError:
            record.request_id = 'N/A'
        return True


def setup_logging(app, level=logging.INFO):
    """Configure structured JSON logging and per-request ID tracking."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    handler.addFilter(RequestIDFilter())

    app.logger.handlers.clear()
    app.logger.addHandler(handler)
    app.logger.setLevel(level)

    @app.before_request
    def _set_request_id():
        g.request_id = request.headers.get('X-Request-ID', str(uuid.uuid4())[:8])

    @app.after_request
    def _add_request_id_header(response):
        rid = getattr(g, 'request_id', None)
        if rid:
            response.headers['X-Request-ID'] = rid
        return response

    return app
