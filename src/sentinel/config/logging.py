"""Structured (JSON) logging configuration.

Single entry point: :func:`configure_logging`. Every component logs through
the standard ``logging`` module; secrets are never logged (we simply never
pass them to loggers — config summaries are redacted by design).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_RESERVED = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JsonLogFormatter(logging.Formatter):
    """One JSON object per line — grep/jq friendly and ingestable by any log stack."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "info", json_format: bool = True) -> None:
    """Configure the root logger once per process (idempotent)."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)  # logs to stderr; stdout stays machine-readable
    handler.setFormatter(
        JsonLogFormatter()
        if json_format
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root.addHandler(handler)
    # Third-party noise reduction
    for noisy in ("urllib3", "httpx", "httpcore", "mlflow", "gunicorn"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (structured fields go through :func:`log_event`)."""
    return logging.getLogger(name)


def log_event(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Emit a structured log event with additional typed fields."""
    safe = {k.replace(".", "_"): v for k, v in fields.items() if k not in _RESERVED}
    logger.log(level, message, extra=safe)
