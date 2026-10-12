"""Configuration package: typed settings + structured logging."""

from sentinel.config.logging import configure_logging, get_logger, log_event
from sentinel.config.settings import ConfigError, SentinelSettings

__all__ = [
    "SentinelSettings",
    "ConfigError",
    "configure_logging",
    "get_logger",
    "log_event",
]
