"""Model registry: versioned artifacts + lifecycle state machine."""

from sentinel.registry.manager import (
    InvalidTransitionError,
    ModelNotFoundError,
    ModelRegistry,
    RegistryError,
)
from sentinel.registry.metadata import (
    ALLOWED_TRANSITIONS,
    ModelStatus,
    ModelVersion,
    RegistryIndex,
    StatusChange,
)

__all__ = [
    "ModelRegistry",
    "RegistryError",
    "InvalidTransitionError",
    "ModelNotFoundError",
    "ModelStatus",
    "ModelVersion",
    "RegistryIndex",
    "StatusChange",
    "ALLOWED_TRANSITIONS",
]
