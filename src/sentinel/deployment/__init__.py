"""Deployment: canary routing, promotion gate, controller, rollback, audit."""

from sentinel.deployment.audit import AuditEntry, AuditLog
from sentinel.deployment.canary import (
    CanaryError,
    CanaryManager,
    CanaryRouter,
    RoutingState,
    read_routing_state,
    write_routing_state,
)
from sentinel.deployment.controller import ControllerState, Decision, DeploymentController
from sentinel.deployment.promotion import DeployDecision, PromotionManager
from sentinel.deployment.rollback import RollbackManager, RollbackResult

__all__ = [
    "AuditEntry",
    "AuditLog",
    "CanaryError",
    "CanaryManager",
    "CanaryRouter",
    "RoutingState",
    "read_routing_state",
    "write_routing_state",
    "ControllerState",
    "Decision",
    "DeploymentController",
    "DeployDecision",
    "PromotionManager",
    "RollbackManager",
    "RollbackResult",
]
