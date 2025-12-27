"""Loop condition tasks."""
from .exit_code import BreakOnCondition, ContinueOnCondition, ContinueOnFailCondition
from .diagnostic import DiagnosticCondition

__all__ = [
    "BreakOnCondition",
    "ContinueOnCondition",
    "ContinueOnFailCondition",
    "DiagnosticCondition",
]
