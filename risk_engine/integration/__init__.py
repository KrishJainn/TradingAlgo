"""Integration interfaces for the risk engine."""

from .agent_interface import AgentInterface, AgentSignal
from .coach_interface import CoachInterface, CoachDecision

__all__ = ["AgentInterface", "AgentSignal", "CoachInterface", "CoachDecision"]
