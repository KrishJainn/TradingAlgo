"""
AQTIS Base Agent.

Abstract base class for all AQTIS agents.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

from aqtis.memory.memory_layer import MemoryLayer
from aqtis.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for AQTIS agents.

    All agents share access to the MemoryLayer and an LLM provider.
    """

    def __init__(
        self,
        name: str,
        memory: MemoryLayer,
        llm: Optional[LLMProvider] = None,
    ):
        self.name = name
        self.memory = memory
        self.llm = llm
        self._status = "initialized"
        self._last_run: Optional[datetime] = None
        self._run_count = 0
        self.logger = logging.getLogger(f"aqtis.agents.{name}")

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary function.

        Args:
            context: Input context for the agent.

        Returns:
            Result dictionary.
        """
        ...

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent with logging and error handling.

        Wraps execute() with standard pre/post processing.
        """
        self._status = "running"
        self._run_count += 1
        start_time = datetime.now()
        self.logger.info(f"Agent {self.name} starting (run #{self._run_count})")

        try:
            result = self.execute(context)
            self._status = "completed"
            self._last_run = datetime.now()
            elapsed = (self._last_run - start_time).total_seconds()
            self.logger.info(f"Agent {self.name} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            self._status = "error"
            self.logger.error(f"Agent {self.name} failed: {e}", exc_info=True)
            return {"error": str(e), "agent": self.name}

    def get_status(self) -> Dict:
        """Get agent status."""
        return {
            "name": self.name,
            "status": self._status,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "run_count": self._run_count,
            "has_llm": self.llm is not None,
        }
