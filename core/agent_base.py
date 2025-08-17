# 1. core/agent_base.py
"""Bazowa klasa agenta - musisz utworzyć lub poprawić importy"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


class AgentStatus(Enum):
    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"
    OVERLOADED = "overloaded"


@dataclass
class AgentResult:
    success: bool
    data: Dict[str, Any] = None
    error: str = None


class BaseAgent:
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.is_active = True

    async def start(self):
        pass

    async def stop(self):
        pass

    async def restart(self):
        await self.stop()
        await self.start()

    async def execute_task(self, task):
        raise NotImplementedError

    async def get_status(self):
        return AgentStatus.IDLE

    async def set_processing_speed(self, speed: str):
        pass

    async def get_state(self):
        return {}
