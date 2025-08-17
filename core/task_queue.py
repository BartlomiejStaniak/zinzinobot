"""System kolejkowania zadań"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Dict, List
import uuid


class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    name: str
    task_type: str
    priority: TaskPriority
    data: Dict[str, Any]
    created_at: datetime
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    task_id: str = None

    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())


class TaskQueue:
    def __init__(self):
        self.tasks = []
        self.active_tasks = []

    async def add_task(self, task: Task) -> str:
        self.tasks.append(task)
        return task.task_id

    async def get_next_task(self) -> Optional[Task]:
        if self.tasks:
            task = self.tasks.pop(0)
            self.active_tasks.append(task)
            return task
        return None

    def is_empty(self) -> bool:
        return len(self.tasks) == 0

    def size(self) -> int:
        return len(self.tasks)

    def has_active_tasks(self) -> bool:
        return len(self.active_tasks) > 0

    async def get_all_tasks(self) -> List[Task]:
        return self.tasks + self.active_tasks

    async def reduce_priority_for_agent(self, agent_name: str):
        pass

    async def filter_critical_only(self):
        self.tasks = [t for t in self.tasks if t.priority == TaskPriority.CRITICAL]

    async def boost_critical_tasks(self):
        self.tasks.sort(key=lambda x: x.priority.value)