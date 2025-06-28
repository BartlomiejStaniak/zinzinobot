#!/usr/bin/env python3
"""
base_agent.py - Bazowa klasa dla wszystkich agentów
Plik: agents/base_agent.py
"""

import asyncio
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Stany agenta"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    WORKING = "working"
    COLLABORATING = "collaborating"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class TaskPriority(Enum):
    """Priorytety zadań"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class CollaborationType(Enum):
    """Typy współpracy między agentami"""
    REQUEST_RESPONSE = "request_response"
    DELEGATION = "delegation"
    COORDINATION = "coordination"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    PARALLEL_EXECUTION = "parallel_execution"


@dataclass
class AgentCapability:
    """Reprezentuje zdolność agenta"""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    confidence_level: float  # 0.0 - 1.0
    resource_requirements: Dict[str, Any]
    collaboration_compatible: bool = True


@dataclass
class AgentTask:
    """Zadanie dla agenta"""
    task_id: str
    task_type: str
    description: str
    input_data: Dict[str, Any]
    priority: TaskPriority
    deadline: Optional[datetime] = None
    requester_agent_id: Optional[str] = None
    collaboration_context: Optional[Dict] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class AgentMetrics:
    """Metryki wydajności agenta"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_completion_time: float = 0.0
    success_rate: float = 1.0
    collaboration_count: int = 0
    knowledge_contributions: int = 0
    last_activity: Optional[datetime] = None
    performance_score: float = 0.5  # Aktualizowane przez Learning Engine


@dataclass
class CollaborationRequest:
    """Żądanie współpracy między agentami"""
    request_id: str
    from_agent_id: str
    to_agent_id: str
    collaboration_type: CollaborationType
    task_context: Dict[str, Any]
    expected_capability: str
    deadline: Optional[datetime] = None
    priority: TaskPriority = TaskPriority.MEDIUM


class BaseAgent(ABC):
    """
    Bazowa klasa dla wszystkich agentów w systemie
    """

    def __init__(self, agent_id: str, agent_type: str, config: Optional[Dict] = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or {}

        # Stan agenta
        self.state = AgentState.INITIALIZING
        self.created_at = datetime.now()
        self.last_heartbeat = datetime.now()

        # Zadania i kolejki
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.active_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: List[AgentTask] = []

        # Capabilities i collaboration
        self.capabilities: Dict[str, AgentCapability] = {}
        self.collaboration_partners: Set[str] = set()
        self.active_collaborations: Dict[str, CollaborationRequest] = {}

        # Performance i learning
        self.metrics = AgentMetrics()
        self.learning_data: Dict[str, Any] = {}
        self.adaptation_history: List[Dict] = []

        # Communication
        self.message_handlers: Dict[str, Callable] = {}
        self.event_listeners: Dict[str, List[Callable]] = {}

        # Inicjalizacja
        self._setup_default_capabilities()
        self._setup_message_handlers()
        self._setup_event_listeners()

        logger.info(f"🤖 Agent {self.agent_id} ({self.agent_type}) initialized")

    @abstractmethod
    async def initialize(self) -> bool:
        """Inicjalizuje agenta - do implementacji w klasach potomnych"""
        pass

    @abstractmethod
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Przetwarza zadanie - do implementacji w klasach potomnych"""
        pass

    @abstractmethod
    def get_specialized_capabilities(self) -> Dict[str, AgentCapability]:
        """Zwraca specjalistyczne zdolności agenta"""
        pass

    def _setup_default_capabilities(self):
        """Ustawia podstawowe zdolności każdego agenta"""
        base_capabilities = {
            'communication': AgentCapability(
                name='communication',
                description='Basic inter-agent communication',
                input_types=['message', 'request'],
                output_types=['response', 'acknowledgment'],
                confidence_level=1.0,
                resource_requirements={'memory': 'low', 'cpu': 'low'}
            ),
            'task_management': AgentCapability(
                name='task_management',
                description='Task queue and execution management',
                input_types=['task', 'command'],
                output_types=['status', 'result'],
                confidence_level=1.0,
                resource_requirements={'memory': 'medium', 'cpu': 'medium'}
            ),
            'collaboration': AgentCapability(
                name='collaboration',
                description='Ability to collaborate with other agents',
                input_types=['collaboration_request', 'knowledge_share'],
                output_types=['collaboration_response', 'shared_knowledge'],
                confidence_level=0.8,
                resource_requirements={'memory': 'medium', 'cpu': 'low'}
            )
        }

        # Dodaj specjalistyczne capabilities
        specialized = self.get_specialized_capabilities()
        self.capabilities = {**base_capabilities, **specialized}

    def _setup_message_handlers(self):
        """Ustawia handlery wiadomości"""
        self.message_handlers = {
            'collaboration_request': self._handle_collaboration_request,
            'task_assignment': self._handle_task_assignment,
            'knowledge_query': self._handle_knowledge_query,
            'status_request': self._handle_status_request,
            'adaptation_signal': self._handle_adaptation_signal
        }

    def _setup_event_listeners(self):
        """Ustawia listenery eventów"""
        self.event_listeners = {
            'task_completed': [],
            'collaboration_started': [],
            'knowledge_updated': [],
            'performance_changed': [],
            'error_occurred': []
        }

    async def start(self):
        """Uruchamia agenta"""
        logger.info(f"🚀 Starting agent {self.agent_id}")

        # Inicjalizacja
        if not await self.initialize():
            logger.error(f"❌ Failed to initialize agent {self.agent_id}")
            self.state = AgentState.ERROR
            return False

        self.state = AgentState.IDLE

        # Uruchom główną pętlę
        await self._main_loop()

        return True

    async def _main_loop(self):
        """Główna pętla agenta"""
        logger.info(f"🔄 Agent {self.agent_id} main loop started")

        while self.state != AgentState.SHUTDOWN:
            try:
                # Aktualizuj heartbeat
                self.last_heartbeat = datetime.now()

                # Sprawdź czy są zadania do wykonania
                if not self.task_queue.empty():
                    await self._process_next_task()

                # Sprawdź collaboration requests
                await self._process_collaboration_requests()

                # Wykonaj maintenance tasks
                await self._perform_maintenance()

                # Krótka przerwa aby nie blokować systemu
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"❌ Error in agent {self.agent_id} main loop: {str(e)}")
                await self._handle_error(e)

    async def _process_next_task(self):
        """Przetwarza następne zadanie z kolejki"""
        try:
            # Pobierz zadanie z kolejki
            task = await asyncio.wait_for(self.task_queue.get(), timeout=0.1)

            logger.info(f"🎯 Agent {self.agent_id} processing task: {task.task_type}")

            # Zmień stan na WORKING
            self.state = AgentState.WORKING
            task.started_at = datetime.now()
            self.active_tasks[task.task_id] = task

            # Przetwórz zadanie
            result = await self.process_task(task)

            # Zapisz rezultat
            task.completed_at = datetime.now()
            task.result = result

            # Aktualizuj metryki
            self._update_metrics(task, success=True)

            # Przenieś do completed tasks
            self.completed_tasks.append(task)
            del self.active_tasks[task.task_id]

            # Emit event
            await self._emit_event('task_completed', {'task': task, 'result': result})

            # Powrót do IDLE
            self.state = AgentState.IDLE

            logger.info(f"✅ Task {task.task_id} completed by agent {self.agent_id}")

        except asyncio.TimeoutError:
            # Brak zadań - kontynuuj
            pass
        except Exception as e:
            logger.error(f"❌ Error processing task in agent {self.agent_id}: {str(e)}")
            await self._handle_task_error(task, e)

    async def _process_collaboration_requests(self):
        """Przetwarza żądania współpracy"""
        # Sprawdź pending collaboration requests
        for req_id, collaboration in list(self.active_collaborations.items()):
            if collaboration.deadline and datetime.now() > collaboration.deadline:
                # Timeout - usuń collaboration
                logger.warning(f"⏰ Collaboration {req_id} timed out")
                del self.active_collaborations[req_id]

    async def _perform_maintenance(self):
        """Wykonuje zadania maintenance"""
        current_time = datetime.now()

        # Czyść stare completed tasks (starsze niż 1 godzina)
        cutoff_time = current_time - timedelta(hours=1)
        self.completed_tasks = [
            task for task in self.completed_tasks
            if task.completed_at and task.completed_at > cutoff_time
        ]

        # Aktualizuj performance score (co 5 minut)
        if not hasattr(self, '_last_performance_update') or \
                (current_time - self._last_performance_update).seconds > 300:
            await self._update_performance_score()
            self._last_performance_update = current_time

    async def add_task(self, task: AgentTask):
        """Dodaje zadanie do kolejki"""
        await self.task_queue.put(task)
        logger.info(f"📝 Task {task.task_id} added to agent {self.agent_id} queue")

    async def request_collaboration(self, target_agent_id: str,
                                    collaboration_type: CollaborationType,
                                    task_context: Dict[str, Any],
                                    expected_capability: str) -> str:
        """Wysyła żądanie współpracy do innego agenta"""

        collaboration_request = CollaborationRequest(
            request_id=str(uuid.uuid4()),
            from_agent_id=self.agent_id,
            to_agent_id=target_agent_id,
            collaboration_type=collaboration_type,
            task_context=task_context,
            expected_capability=expected_capability,
            deadline=datetime.now() + timedelta(minutes=30)  # 30 min deadline
        )

        self.active_collaborations[collaboration_request.request_id] = collaboration_request

        # Wyślij żądanie przez system komunikacji
        await self._send_collaboration_request(collaboration_request)

        logger.info(f"🤝 Collaboration request sent from {self.agent_id} to {target_agent_id}")
        return collaboration_request.request_id

    async def _handle_collaboration_request(self, request: CollaborationRequest) -> Dict[str, Any]:
        """Obsługuje żądanie współpracy"""
        logger.info(f"🤝 Agent {self.agent_id} received collaboration request from {request.from_agent_id}")

        # Sprawdź czy agent ma wymaganą capability
        if request.expected_capability not in self.capabilities:
            return {
                'status': 'rejected',
                'reason': f'Capability {request.expected_capability} not available'
            }

        # Sprawdź czy agent nie jest zbyt zajęty
        if len(self.active_tasks) > 5:  # Max 5 concurrent tasks
            return {
                'status': 'rejected',
                'reason': 'Agent too busy'
            }

        # Zaakceptuj współpracę
        self.collaboration_partners.add(request.from_agent_id)
        self.state = AgentState.COLLABORATING

        # Emit event
        await self._emit_event('collaboration_started', {'request': request})

        return {
            'status': 'accepted',
            'agent_id': self.agent_id,
            'estimated_completion': datetime.now() + timedelta(minutes=10)
        }

    async def _handle_task_assignment(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Obsługuje przypisanie zadania"""
        task = AgentTask(
            task_id=task_data.get('task_id', str(uuid.uuid4())),
            task_type=task_data['task_type'],
            description=task_data['description'],
            input_data=task_data['input_data'],
            priority=TaskPriority(task_data.get('priority', TaskPriority.MEDIUM.value)),
            requester_agent_id=task_data.get('requester_agent_id')
        )

        await self.add_task(task)

        return {
            'status': 'accepted',
            'task_id': task.task_id,
            'estimated_completion': datetime.now() + timedelta(minutes=5)
        }

    async def _handle_knowledge_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Obsługuje zapytania o wiedzę"""
        query_type = query.get('type', 'general')

        if query_type == 'capabilities':
            return {
                'agent_id': self.agent_id,
                'capabilities': {name: {
                    'name': cap.name,
                    'description': cap.description,
                    'confidence_level': cap.confidence_level
                } for name, cap in self.capabilities.items()}
            }

        elif query_type == 'performance':
            return {
                'agent_id': self.agent_id,
                'metrics': {
                    'tasks_completed': self.metrics.tasks_completed,
                    'success_rate': self.metrics.success_rate,
                    'performance_score': self.metrics.performance_score
                }
            }

        return {'status': 'unknown_query_type'}

    async def _handle_status_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Obsługuje żądania statusu"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'state': self.state.value,
            'active_tasks': len(self.active_tasks),
            'queue_length': self.task_queue.qsize(),
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'performance_score': self.metrics.performance_score
        }

    async def _handle_adaptation_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Obsługuje sygnały adaptacji z Learning Engine"""
        adaptation_type = signal.get('type')

        if adaptation_type == 'priority_adjustment':
            # Dostosuj priorytety zadań
            adjustments = signal.get('adjustments', {})
            await self._apply_priority_adjustments(adjustments)

        elif adaptation_type == 'capability_boost':
            # Zwiększ confidence level dla określonych capabilities
            capabilities = signal.get('capabilities', [])
            boost_factor = signal.get('boost_factor', 1.1)
            await self._boost_capabilities(capabilities, boost_factor)

        elif adaptation_type == 'collaboration_preference':
            # Dostosuj preferencje współpracy
            preferences = signal.get('preferences', {})
            await self._update_collaboration_preferences(preferences)

        # Zapisz historię adaptacji
        self.adaptation_history.append({
            'timestamp': datetime.now().isoformat(),
            'signal': signal,
            'applied': True
        })

        return {'status': 'adaptation_applied', 'signal_type': adaptation_type}

    async def _apply_priority_adjustments(self, adjustments: Dict[str, float]):
        """Stosuje dostosowania priorytetów"""
        # Implementacja dostosowań priorytetów
        logger.info(f"🎯 Agent {self.agent_id} applying priority adjustments: {adjustments}")

    async def _boost_capabilities(self, capabilities: List[str], boost_factor: float):
        """Zwiększa confidence level dla określonych capabilities"""
        for cap_name in capabilities:
            if cap_name in self.capabilities:
                old_confidence = self.capabilities[cap_name].confidence_level
                new_confidence = min(old_confidence * boost_factor, 1.0)
                self.capabilities[cap_name].confidence_level = new_confidence

                logger.info(f"📈 Boosted {cap_name} confidence: {old_confidence:.2f} → {new_confidence:.2f}")

    async def _update_collaboration_preferences(self, preferences: Dict[str, Any]):
        """Aktualizuje preferencje współpracy"""
        # Implementacja aktualizacji preferencji
        logger.info(f"🤝 Agent {self.agent_id} updating collaboration preferences")

    def _update_metrics(self, task: AgentTask, success: bool):
        """Aktualizuje metryki agenta"""
        if success:
            self.metrics.tasks_completed += 1
        else:
            self.metrics.tasks_failed += 1

        # Oblicz success rate
        total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
        self.metrics.success_rate = self.metrics.tasks_completed / total_tasks if total_tasks > 0 else 1.0

        # Oblicz średni czas wykonania
        if task.started_at and task.completed_at:
            execution_time = (task.completed_at - task.started_at).total_seconds()
            if self.metrics.average_completion_time == 0:
                self.metrics.average_completion_time = execution_time
            else:
                # Moving average
                self.metrics.average_completion_time = (
                        self.metrics.average_completion_time * 0.8 + execution_time * 0.2
                )

        self.metrics.last_activity = datetime.now()

    async def _update_performance_score(self):
        """Aktualizuje performance score agenta"""
        # Oblicz score na podstawie metryk
        base_score = self.metrics.success_rate

        # Bonus za szybkość wykonania (jeśli < 30 sekund)
        if self.metrics.average_completion_time > 0 and self.metrics.average_completion_time < 30:
            speed_bonus = 0.1
        else:
            speed_bonus = 0.0

        # Bonus za współpracę
        collaboration_bonus = min(self.metrics.collaboration_count * 0.01, 0.1)

        # Oblicz final score
        self.metrics.performance_score = min(base_score + speed_bonus + collaboration_bonus, 1.0)

    async def _emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emituje event do listenerów"""
        if event_type in self.event_listeners:
            for listener in self.event_listeners[event_type]:
                try:
                    await listener(event_data)
                except Exception as e:
                    logger.error(f"Error in event listener: {str(e)}")

    async def _send_collaboration_request(self, request: CollaborationRequest):
        """Wysyła żądanie współpracy (implementacja zależy od systemu komunikacji)"""
        # Placeholder - będzie implementowane w AgentMessenger
        logger.info(f"📤 Sending collaboration request: {request.request_id}")

    async def _handle_error(self, error: Exception):
        """Obsługuje błędy agenta"""
        logger.error(f"❌ Agent {self.agent_id} error: {str(error)}")

        # Emit error event
        await self._emit_event('error_occurred', {
            'agent_id': self.agent_id,
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        })

        # Przywróć stan IDLE jeśli to możliwe
        if self.state != AgentState.SHUTDOWN:
            self.state = AgentState.IDLE

    async def _handle_task_error(self, task: AgentTask, error: Exception):
        """Obsługuje błąd w zadaniu"""
        task.error = str(error)
        task.completed_at = datetime.now()

        # Aktualizuj metryki
        self._update_metrics(task, success=False)

        # Przenieś do completed tasks
        self.completed_tasks.append(task)
        if task.task_id in self.active_tasks:
            del self.active_tasks[task.task_id]

        logger.error(f"❌ Task {task.task_id} failed in agent {self.agent_id}: {str(error)}")

    async def shutdown(self):
        """Zamyka agenta gracefully"""
        logger.info(f"🛑 Shutting down agent {self.agent_id}")

        self.state = AgentState.SHUTDOWN

        # Dokończ aktywne zadania
        while self.active_tasks:
            await asyncio.sleep(0.1)

        logger.info(f"✅ Agent {self.agent_id} shutdown complete")

    def get_status_summary(self) -> Dict[str, Any]:
        """Zwraca podsumowanie statusu agenta"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'state': self.state.value,
            'capabilities': list(self.capabilities.keys()),
            'performance_score': self.metrics.performance_score,
            'tasks_completed': self.metrics.tasks_completed,
            'success_rate': self.metrics.success_rate,
            'active_tasks': len(self.active_tasks),
            'queue_length': self.task_queue.qsize(),
            'collaboration_partners': len(self.collaboration_partners),
            'uptime_hours': (datetime.now() - self.created_at).total_seconds() / 3600
        }


# Test funkcji
async def test_base_agent():
    """Test bazowej klasy agenta"""

    # Przykładowy agent do testów
    class TestAgent(BaseAgent):
        async def initialize(self) -> bool:
            return True

        async def process_task(self, task: AgentTask) -> Dict[str, Any]:
            # Symuluj przetwarzanie zadania
            await asyncio.sleep(0.1)
            return {'status': 'completed', 'result': f'Processed {task.task_type}'}

        def get_specialized_capabilities(self) -> Dict[str, AgentCapability]:
            return {
                'test_capability': AgentCapability(
                    name='test_capability',
                    description='Test capability for demo',
                    input_types=['test_input'],
                    output_types=['test_output'],
                    confidence_level=0.9,
                    resource_requirements={'memory': 'low'}
                )
            }

    print("🤖 Testing Base Agent...")
    print("=" * 50)

    # Stwórz test agenta
    agent = TestAgent('test_agent_1', 'test_agent')

    # Test 1: Sprawdź inicjalizację
    print(f"Agent ID: {agent.agent_id}")
    print(f"Agent Type: {agent.agent_type}")
    print(f"State: {agent.state.value}")
    print(f"Capabilities: {list(agent.capabilities.keys())}")

    # Test 2: Dodaj zadanie
    test_task = AgentTask(
        task_id='test_task_1',
        task_type='test_processing',
        description='Test task for agent',
        input_data={'test_data': 'sample'},
        priority=TaskPriority.MEDIUM
    )

    await agent.add_task(test_task)
    print(f"Task queue length: {agent.task_queue.qsize()}")

    # Test 3: Sprawdź status
    status = agent.get_status_summary()
    print(f"\n📊 Agent Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(test_base_agent())