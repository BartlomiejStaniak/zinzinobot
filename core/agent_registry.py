#!/usr/bin/env python3
"""
agent_registry.py - Rejestr agentów i system discovery
Plik: core/agent_registry.py
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

from agents.base_agent import BaseAgent, AgentCapability, AgentState, TaskPriority

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentRegistration:
    """Informacje o zarejestrowanym agencie"""
    agent_id: str
    agent_type: str
    capabilities: Dict[str, AgentCapability]
    current_state: AgentState
    performance_score: float
    last_heartbeat: datetime
    registration_time: datetime
    endpoint: Optional[str] = None  # For remote agents
    metadata: Dict[str, Any] = None


@dataclass
class CapabilityMatch:
    """Dopasowanie capability do wymagań"""
    agent_id: str
    capability_name: str
    confidence_score: float
    performance_score: float
    current_load: float  # 0.0 - 1.0
    match_score: float  # Overall match quality
    estimated_completion_time: timedelta


@dataclass
class CollaborationGroup:
    """Grupa agentów do współpracy"""
    group_id: str
    leader_agent_id: str
    member_agent_ids: List[str]
    shared_objective: str
    created_at: datetime
    expected_completion: datetime
    status: str  # 'active', 'completed', 'failed'


class AgentRegistry:
    """
    Centralny rejestr agentów - zarządza discovery, load balancing i coordination
    """

    def __init__(self, db_path: str = "data/agent_registry.db"):
        self.db_path = db_path

        # Aktywne agenty (local instances)
        self.active_agents: Dict[str, BaseAgent] = {}

        # Registry cache
        self.registered_agents: Dict[str, AgentRegistration] = {}
        self.capability_index: Dict[str, Set[str]] = defaultdict(set)  # capability -> agent_ids
        self.type_index: Dict[str, Set[str]] = defaultdict(set)  # agent_type -> agent_ids

        # Collaboration management
        self.collaboration_groups: Dict[str, CollaborationGroup] = {}

        # Load balancing
        self.agent_loads: Dict[str, float] = defaultdict(float)  # agent_id -> current_load

        # Discovery strategies
        self.discovery_strategies = {
            'best_performance': self._find_by_performance,
            'least_loaded': self._find_by_load,
            'highest_confidence': self._find_by_confidence,
            'fastest_completion': self._find_by_speed,
            'collaborative': self._find_collaborative_agents
        }

        # Configuration
        self.config = {
            'heartbeat_timeout': 300,  # 5 minutes
            'max_load_threshold': 0.8,  # 80% max load
            'collaboration_preference_weight': 0.3,
            'performance_weight': 0.4,
            'load_weight': 0.3
        }

        self._init_database()
        self._start_maintenance_tasks()

    def _init_database(self):
        """Inicjalizuje bazę danych registry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tabela registered agents
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS registered_agents (
                agent_id TEXT PRIMARY KEY,
                agent_type TEXT NOT NULL,
                capabilities TEXT NOT NULL,
                current_state TEXT NOT NULL,
                performance_score REAL NOT NULL,
                last_heartbeat TEXT NOT NULL,
                registration_time TEXT NOT NULL,
                endpoint TEXT,
                metadata TEXT,
                is_active BOOLEAN DEFAULT 1
            )
        ''')

        # Tabela capabilities mapping
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_capabilities (
                agent_id TEXT,
                capability_name TEXT,
                confidence_level REAL,
                resource_requirements TEXT,
                PRIMARY KEY (agent_id, capability_name),
                FOREIGN KEY (agent_id) REFERENCES registered_agents(agent_id)
            )
        ''')

        # Tabela collaboration groups
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collaboration_groups (
                group_id TEXT PRIMARY KEY,
                leader_agent_id TEXT,
                member_agent_ids TEXT,
                shared_objective TEXT,
                created_at TEXT,
                expected_completion TEXT,
                status TEXT,
                result TEXT
            )
        ''')

        # Tabela agent performance history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                performance_score REAL,
                tasks_completed INTEGER,
                success_rate REAL,
                average_completion_time REAL,
                timestamp TEXT,
                FOREIGN KEY (agent_id) REFERENCES registered_agents(agent_id)
            )
        ''')

        conn.commit()
        conn.close()

    def _start_maintenance_tasks(self):
        """Uruchamia zadania maintenance w tle"""
        asyncio.create_task(self._heartbeat_monitor())
        asyncio.create_task(self._performance_tracker())
        asyncio.create_task(self._load_balancer())

    async def register_agent(self, agent: BaseAgent, endpoint: Optional[str] = None) -> bool:
        """Rejestruje agenta w systemie"""
        try:
            registration = AgentRegistration(
                agent_id=agent.agent_id,
                agent_type=agent.agent_type,
                capabilities=agent.capabilities,
                current_state=agent.state,
                performance_score=agent.metrics.performance_score,
                last_heartbeat=datetime.now(),
                registration_time=datetime.now(),
                endpoint=endpoint,
                metadata={'version': '1.0', 'features': []}
            )

            # Zapisz do cache
            self.registered_agents[agent.agent_id] = registration
            self.active_agents[agent.agent_id] = agent

            # Indeksuj capabilities
            for cap_name in agent.capabilities.keys():
                self.capability_index[cap_name].add(agent.agent_id)

            # Indeksuj type
            self.type_index[agent.agent_type].add(agent.agent_id)

            # Zapisz do bazy danych
            await self._save_registration_to_db(registration)

            logger.info(f"✅ Agent {agent.agent_id} ({agent.agent_type}) registered successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to register agent {agent.agent_id}: {str(e)}")
            return False

    async def unregister_agent(self, agent_id: str) -> bool:
        """Wyrejestrowuje agenta"""
        try:
            if agent_id in self.registered_agents:
                registration = self.registered_agents[agent_id]

                # Usuń z indeksów
                for cap_name in registration.capabilities.keys():
                    self.capability_index[cap_name].discard(agent_id)

                self.type_index[registration.agent_type].discard(agent_id)

                # Usuń z cache
                del self.registered_agents[agent_id]
                if agent_id in self.active_agents:
                    del self.active_agents[agent_id]

                # Zaktualizuj w bazie danych
                await self._mark_agent_inactive(agent_id)

                logger.info(f"✅ Agent {agent_id} unregistered successfully")
                return True

            return False

        except Exception as e:
            logger.error(f"❌ Failed to unregister agent {agent_id}: {str(e)}")
            return False

    async def find_agents_by_capability(self, capability_name: str,
                                        strategy: str = 'best_performance',
                                        max_results: int = 5) -> List[CapabilityMatch]:
        """Znajduje agentów z określoną capability"""

        # Znajdź agentów z capability
        candidate_agent_ids = self.capability_index.get(capability_name, set())

        if not candidate_agent_ids:
            logger.warning(f"No agents found with capability: {capability_name}")
            return []

        # Filtruj aktywnych agentów
        active_candidates = [
            agent_id for agent_id in candidate_agent_ids
            if agent_id in self.registered_agents and
               self.registered_agents[agent_id].current_state in [AgentState.IDLE, AgentState.WORKING]
        ]

        if not active_candidates:
            logger.warning(f"No active agents found with capability: {capability_name}")
            return []

        # Zastosuj strategię discovery
        discovery_func = self.discovery_strategies.get(strategy, self._find_by_performance)
        matches = await discovery_func(active_candidates, capability_name)

        # Sortuj i zwróć top results
        matches.sort(key=lambda m: m.match_score, reverse=True)
        return matches[:max_results]

    async def find_agents_by_type(self, agent_type: str) -> List[str]:
        """Znajduje agentów określonego typu"""
        agent_ids = self.type_index.get(agent_type, set())

        # Zwróć tylko aktywnych
        return [
            agent_id for agent_id in agent_ids
            if agent_id in self.registered_agents and
               self.registered_agents[agent_id].current_state != AgentState.SHUTDOWN
        ]

    async def create_collaboration_group(self, leader_agent_id: str,
                                         member_agent_ids: List[str],
                                         shared_objective: str,
                                         expected_duration: timedelta) -> str:
        """Tworzy grupę współpracy agentów"""

        group_id = f"collab_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{leader_agent_id}"

        collaboration_group = CollaborationGroup(
            group_id=group_id,
            leader_agent_id=leader_agent_id,
            member_agent_ids=member_agent_ids,
            shared_objective=shared_objective,
            created_at=datetime.now(),
            expected_completion=datetime.now() + expected_duration,
            status='active'
        )

        self.collaboration_groups[group_id] = collaboration_group

        # Zapisz do bazy danych
        await self._save_collaboration_group(collaboration_group)

        logger.info(f"🤝 Collaboration group {group_id} created with {len(member_agent_ids) + 1} agents")
        return group_id

    async def get_optimal_agent_for_task(self, required_capabilities: List[str],
                                         task_priority: TaskPriority,
                                         estimated_duration: timedelta,
                                         collaboration_preferred: bool = False) -> Optional[str]:
        """Znajduje optymalnego agenta dla zadania"""

        # Znajdź agentów z wymaganymi capabilities
        candidate_scores = defaultdict(float)

        for capability in required_capabilities:
            matches = await self.find_agents_by_capability(capability, 'best_performance')
            for match in matches:
                # Weighted score based on capability importance
                weight = 1.0 / len(required_capabilities)
                candidate_scores[match.agent_id] += match.match_score * weight

        if not candidate_scores:
            return None

        # Znajdź najlepszego kandydata
        best_agent_id = max(candidate_scores, key=candidate_scores.get)

        # Sprawdź load balancing
        current_load = self.agent_loads.get(best_agent_id, 0.0)
        if current_load > self.config['max_load_threshold']:
            # Znajdź alternatywę z niższym load
            alternative_candidates = [
                (agent_id, score) for agent_id, score in candidate_scores.items()
                if self.agent_loads.get(agent_id, 0.0) <= self.config['max_load_threshold']
            ]

            if alternative_candidates:
                best_agent_id = max(alternative_candidates, key=lambda x: x[1])[0]

        return best_agent_id

    async def _find_by_performance(self, agent_ids: List[str], capability_name: str) -> List[CapabilityMatch]:
        """Strategia discovery: najlepsza wydajność"""
        matches = []

        for agent_id in agent_ids:
            registration = self.registered_agents[agent_id]
            capability = registration.capabilities.get(capability_name)

            if capability:
                current_load = self.agent_loads.get(agent_id, 0.0)

                # Oblicz match score based on performance
                performance_factor = registration.performance_score
                confidence_factor = capability.confidence_level
                load_factor = 1.0 - current_load  # Lower load = higher score

                match_score = (
                        performance_factor * self.config['performance_weight'] +
                        confidence_factor * 0.3 +
                        load_factor * self.config['load_weight']
                )

                estimated_time = self._estimate_completion_time(agent_id, current_load)

                match = CapabilityMatch(
                    agent_id=agent_id,
                    capability_name=capability_name,
                    confidence_score=capability.confidence_level,
                    performance_score=registration.performance_score,
                    current_load=current_load,
                    match_score=match_score,
                    estimated_completion_time=estimated_time
                )

                matches.append(match)

        return matches

    async def _find_by_load(self, agent_ids: List[str], capability_name: str) -> List[CapabilityMatch]:
        """Strategia discovery: najmniejsze obciążenie"""
        matches = []

        for agent_id in agent_ids:
            registration = self.registered_agents[agent_id]
            capability = registration.capabilities.get(capability_name)

            if capability:
                current_load = self.agent_loads.get(agent_id, 0.0)

                # Prioritize low load
                match_score = (1.0 - current_load) * 0.7 + capability.confidence_level * 0.3

                estimated_time = self._estimate_completion_time(agent_id, current_load)

                match = CapabilityMatch(
                    agent_id=agent_id,
                    capability_name=capability_name,
                    confidence_score=capability.confidence_level,
                    performance_score=registration.performance_score,
                    current_load=current_load,
                    match_score=match_score,
                    estimated_completion_time=estimated_time
                )

                matches.append(match)

        return matches

    async def _find_by_confidence(self, agent_ids: List[str], capability_name: str) -> List[CapabilityMatch]:
        """Strategia discovery: najwyższa pewność"""
        matches = []

        for agent_id in agent_ids:
            registration = self.registered_agents[agent_id]
            capability = registration.capabilities.get(capability_name)

            if capability:
                current_load = self.agent_loads.get(agent_id, 0.0)

                # Prioritize confidence
                match_score = capability.confidence_level * 0.8 + (1.0 - current_load) * 0.2

                estimated_time = self._estimate_completion_time(agent_id, current_load)

                match = CapabilityMatch(
                    agent_id=agent_id,
                    capability_name=capability_name,
                    confidence_score=capability.confidence_level,
                    performance_score=registration.performance_score,
                    current_load=current_load,
                    match_score=match_score,
                    estimated_completion_time=estimated_time
                )

                matches.append(match)

        return matches

    async def _find_by_speed(self, agent_ids: List[str], capability_name: str) -> List[CapabilityMatch]:
        """Strategia discovery: najszybsze wykonanie"""
        matches = []

        for agent_id in agent_ids:
            registration = self.registered_agents[agent_id]
            capability = registration.capabilities.get(capability_name)

            if capability and agent_id in self.active_agents:
                current_load = self.agent_loads.get(agent_id, 0.0)
                agent = self.active_agents[agent_id]

                # Use average completion time for speed estimate
                avg_time = agent.metrics.average_completion_time
                speed_factor = 1.0 / (avg_time + 1) if avg_time > 0 else 1.0  # +1 to avoid division by zero

                match_score = speed_factor * 0.6 + capability.confidence_level * 0.4

                estimated_time = self._estimate_completion_time(agent_id, current_load)

                match = CapabilityMatch(
                    agent_id=agent_id,
                    capability_name=capability_name,
                    confidence_score=capability.confidence_level,
                    performance_score=registration.performance_score,
                    current_load=current_load,
                    match_score=match_score,
                    estimated_completion_time=estimated_time
                )

                matches.append(match)

        return matches

    async def _find_collaborative_agents(self, agent_ids: List[str], capability_name: str) -> List[CapabilityMatch]:
        """Strategia discovery: agenci preferujący współpracę"""
        matches = []

        for agent_id in agent_ids:
            registration = self.registered_agents[agent_id]
            capability = registration.capabilities.get(capability_name)

            if capability and agent_id in self.active_agents:
                agent = self.active_agents[agent_id]
                current_load = self.agent_loads.get(agent_id, 0.0)

                # Bonus for agents with collaboration experience
                collaboration_factor = min(agent.metrics.collaboration_count * 0.1, 0.5)

                match_score = (
                        capability.confidence_level * 0.4 +
                        collaboration_factor * 0.4 +
                        (1.0 - current_load) * 0.2
                )

                estimated_time = self._estimate_completion_time(agent_id, current_load)

                match = CapabilityMatch(
                    agent_id=agent_id,
                    capability_name=capability_name,
                    confidence_score=capability.confidence_level,
                    performance_score=registration.performance_score,
                    current_load=current_load,
                    match_score=match_score,
                    estimated_completion_time=estimated_time
                )

                matches.append(match)

        return matches

    def _estimate_completion_time(self, agent_id: str, current_load: float) -> timedelta:
        """Szacuje czas wykonania zadania"""
        base_time = 60  # 60 seconds base time

        if agent_id in self.active_agents:
            agent = self.active_agents[agent_id]
            if agent.metrics.average_completion_time > 0:
                base_time = agent.metrics.average_completion_time

        # Adjust for current load
        adjusted_time = base_time * (1 + current_load)

        return timedelta(seconds=adjusted_time)

    async def update_agent_heartbeat(self, agent_id: str):
        """Aktualizuje heartbeat agenta"""
        if agent_id in self.registered_agents:
            self.registered_agents[agent_id].last_heartbeat = datetime.now()

            # Aktualizuj stan agenta
            if agent_id in self.active_agents:
                agent = self.active_agents[agent_id]
                self.registered_agents[agent_id].current_state = agent.state
                self.registered_agents[agent_id].performance_score = agent.metrics.performance_score

    async def update_agent_load(self, agent_id: str, load: float):
        """Aktualizuje obciążenie agenta"""
        self.agent_loads[agent_id] = max(0.0, min(1.0, load))  # Clamp to 0.0-1.0

    async def _heartbeat_monitor(self):
        """Monitoruje heartbeat agentów"""
        while True:
            try:
                current_time = datetime.now()
                timeout_threshold = current_time - timedelta(seconds=self.config['heartbeat_timeout'])

                # Sprawdź expired heartbeats
                expired_agents = []
                for agent_id, registration in self.registered_agents.items():
                    if registration.last_heartbeat < timeout_threshold:
                        expired_agents.append(agent_id)

                # Usuń expired agentów
                for agent_id in expired_agents:
                    logger.warning(f"⚠️ Agent {agent_id} heartbeat timeout - removing from registry")
                    await self.unregister_agent(agent_id)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {str(e)}")
                await asyncio.sleep(60)

    async def _performance_tracker(self):
        """Śledzi wydajność agentów"""
        while True:
            try:
                # Zapisz performance history dla aktywnych agentów
                for agent_id, agent in self.active_agents.items():
                    await self._save_performance_history(agent_id, agent)

                await asyncio.sleep(300)  # Every 5 minutes

            except Exception as e:
                logger.error(f"Error in performance tracker: {str(e)}")
                await asyncio.sleep(300)

    async def _load_balancer(self):
        """Load balancing logic"""
        while True:
            try:
                # Oblicz current load dla każdego agenta
                for agent_id, agent in self.active_agents.items():
                    # Load based on queue size and active tasks
                    queue_size = agent.task_queue.qsize()
                    active_tasks = len(agent.active_tasks)

                    # Normalize load (assume max 10 concurrent items)
                    load = min((queue_size + active_tasks) / 10.0, 1.0)
                    await self.update_agent_load(agent_id, load)

                await asyncio.sleep(30)  # Every 30 seconds

            except Exception as e:
                logger.error(f"Error in load balancer: {str(e)}")
                await asyncio.sleep(30)

    async def _save_registration_to_db(self, registration: AgentRegistration):
        """Zapisuje rejestrację do bazy danych"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Główna tabela
        cursor.execute('''
            INSERT OR REPLACE INTO registered_agents
            (agent_id, agent_type, capabilities, current_state, performance_score,
             last_heartbeat, registration_time, endpoint, metadata, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            registration.agent_id,
            registration.agent_type,
            json.dumps({name: asdict(cap) for name, cap in registration.capabilities.items()}),
            registration.current_state.value,
            registration.performance_score,
            registration.last_heartbeat.isoformat(),
            registration.registration_time.isoformat(),
            registration.endpoint,
            json.dumps(registration.metadata or {}),
            1
        ))

        # Capabilities mapping
        for cap_name, capability in registration.capabilities.items():
            cursor.execute('''
                INSERT OR REPLACE INTO agent_capabilities
                (agent_id, capability_name, confidence_level, resource_requirements)
                VALUES (?, ?, ?, ?)
            ''', (
                registration.agent_id,
                cap_name,
                capability.confidence_level,
                json.dumps(capability.resource_requirements)
            ))

        conn.commit()
        conn.close()

    async def _mark_agent_inactive(self, agent_id: str):
        """Oznacza agenta jako nieaktywnego"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE registered_agents SET is_active = 0 WHERE agent_id = ?
        ''', (agent_id,))

        conn.commit()
        conn.close()

    async def _save_collaboration_group(self, group: CollaborationGroup):
        """Zapisuje grupę współpracy"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO collaboration_groups
            (group_id, leader_agent_id, member_agent_ids, shared_objective,
             created_at, expected_completion, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            group.group_id,
            group.leader_agent_id,
            json.dumps(group.member_agent_ids),
            group.shared_objective,
            group.created_at.isoformat(),
            group.expected_completion.isoformat(),
            group.status
        ))

        conn.commit()
        conn.close()

    async def _save_performance_history(self, agent_id: str, agent: BaseAgent):
        """Zapisuje historię wydajności"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO agent_performance_history
            (agent_id, performance_score, tasks_completed, success_rate,
             average_completion_time, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            agent_id,
            agent.metrics.performance_score,
            agent.metrics.tasks_completed,
            agent.metrics.success_rate,
            agent.metrics.average_completion_time,
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

    def get_registry_status(self) -> Dict[str, Any]:
        """Zwraca status registry"""
        active_count = len(self.active_agents)
        registered_count = len(self.registered_agents)

        # Statystyki typów agentów
        type_stats = defaultdict(int)
        for registration in self.registered_agents.values():
            type_stats[registration.agent_type] += 1

        # Statystyki capabilities
        capability_stats = defaultdict(int)
        for cap_name, agent_ids in self.capability_index.items():
            capability_stats[cap_name] = len(agent_ids)

        # Load statistics
        total_load = sum(self.agent_loads.values())
        avg_load = total_load / len(self.agent_loads) if self.agent_loads else 0.0

        return {
            'registry_stats': {
                'active_agents': active_count,
                'registered_agents': registered_count,
                'collaboration_groups': len(self.collaboration_groups),
                'average_load': avg_load
            },
            'agent_types': dict(type_stats),
            'capabilities': dict(capability_stats),
            'load_distribution': dict(self.agent_loads),
            'last_updated': datetime.now().isoformat()
        }


# Test funkcji
async def test_agent_registry():
    """Test agent registry"""

    # Import dla testu
    from agents.base_agent import BaseAgent, AgentCapability, TaskPriority, AgentTask

    # Test agent class
    class TestAgent(BaseAgent):
        async def initialize(self) -> bool:
            return True

        async def process_task(self, task: AgentTask) -> Dict[str, Any]:
            await asyncio.sleep(0.1)
            return {'status': 'completed'}

        def get_specialized_capabilities(self) -> Dict[str, AgentCapability]:
            return {
                'content_creation': AgentCapability(
                    name='content_creation',
                    description='Creates social media content',
                    input_types=['knowledge_points', 'requirements'],
                    output_types=['content', 'posts'],
                    confidence_level=0.9,
                    resource_requirements={'memory': 'medium', 'cpu': 'high'}
                )
            }

    print("🏢 Testing Agent Registry...")
    print("=" * 50)

    # Stwórz registry
    registry = AgentRegistry()

    # Test 1: Rejestracja agentów
    agent1 = TestAgent('content_agent_1', 'content_creator')
    agent2 = TestAgent('content_agent_2', 'content_creator')

    success1 = await registry.register_agent(agent1)
    success2 = await registry.register_agent(agent2)

    print(f"Agent 1 registration: {success1}")
    print(f"Agent 2 registration: {success2}")

    # Test 2: Discovery by capability
    matches = await registry.find_agents_by_capability('content_creation', 'best_performance')

    print(f"\n🔍 Found {len(matches)} agents with content_creation capability:")
    for match in matches:
        print(f"  - {match.agent_id}: score={match.match_score:.2f}, load={match.current_load:.2f}")

    # Test 3: Find optimal agent for task
    optimal_agent = await registry.get_optimal_agent_for_task(
        required_capabilities=['content_creation'],
        task_priority=TaskPriority.HIGH,
        estimated_duration=timedelta(minutes=5)
    )

    print(f"\n🎯 Optimal agent for task: {optimal_agent}")

    # Test 4: Registry status
    status = registry.get_registry_status()
    print(f"\n📊 Registry Status:")
    print(f"  Active agents: {status['registry_stats']['active_agents']}")
    print(f"  Agent types: {status['agent_types']}")
    print(f"  Capabilities: {status['capabilities']}")


if __name__ == "__main__":
    asyncio.run(test_agent_registry())