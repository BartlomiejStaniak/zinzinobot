"""
Master Controller - Główny kontroler systemu AI dla Zinzino
Koordynuje pracę wszystkich agentów i zarządza przepływem zadań
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json

from core.agent_base import BaseAgent, AgentStatus
from core.task_queue import TaskQueue, Task, TaskPriority, TaskStatus
from core.memory_manager import MemoryManager
from core.facebook_client import FacebookClient
from agents.content_creator import ContentCreatorAgent
from agents.engagement_bot import EngagementBotAgent
from agents.zinzino_specialist import ZinzinoSpecialistAgent


class SystemMode(Enum):
    """Tryby pracy systemu"""
    STARTUP = "startup"
    NORMAL = "normal"
    HIGH_ACTIVITY = "high_activity"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"


@dataclass
class SystemHealth:
    """Stan zdrowia systemu"""
    overall_status: str
    active_agents: int
    pending_tasks: int
    completed_tasks_today: int
    error_rate: float
    last_update: datetime
    memory_usage: float
    response_time: float


class MasterController:
    """
    Główny kontroler systemu - orkiestra wszystkich agentów AI

    Odpowiedzialności:
    - Koordynacja pracy agentów
    - Zarządzanie zadaniami
    - Monitorowanie stanu systemu
    - Optymalizacja wydajności
    - Obsługa sytuacji awaryjnych
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Komponenty systemu
        self.task_queue = TaskQueue()
        self.memory_manager = MemoryManager(config.get('memory', {}))
        self.facebook_client = FacebookClient(config.get('facebook', {}))

        # Agenci
        self.agents: Dict[str, BaseAgent] = {}
        self._initialize_agents()

        # Stan systemu
        self.mode = SystemMode.STARTUP
        self.is_running = False
        self.start_time = None
        self.health_metrics = {}

        # Statystyki
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'posts_created': 0,
            'interactions_handled': 0,
            'uptime': 0
        }

        # Harmonogram zadań
        self.scheduled_tasks = []
        self.recurring_tasks = self._setup_recurring_tasks()

        self.logger.info("Master Controller zainicjalizowany")

    def _initialize_agents(self):
        """Inicjalizacja wszystkich agentów"""
        try:
            # Content Creator Agent
            self.agents['content_creator'] = ContentCreatorAgent(
                name="ContentCreator",
                config=self.config.get('agents', {}).get('content_creator', {}),
                facebook_client=self.facebook_client,
                memory_manager=self.memory_manager
            )

            # Engagement Bot Agent
            self.agents['engagement_bot'] = EngagementBotAgent(
                name="EngagementBot",
                config=self.config.get('agents', {}).get('engagement_bot', {}),
                facebook_client=self.facebook_client,
                memory_manager=self.memory_manager
            )

            # Zinzino Specialist Agent
            self.agents['zinzino_specialist'] = ZinzinoSpecialistAgent(
                name="ZinzinoSpecialist",
                config=self.config.get('agents', {}).get('zinzino_specialist', {}),
                memory_manager=self.memory_manager
            )

            self.logger.info(f"Zainicjalizowano {len(self.agents)} agentów")

        except Exception as e:
            self.logger.error(f"Błąd podczas inicjalizacji agentów: {e}")
            raise

    def _setup_recurring_tasks(self) -> List[Dict]:
        """Konfiguracja zadań cyklicznych"""
        return [
            {
                'name': 'daily_content_creation',
                'agent': 'content_creator',
                'task_type': 'create_daily_posts',
                'schedule': 'daily',
                'time': '09:00',
                'priority': TaskPriority.HIGH
            },
            {
                'name': 'engagement_monitoring',
                'agent': 'engagement_bot',
                'task_type': 'monitor_interactions',
                'schedule': 'hourly',
                'priority': TaskPriority.MEDIUM
            },
            {
                'name': 'health_check',
                'agent': 'master',
                'task_type': 'system_health_check',
                'schedule': 'every_15_minutes',
                'priority': TaskPriority.LOW
            },
            {
                'name': 'memory_cleanup',
                'agent': 'master',
                'task_type': 'cleanup_memory',
                'schedule': 'daily',
                'time': '02:00',
                'priority': TaskPriority.LOW
            }
        ]

    async def start(self):
        """Uruchomienie systemu"""
        self.logger.info("🚀 Uruchamianie Master Controller...")

        try:
            self.start_time = datetime.now()
            self.is_running = True

            # Uruchomienie agentów
            await self._start_agents()

            # Przejście do trybu normalnego
            await self._transition_to_mode(SystemMode.NORMAL)

            # Uruchomienie głównej pętli
            await self._main_loop()

        except Exception as e:
            self.logger.error(f"Błąd podczas uruchamiania: {e}")
            await self.emergency_shutdown()
            raise

    async def _start_agents(self):
        """Uruchomienie wszystkich agentów"""
        for agent_name, agent in self.agents.items():
            try:
                await agent.start()
                self.logger.info(f"✅ Agent {agent_name} uruchomiony")
            except Exception as e:
                self.logger.error(f"❌ Błąd uruchamiania agenta {agent_name}: {e}")
                raise

    async def _main_loop(self):
        """Główna pętla systemu"""
        self.logger.info("🔄 Uruchamianie głównej pętli systemu")

        while self.is_running:
            try:
                # Przetwarzanie zadań
                await self._process_tasks()

                # Monitorowanie agentów
                await self._monitor_agents()

                # Sprawdzenie stanu systemu
                await self._check_system_health()

                # Harmonogram zadań cyklicznych
                await self._process_scheduled_tasks()

                # Optymalizacja wydajności
                await self._optimize_performance()

                # Krótka przerwa
                await asyncio.sleep(5)

            except Exception as e:
                self.logger.error(f"Błąd w głównej pętli: {e}")
                await asyncio.sleep(10)

    async def _process_tasks(self):
        """Przetwarzanie zadań z kolejki"""
        while not self.task_queue.is_empty():
            task = await self.task_queue.get_next_task()
            if task:
                await self._execute_task(task)

    async def _execute_task(self, task: Task):
        """Wykonanie pojedynczego zadania"""
        try:
            self.logger.info(f"📋 Wykonywanie zadania: {task.name}")

            # Przypisanie zadania odpowiedniemu agentowi
            agent = self._get_agent_for_task(task)
            if not agent:
                task.status = TaskStatus.FAILED
                task.error = "Brak odpowiedniego agenta"
                return

            # Wykonanie zadania
            result = await agent.execute_task(task)

            if result.success:
                task.status = TaskStatus.COMPLETED
                task.result = result.data
                self.stats['tasks_completed'] += 1
                self.logger.info(f"✅ Zadanie {task.name} zakończone sukcesem")
            else:
                task.status = TaskStatus.FAILED
                task.error = result.error
                self.stats['tasks_failed'] += 1
                self.logger.error(f"❌ Zadanie {task.name} zakończone błędem: {result.error}")

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.stats['tasks_failed'] += 1
            self.logger.error(f"❌ Błąd wykonania zadania {task.name}: {e}")

    def _get_agent_for_task(self, task: Task) -> Optional[BaseAgent]:
        """Przypisanie zadania odpowiedniemu agentowi"""
        agent_mapping = {
            'create_post': 'content_creator',
            'create_daily_posts': 'content_creator',
            'respond_to_comment': 'engagement_bot',
            'monitor_interactions': 'engagement_bot',
            'answer_question': 'zinzino_specialist',
            'product_recommendation': 'zinzino_specialist'
        }

        agent_name = agent_mapping.get(task.task_type)
        return self.agents.get(agent_name) if agent_name else None

    async def _monitor_agents(self):
        """Monitorowanie stanu agentów"""
        for agent_name, agent in self.agents.items():
            try:
                status = await agent.get_status()

                if status.status == AgentStatus.ERROR:
                    self.logger.warning(f"⚠️ Agent {agent_name} w stanie błędu")
                    await self._handle_agent_error(agent_name, agent)

                elif status.status == AgentStatus.OVERLOADED:
                    self.logger.warning(f"⚠️ Agent {agent_name} przeciążony")
                    await self._handle_agent_overload(agent_name, agent)

            except Exception as e:
                self.logger.error(f"Błąd monitorowania agenta {agent_name}: {e}")

    async def _handle_agent_error(self, agent_name: str, agent: BaseAgent):
        """Obsługa błędu agenta"""
        try:
            # Próba restartu agenta
            await agent.restart()
            self.logger.info(f"🔄 Agent {agent_name} zrestartowany")
        except Exception as e:
            self.logger.error(f"Nie udało się zrestartować agenta {agent_name}: {e}")
            # Przejście do trybu awaryjnego jeśli krytyczny agent
            if agent_name in ['content_creator', 'engagement_bot']:
                await self._transition_to_mode(SystemMode.EMERGENCY)

    async def _handle_agent_overload(self, agent_name: str, agent: BaseAgent):
        """Obsługa przeciążenia agenta"""
        # Zmniejszenie obciążenia
        await self.task_queue.reduce_priority_for_agent(agent_name)

        # Przejście do trybu wysokiej aktywności
        if self.mode != SystemMode.HIGH_ACTIVITY:
            await self._transition_to_mode(SystemMode.HIGH_ACTIVITY)

    async def _check_system_health(self):
        """Sprawdzenie stanu zdrowia systemu"""
        try:
            health = SystemHealth(
                overall_status="healthy",
                active_agents=len([a for a in self.agents.values() if a.is_active]),
                pending_tasks=self.task_queue.size(),
                completed_tasks_today=self.stats['tasks_completed'],
                error_rate=self._calculate_error_rate(),
                last_update=datetime.now(),
                memory_usage=await self.memory_manager.get_usage_stats(),
                response_time=await self._measure_response_time()
            )

            self.health_metrics = health.__dict__

            # Sprawdzenie krytycznych metryk
            if health.error_rate > 0.1:  # 10% błędów
                self.logger.warning(f"⚠️ Wysoki współczynnik błędów: {health.error_rate:.2%}")

            if health.memory_usage > 0.8:  # 80% pamięci
                self.logger.warning(f"⚠️ Wysokie użycie pamięci: {health.memory_usage:.2%}")
                await self.memory_manager.cleanup()

        except Exception as e:
            self.logger.error(f"Błąd sprawdzania zdrowia systemu: {e}")

    def _calculate_error_rate(self) -> float:
        """Obliczenie współczynnika błędów"""
        total_tasks = self.stats['tasks_completed'] + self.stats['tasks_failed']
        if total_tasks == 0:
            return 0.0
        return self.stats['tasks_failed'] / total_tasks

    async def _measure_response_time(self) -> float:
        """Pomiar czasu odpowiedzi systemu"""
        start_time = datetime.now()
        # Symulacja prostego zadania
        await asyncio.sleep(0.001)
        end_time = datetime.now()
        return (end_time - start_time).total_seconds()

    async def _process_scheduled_tasks(self):
        """Przetwarzanie zadań zaplanowanych"""
        current_time = datetime.now()

        for task_config in self.recurring_tasks:
            if self._should_execute_scheduled_task(task_config, current_time):
                await self._create_scheduled_task(task_config)

    def _should_execute_scheduled_task(self, task_config: Dict, current_time: datetime) -> bool:
        """Sprawdzenie czy zadanie powinno być wykonane"""
        schedule = task_config['schedule']

        if schedule == 'hourly':
            return current_time.minute == 0
        elif schedule == 'daily':
            time_str = task_config.get('time', '09:00')
            hour, minute = map(int, time_str.split(':'))
            return current_time.hour == hour and current_time.minute == minute
        elif schedule == 'every_15_minutes':
            return current_time.minute % 15 == 0

        return False

    async def _create_scheduled_task(self, task_config: Dict):
        """Utworzenie zadania zaplanowanego"""
        task = Task(
            name=task_config['name'],
            task_type=task_config['task_type'],
            priority=task_config['priority'],
            data=task_config.get('data', {}),
            created_at=datetime.now()
        )

        await self.task_queue.add_task(task)
        self.logger.info(f"📅 Dodano zadanie zaplanowane: {task.name}")

    async def _optimize_performance(self):
        """Optymalizacja wydajności systemu"""
        # Dynamiczne dostosowanie priorytetów zadań
        await self._adjust_task_priorities()

        # Optymalizacja pamięci
        if self.health_metrics.get('memory_usage', 0) > 0.7:
            await self.memory_manager.optimize()

        # Dostosowanie trybu pracy
        await self._adjust_system_mode()

    async def _adjust_task_priorities(self):
        """Dostosowanie priorytetów zadań"""
        queue_size = self.task_queue.size()

        if queue_size > 100:  # Duża kolejka
            # Podwyższenie priorytetów krytycznych zadań
            await self.task_queue.boost_critical_tasks()
        elif queue_size < 10:  # Mała kolejka
            # Dodanie zadań o niskim priorytecie
            await self._add_maintenance_tasks()

    async def _add_maintenance_tasks(self):
        """Dodanie zadań konserwacyjnych"""
        maintenance_tasks = [
            {
                'name': 'cleanup_old_data',
                'task_type': 'maintenance',
                'priority': TaskPriority.LOW
            },
            {
                'name': 'optimize_database',
                'task_type': 'maintenance',
                'priority': TaskPriority.LOW
            }
        ]

        for task_data in maintenance_tasks:
            task = Task(**task_data, created_at=datetime.now())
            await self.task_queue.add_task(task)

    async def _adjust_system_mode(self):
        """Dostosowanie trybu pracy systemu"""
        error_rate = self._calculate_error_rate()
        queue_size = self.task_queue.size()

        if error_rate > 0.2:  # 20% błędów
            await self._transition_to_mode(SystemMode.EMERGENCY)
        elif queue_size > 200:  # Bardzo duża kolejka
            await self._transition_to_mode(SystemMode.HIGH_ACTIVITY)
        elif queue_size < 50 and error_rate < 0.05:  # Normalna sytuacja
            await self._transition_to_mode(SystemMode.NORMAL)

    async def _transition_to_mode(self, new_mode: SystemMode):
        """Przejście do nowego trybu pracy"""
        if self.mode == new_mode:
            return

        self.logger.info(f"🔄 Przejście z trybu {self.mode.value} do {new_mode.value}")

        old_mode = self.mode
        self.mode = new_mode

        # Dostosowanie konfiguracji dla nowego trybu
        await self._apply_mode_configuration(new_mode, old_mode)

    async def _apply_mode_configuration(self, new_mode: SystemMode, old_mode: SystemMode):
        """Zastosowanie konfiguracji dla nowego trybu"""
        if new_mode == SystemMode.HIGH_ACTIVITY:
            # Zwiększenie częstotliwości przetwarzania
            for agent in self.agents.values():
                await agent.set_processing_speed('high')

        elif new_mode == SystemMode.EMERGENCY:
            # Tryb awaryjny - tylko krytyczne zadania
            await self.task_queue.filter_critical_only()

            # Powiadomienie o trybie awaryjnym
            self.logger.critical("🚨 System w trybie awaryjnym!")

        elif new_mode == SystemMode.NORMAL:
            # Powrót do normalnej pracy
            for agent in self.agents.values():
                await agent.set_processing_speed('normal')

    async def add_task(self, task_data: Dict[str, Any]) -> str:
        """Dodanie zadania do kolejki"""
        task = Task(
            name=task_data.get('name', 'unnamed_task'),
            task_type=task_data['task_type'],
            priority=TaskPriority(task_data.get('priority', 'medium')),
            data=task_data.get('data', {}),
            created_at=datetime.now()
        )

        task_id = await self.task_queue.add_task(task)
        self.logger.info(f"➕ Dodano zadanie: {task.name}")

        return task_id

    async def get_system_status(self) -> Dict[str, Any]:
        """Pobranie statusu systemu"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        return {
            'mode': self.mode.value,
            'is_running': self.is_running,
            'uptime': uptime,
            'agents': {
                name: {
                    'status': (await agent.get_status()).status.value,
                    'tasks_completed': (await agent.get_status()).tasks_completed,
                    'is_active': agent.is_active
                }
                for name, agent in self.agents.items()
            },
            'queue_size': self.task_queue.size(),
            'stats': self.stats,
            'health': self.health_metrics
        }

    async def emergency_shutdown(self):
        """Awaryjne wyłączenie systemu"""
        self.logger.critical("🚨 AWARYJNE WYŁĄCZENIE SYSTEMU")

        self.is_running = False

        # Zatrzymanie wszystkich agentów
        for agent_name, agent in self.agents.items():
            try:
                await agent.stop()
                self.logger.info(f"🛑 Agent {agent_name} zatrzymany")
            except Exception as e:
                self.logger.error(f"Błąd zatrzymywania agenta {agent_name}: {e}")

        # Zapis stanu do odzyskania
        await self._save_state_for_recovery()

        self.logger.info("🛑 System zatrzymany")

    async def _save_state_for_recovery(self):
        """Zapis stanu systemu do odzyskania"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'mode': self.mode.value,
                'stats': self.stats,
                'pending_tasks': await self.task_queue.get_all_tasks(),
                'agent_states': {}
            }

            # Zapis stanu agentów
            for agent_name, agent in self.agents.items():
                try:
                    state['agent_states'][agent_name] = await agent.get_state()
                except Exception as e:
                    self.logger.error(f"Błąd zapisu stanu agenta {agent_name}: {e}")

            # Zapis do pliku
            with open('system_state_backup.json', 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)

            self.logger.info("💾 Stan systemu zapisany do odzyskania")

        except Exception as e:
            self.logger.error(f"Błąd zapisu stanu systemu: {e}")

    async def graceful_shutdown(self):
        """Eleganckie wyłączenie systemu"""
        self.logger.info("🛑 Rozpoczęcie eleganckiego wyłączenia systemu")

        self.is_running = False

        # Dokończenie bieżących zadań
        await self._finish_current_tasks()

        # Zatrzymanie agentów
        for agent_name, agent in self.agents.items():
            await agent.stop()
            self.logger.info(f"✅ Agent {agent_name} zatrzymany")

        # Zapis końcowych statystyk
        await self._save_final_stats()

        self.logger.info("✅ System zatrzymany elegancko")

    async def _finish_current_tasks(self):
        """Dokończenie bieżących zadań"""
        self.logger.info("⏳ Oczekiwanie na dokończenie bieżących zadań...")

        timeout = 300  # 5 minut
        start_time = datetime.now()

        while self.task_queue.has_active_tasks():
            if (datetime.now() - start_time).total_seconds() > timeout:
                self.logger.warning("⏰ Timeout oczekiwania na zadania")
                break

            await asyncio.sleep(5)

        self.logger.info("✅ Wszystkie zadania zakończone")

    async def _save_final_stats(self):
        """Zapis końcowych statystyk"""
        try:
            final_stats = {
                'session_end': datetime.now().isoformat(),
                'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                'final_stats': self.stats,
                'final_health': self.health_metrics
            }

            with open('session_stats.json', 'w', encoding='utf-8') as f:
                json.dump(final_stats, f, indent=2, ensure_ascii=False)

            self.logger.info("📊 Końcowe statystyki zapisane")

        except Exception as e:
            self.logger.error(f"Błąd zapisu końcowych statystyk: {e}")


# Funkcja pomocnicza do uruchomienia systemu
async def run_master_controller(config_path: str = 'config.json'):
    """Uruchomienie Master Controller z konfiguracją"""
    try:
        # Wczytanie konfiguracji
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Utworzenie i uruchomienie kontrolera
        controller = MasterController(config)
        await controller.start()

    except KeyboardInterrupt:
        print("\n🛑 Otrzymano sygnał przerwania")
        await controller.graceful_shutdown()
    except Exception as e:
        print(f"❌ Błąd krytyczny: {e}")
        if 'controller' in locals():
            await controller.emergency_shutdown()


if __name__ == "__main__":
    # Uruchomienie systemu
    asyncio.run(run_master_controller())