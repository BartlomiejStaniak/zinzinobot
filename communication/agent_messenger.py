#!/usr/bin/env python3
"""
agent_messenger.py - System komunikacji między agentami
Plik: communication/agent_messenger.py
"""

import asyncio
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from agents.base_agent import BaseAgent, CollaborationRequest, CollaborationType
from core.agent_registry import AgentRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Typy wiadomości między agentami"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    COLLABORATION_REQUEST = "collaboration_request"
    COLLABORATION_RESPONSE = "collaboration_response"
    KNOWLEDGE_SHARE = "knowledge_share"
    TASK_ASSIGNMENT = "task_assignment"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"


class MessagePriority(Enum):
    """Priorytety wiadomości"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class DeliveryMode(Enum):
    """Tryby dostarczania wiadomości"""
    DIRECT = "direct"  # Bezpośrednio do agenta
    QUEUE = "queue"  # Przez kolejkę wiadomości
    BROADCAST = "broadcast"  # Do wszystkich agentów
    MULTICAST = "multicast"  # Do grupy agentów


@dataclass
class Message:
    """Wiadomość między agentami"""
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    subject: str
    payload: Dict[str, Any]
    priority: MessagePriority
    delivery_mode: DeliveryMode
    created_at: datetime
    expires_at: Optional[datetime] = None
    requires_response: bool = False
    conversation_id: Optional[str] = None
    delivery_attempts: int = 0
    max_delivery_attempts: int = 3
    delivered_at: Optional[datetime] = None
    response_timeout: Optional[timedelta] = None


@dataclass
class MessageResponse:
    """Odpowiedź na wiadomość"""
    response_id: str
    original_message_id: str
    sender_id: str
    payload: Dict[str, Any]
    created_at: datetime
    success: bool
    error_message: Optional[str] = None


@dataclass
class Conversation:
    """Konwersacja między agentami"""
    conversation_id: str
    participants: List[str]
    initiated_by: str
    topic: str
    started_at: datetime
    last_activity: datetime
    status: str  # 'active', 'completed', 'timeout'
    messages: List[str]  # Message IDs


class AgentMessenger:
    """
    System komunikacji między agentami - obsługuje routing, delivery, i conversation management
    """

    def __init__(self, agent_registry: AgentRegistry):
        self.agent_registry = agent_registry

        # Message queues per agent
        self.agent_queues: Dict[str, asyncio.Queue] = {}

        # Message storage
        self.messages: Dict[str, Message] = {}
        self.responses: Dict[str, MessageResponse] = {}
        self.conversations: Dict[str, Conversation] = {}

        # Delivery tracking
        self.pending_deliveries: Dict[str, Message] = {}
        self.delivery_confirmations: Dict[str, datetime] = {}

        # Message handlers
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.response_handlers: Dict[str, Callable] = {}  # message_id -> handler

        # Broadcasting
        self.broadcast_subscribers: Dict[str, List[str]] = {}  # topic -> agent_ids

        # Configuration
        self.config = {
            'default_message_ttl': 3600,  # 1 hour
            'max_queue_size': 1000,
            'delivery_retry_interval': 30,  # 30 seconds
            'conversation_timeout': 1800,  # 30 minutes
            'max_broadcast_recipients': 100
        }

        self._setup_default_handlers()
        self._start_background_tasks()

    def _setup_default_handlers(self):
        """Ustawia domyślne handlery wiadomości"""
        self.message_handlers = {
            MessageType.COLLABORATION_REQUEST: self._handle_collaboration_request,
            MessageType.TASK_ASSIGNMENT: self._handle_task_assignment,
            MessageType.KNOWLEDGE_SHARE: self._handle_knowledge_share,
            MessageType.STATUS_UPDATE: self._handle_status_update,
            MessageType.ERROR_REPORT: self._handle_error_report
        }

    def _start_background_tasks(self):
        """Uruchamia zadania w tle"""
        asyncio.create_task(self._message_delivery_worker())
        asyncio.create_task(self._conversation_timeout_monitor())
        asyncio.create_task(self._cleanup_expired_messages())

    async def register_agent_queue(self, agent_id: str):
        """Rejestruje kolejkę wiadomości dla agenta"""
        if agent_id not in self.agent_queues:
            self.agent_queues[agent_id] = asyncio.Queue(maxsize=self.config['max_queue_size'])
            logger.info(f"📪 Message queue registered for agent {agent_id}")

    async def unregister_agent_queue(self, agent_id: str):
        """Wyrejestrowuje kolejkę agenta"""
        if agent_id in self.agent_queues:
            del self.agent_queues[agent_id]
            logger.info(f"📪 Message queue unregistered for agent {agent_id}")

    async def send_message(self, message: Message) -> bool:
        """Wysyła wiadomość"""
        try:
            # Zapisz wiadomość
            self.messages[message.message_id] = message

            # Określ tryb dostarczania
            if message.delivery_mode == DeliveryMode.DIRECT:
                success = await self._deliver_direct(message)
            elif message.delivery_mode == DeliveryMode.QUEUE:
                success = await self._deliver_to_queue(message)
            elif message.delivery_mode == DeliveryMode.BROADCAST:
                success = await self._deliver_broadcast(message)
            elif message.delivery_mode == DeliveryMode.MULTICAST:
                success = await self._deliver_multicast(message)
            else:
                logger.error(f"Unknown delivery mode: {message.delivery_mode}")
                return False

            if success:
                logger.info(f"📤 Message {message.message_id} sent from {message.sender_id} to {message.recipient_id}")

            return success

        except Exception as e:
            logger.error(f"❌ Failed to send message {message.message_id}: {str(e)}")
            return False

    async def send_request(self, sender_id: str, recipient_id: str, subject: str,
                           payload: Dict[str, Any], timeout: timedelta = timedelta(minutes=5)) -> Optional[
        MessageResponse]:
        """Wysyła request i czeka na response"""

        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.REQUEST,
            sender_id=sender_id,
            recipient_id=recipient_id,
            subject=subject,
            payload=payload,
            priority=MessagePriority.NORMAL,
            delivery_mode=DeliveryMode.DIRECT,
            created_at=datetime.now(),
            requires_response=True,
            response_timeout=timeout
        )

        # Wyślij message
        success = await self.send_message(message)
        if not success:
            return None

        # Czekaj na response
        try:
            response = await self._wait_for_response(message.message_id, timeout)
            return response
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Request {message.message_id} timed out")
            return None

    async def send_response(self, original_message_id: str, sender_id: str,
                            payload: Dict[str, Any], success: bool = True,
                            error_message: Optional[str] = None) -> bool:
        """Wysyła odpowiedź na wiadomość"""

        if original_message_id not in self.messages:
            logger.error(f"Cannot send response - original message {original_message_id} not found")
            return False

        original_message = self.messages[original_message_id]

        response = MessageResponse(
            response_id=str(uuid.uuid4()),
            original_message_id=original_message_id,
            sender_id=sender_id,
            payload=payload,
            created_at=datetime.now(),
            success=success,
            error_message=error_message
        )

        self.responses[response.response_id] = response

        # Powiadom oczekujący handler
        if original_message_id in self.response_handlers:
            handler = self.response_handlers[original_message_id]
            await handler(response)
            del self.response_handlers[original_message_id]

        logger.info(f"📥 Response sent for message {original_message_id}")
        return True

    async def broadcast_message(self, sender_id: str, topic: str, payload: Dict[str, Any],
                                priority: MessagePriority = MessagePriority.NORMAL) -> int:
        """Wysyła broadcast do wszystkich subskrybentów tematu"""

        subscribers = self.broadcast_subscribers.get(topic, [])

        if not subscribers:
            logger.warning(f"No subscribers for broadcast topic: {topic}")
            return 0

        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.BROADCAST,
            sender_id=sender_id,
            recipient_id=None,
            subject=f"Broadcast: {topic}",
            payload=payload,
            priority=priority,
            delivery_mode=DeliveryMode.BROADCAST,
            created_at=datetime.now()
        )

        success_count = 0

        for subscriber_id in subscribers:
            try:
                # Stwórz kopię wiadomości dla każdego subskrybenta
                subscriber_message = Message(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.NOTIFICATION,
                    sender_id=sender_id,
                    recipient_id=subscriber_id,
                    subject=message.subject,
                    payload=message.payload,
                    priority=priority,
                    delivery_mode=DeliveryMode.QUEUE,
                    created_at=datetime.now()
                )

                if await self.send_message(subscriber_message):
                    success_count += 1

            except Exception as e:
                logger.error(f"Failed to deliver broadcast to {subscriber_id}: {str(e)}")

        logger.info(f"📡 Broadcast sent to {success_count}/{len(subscribers)} subscribers")
        return success_count

    async def start_conversation(self, initiator_id: str, participants: List[str],
                                 topic: str) -> str:
        """Rozpoczyna konwersację między agentami"""

        conversation_id = str(uuid.uuid4())

        conversation = Conversation(
            conversation_id=conversation_id,
            participants=[initiator_id] + participants,
            initiated_by=initiator_id,
            topic=topic,
            started_at=datetime.now(),
            last_activity=datetime.now(),
            status='active',
            messages=[]
        )

        self.conversations[conversation_id] = conversation

        # Powiadom uczestników o rozpoczęciu konwersacji
        for participant_id in participants:
            notification = Message(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.NOTIFICATION,
                sender_id=initiator_id,
                recipient_id=participant_id,
                subject=f"Conversation started: {topic}",
                payload={
                    'conversation_id': conversation_id,
                    'topic': topic,
                    'participants': conversation.participants
                },
                priority=MessagePriority.NORMAL,
                delivery_mode=DeliveryMode.DIRECT,
                created_at=datetime.now(),
                conversation_id=conversation_id
            )

            await self.send_message(notification)

        logger.info(f"💬 Conversation {conversation_id} started by {initiator_id}")
        return conversation_id

    async def send_collaboration_request(self, collaboration_request: CollaborationRequest) -> bool:
        """Wysyła żądanie współpracy"""

        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.COLLABORATION_REQUEST,
            sender_id=collaboration_request.from_agent_id,
            recipient_id=collaboration_request.to_agent_id,
            subject=f"Collaboration Request: {collaboration_request.expected_capability}",
            payload={
                'request_id': collaboration_request.request_id,
                'collaboration_type': collaboration_request.collaboration_type.value,
                'task_context': collaboration_request.task_context,
                'expected_capability': collaboration_request.expected_capability,
                'deadline': collaboration_request.deadline.isoformat() if collaboration_request.deadline else None,
                'priority': collaboration_request.priority.value
            },
            priority=MessagePriority.HIGH,
            delivery_mode=DeliveryMode.DIRECT,
            created_at=datetime.now(),
            requires_response=True,
            response_timeout=timedelta(minutes=5)
        )

        return await self.send_message(message)

    async def subscribe_to_broadcast(self, agent_id: str, topic: str):
        """Subskrybuje agenta do broadcasts"""
        if topic not in self.broadcast_subscribers:
            self.broadcast_subscribers[topic] = []

        if agent_id not in self.broadcast_subscribers[topic]:
            self.broadcast_subscribers[topic].append(agent_id)
            logger.info(f"📻 Agent {agent_id} subscribed to topic: {topic}")

    async def unsubscribe_from_broadcast(self, agent_id: str, topic: str):
        """Usuwa subskrypcję"""
        if topic in self.broadcast_subscribers:
            if agent_id in self.broadcast_subscribers[topic]:
                self.broadcast_subscribers[topic].remove(agent_id)
                logger.info(f"📻 Agent {agent_id} unsubscribed from topic: {topic}")

    async def get_agent_messages(self, agent_id: str, limit: int = 10) -> List[Message]:
        """Pobiera wiadomości dla agenta"""
        if agent_id not in self.agent_queues:
            return []

        messages = []
        queue = self.agent_queues[agent_id]

        try:
            for _ in range(min(limit, queue.qsize())):
                message = await asyncio.wait_for(queue.get(), timeout=0.1)
                messages.append(message)
        except asyncio.TimeoutError:
            pass

        return messages

    async def _deliver_direct(self, message: Message) -> bool:
        """Dostarcza wiadomość bezpośrednio"""
        if not message.recipient_id:
            return False

        # Sprawdź czy agent jest aktywny
        active_agents = self.agent_registry.active_agents

        if message.recipient_id in active_agents:
            agent = active_agents[message.recipient_id]

            # Wywołaj odpowiedni handler w agencie
            if message.message_type in agent.message_handlers:
                try:
                    handler = agent.message_handlers[message.message_type]

                    # Jeśli to collaboration request, przekaż CollaborationRequest object
                    if message.message_type == MessageType.COLLABORATION_REQUEST:
                        collab_request = CollaborationRequest(
                            request_id=message.payload['request_id'],
                            from_agent_id=message.sender_id,
                            to_agent_id=message.recipient_id,
                            collaboration_type=CollaborationType(message.payload['collaboration_type']),
                            task_context=message.payload['task_context'],
                            expected_capability=message.payload['expected_capability']
                        )
                        result = await handler(collab_request)
                    else:
                        result = await handler(message.payload)

                    # Jeśli wymaga odpowiedzi, wyślij ją
                    if message.requires_response:
                        await self.send_response(message.message_id, message.recipient_id, result, True)

                    message.delivered_at = datetime.now()
                    return True

                except Exception as e:
                    logger.error(f"Error handling message {message.message_id}: {str(e)}")
                    if message.requires_response:
                        await self.send_response(message.message_id, message.recipient_id,
                                                 {}, False, str(e))
                    return False
            else:
                # Dostarcz do kolejki jeśli brak handlera
                return await self._deliver_to_queue(message)
        else:
            # Agent nieaktywny - spróbuj kolejkę
            return await self._deliver_to_queue(message)

    async def _deliver_to_queue(self, message: Message) -> bool:
        """Dostarcza wiadomość do kolejki agenta"""
        if not message.recipient_id:
            return False

        if message.recipient_id not in self.agent_queues:
            await self.register_agent_queue(message.recipient_id)

        queue = self.agent_queues[message.recipient_id]

        try:
            await queue.put(message)
            message.delivered_at = datetime.now()
            return True
        except asyncio.QueueFull:
            logger.error(f"Queue full for agent {message.recipient_id}")
            return False

    async def _deliver_broadcast(self, message: Message) -> bool:
        """Dostarcza broadcast do wszystkich aktywnych agentów"""
        active_agents = list(self.agent_registry.active_agents.keys())

        if len(active_agents) > self.config['max_broadcast_recipients']:
            logger.warning(f"Too many broadcast recipients: {len(active_agents)}")
            return False

        success_count = 0

        for agent_id in active_agents:
            if agent_id != message.sender_id:  # Nie wysyłaj do siebie
                agent_message = Message(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.NOTIFICATION,
                    sender_id=message.sender_id,
                    recipient_id=agent_id,
                    subject=message.subject,
                    payload=message.payload,
                    priority=message.priority,
                    delivery_mode=DeliveryMode.QUEUE,
                    created_at=datetime.now()
                )

                if await self._deliver_to_queue(agent_message):
                    success_count += 1

        return success_count > 0

    async def _deliver_multicast(self, message: Message) -> bool:
        """Dostarcza do grupy agentów (multicast)"""
        recipients = message.payload.get('recipients', [])

        if not recipients:
            return False

        success_count = 0

        for recipient_id in recipients:
            agent_message = Message(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.NOTIFICATION,
                sender_id=message.sender_id,
                recipient_id=recipient_id,
                subject=message.subject,
                payload=message.payload,
                priority=message.priority,
                delivery_mode=DeliveryMode.DIRECT,
                created_at=datetime.now()
            )

            if await self._deliver_direct(agent_message):
                success_count += 1

        return success_count > 0

    async def _wait_for_response(self, message_id: str, timeout: timedelta) -> MessageResponse:
        """Czeka na odpowiedź na wiadomość"""

        # Utwórz future dla response
        response_future = asyncio.Future()

        async def response_handler(response: MessageResponse):
            if not response_future.done():
                response_future.set_result(response)

        # Zarejestruj handler
        self.response_handlers[message_id] = response_handler

        try:
            # Czekaj na response z timeout
            response = await asyncio.wait_for(response_future, timeout=timeout.total_seconds())
            return response
        except asyncio.TimeoutError:
            # Cleanup handler
            if message_id in self.response_handlers:
                del self.response_handlers[message_id]
            raise

    async def _handle_collaboration_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Domyślny handler dla collaboration requests"""
        return {'status': 'handled_by_messenger', 'timestamp': datetime.now().isoformat()}

    async def _handle_task_assignment(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Domyślny handler dla task assignments"""
        return {'status': 'task_received', 'timestamp': datetime.now().isoformat()}

    async def _handle_knowledge_share(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Domyślny handler dla knowledge sharing"""
        return {'status': 'knowledge_received', 'timestamp': datetime.now().isoformat()}

    async def _handle_status_update(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Domyślny handler dla status updates"""
        return {'status': 'status_acknowledged', 'timestamp': datetime.now().isoformat()}

    async def _handle_error_report(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Domyślny handler dla error reports"""
        logger.error(f"Error reported: {payload}")
        return {'status': 'error_logged', 'timestamp': datetime.now().isoformat()}

    async def _message_delivery_worker(self):
        """Worker do retry failed deliveries"""
        while True:
            try:
                # Sprawdź pending deliveries
                current_time = datetime.now()

                for message_id, message in list(self.pending_deliveries.items()):
                    if message.delivery_attempts >= message.max_delivery_attempts:
                        # Max attempts reached
                        logger.error(
                            f"❌ Message {message_id} delivery failed after {message.max_delivery_attempts} attempts")
                        del self.pending_deliveries[message_id]
                        continue

                    # Retry delivery
                    message.delivery_attempts += 1
                    success = await self._deliver_direct(message)

                    if success:
                        del self.pending_deliveries[message_id]
                        logger.info(f"✅ Message {message_id} delivered on retry {message.delivery_attempts}")

                await asyncio.sleep(self.config['delivery_retry_interval'])

            except Exception as e:
                logger.error(f"Error in delivery worker: {str(e)}")
                await asyncio.sleep(30)

    async def _conversation_timeout_monitor(self):
        """Monitor dla timeout konwersacji"""
        while True:
            try:
                current_time = datetime.now()
                timeout_threshold = current_time - timedelta(seconds=self.config['conversation_timeout'])

                expired_conversations = []
                for conv_id, conversation in self.conversations.items():
                    if (conversation.status == 'active' and
                            conversation.last_activity < timeout_threshold):
                        expired_conversations.append(conv_id)

                for conv_id in expired_conversations:
                    self.conversations[conv_id].status = 'timeout'
                    logger.info(f"💬 Conversation {conv_id} timed out")

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in conversation timeout monitor: {str(e)}")
                await asyncio.sleep(60)

    async def _cleanup_expired_messages(self):
        """Czyści wygasłe wiadomości"""
        while True:
            try:
                current_time = datetime.now()

                # Cleanup expired messages
                expired_messages = []
                for msg_id, message in self.messages.items():
                    if (message.expires_at and current_time > message.expires_at):
                        expired_messages.append(msg_id)

                for msg_id in expired_messages:
                    del self.messages[msg_id]

                # Cleanup old responses (older than 1 hour)
                old_responses = []
                one_hour_ago = current_time - timedelta(hours=1)

                for resp_id, response in self.responses.items():
                    if response.created_at < one_hour_ago:
                        old_responses.append(resp_id)

                for resp_id in old_responses:
                    del self.responses[resp_id]

                logger.info(
                    f"🧹 Cleaned up {len(expired_messages)} expired messages and {len(old_responses)} old responses")

                await asyncio.sleep(3600)  # Cleanup every hour

            except Exception as e:
                logger.error(f"Error in cleanup: {str(e)}")
                await asyncio.sleep(3600)

    def get_messenger_status(self) -> Dict[str, Any]:
        """Zwraca status messengera"""
        total_queue_size = sum(queue.qsize() for queue in self.agent_queues.values())

        return {
            'registered_queues': len(self.agent_queues),
            'total_queued_messages': total_queue_size,
            'stored_messages': len(self.messages),
            'stored_responses': len(self.responses),
            'active_conversations': len([c for c in self.conversations.values() if c.status == 'active']),
            'pending_deliveries': len(self.pending_deliveries),
            'broadcast_topics': len(self.broadcast_subscribers),
            'last_updated': datetime.now().isoformat()
        }


# Test funkcji
async def test_agent_messenger():
    """Test agent messenger"""

    from core.agent_registry import AgentRegistry
    from agents.base_agent import BaseAgent, AgentCapability, AgentTask

    # Test agent class
    class TestAgent(BaseAgent):
        async def initialize(self) -> bool:
            return True

        async def process_task(self, task: AgentTask) -> Dict[str, Any]:
            return {'status': 'completed'}

        def get_specialized_capabilities(self) -> Dict[str, AgentCapability]:
            return {}

    print("📡 Testing Agent Messenger...")
    print("=" * 50)

    # Setup
    registry = AgentRegistry()
    messenger = AgentMessenger(registry)

    # Create test agents
    agent1 = TestAgent('test_agent_1', 'test_agent')
    agent2 = TestAgent('test_agent_2', 'test_agent')

    await registry.register_agent(agent1)
    await registry.register_agent(agent2)

    await messenger.register_agent_queue('test_agent_1')
    await messenger.register_agent_queue('test_agent_2')

    # Test 1: Send simple message
    message = Message(
        message_id=str(uuid.uuid4()),
        message_type=MessageType.NOTIFICATION,
        sender_id='test_agent_1',
        recipient_id='test_agent_2',
        subject='Test message',
        payload={'test_data': 'hello'},
        priority=MessagePriority.NORMAL,
        delivery_mode=DeliveryMode.QUEUE,
        created_at=datetime.now()
    )

    success = await messenger.send_message(message)
    print(f"Message sent: {success}")

    # Test 2: Send request and wait for response
    response = await messenger.send_request(
        sender_id='test_agent_1',
        recipient_id='test_agent_2',
        subject='Test request',
        payload={'question': 'How are you?'},
        timeout=timedelta(seconds=5)
    )

    if response:
        print(f"Response received: {response.success}")
    else:
        print("No response received")

    # Test 3: Broadcast message
    await messenger.subscribe_to_broadcast('test_agent_1', 'general')
    await messenger.subscribe_to_broadcast('test_agent_2', 'general')

    broadcast_count = await messenger.broadcast_message(
        sender_id='system',
        topic='general',
        payload={'announcement': 'System maintenance in 5 minutes'}
    )

    print(f"Broadcast sent to {broadcast_count} agents")

    # Test 4: Messenger status
    status = messenger.get_messenger_status()
    print(f"\n📊 Messenger Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(test_agent_messenger())