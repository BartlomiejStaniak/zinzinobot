#!/usr/bin/.env python3
"""
protocol_handler.py - Obsługa protokołów komunikacji między agentami
Plik: communication/protocol_handler.py
"""

import asyncio
import json
import ssl
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging
import websockets
import aiohttp
from urllib.parse import urlparse

from communication.agent_messenger import AgentMessenger, Message, MessageType, MessagePriority

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProtocolType(Enum):
    """Typy protokołów komunikacji"""
    LOCAL = "local"  # Lokalna komunikacja (in-process)
    HTTP_REST = "http_rest"  # REST API over HTTP
    WEBSOCKET = "websocket"  # WebSocket connections
    GRPC = "grpc"  # gRPC for high-performance
    MQTT = "mqtt"  # MQTT for IoT/distributed systems
    REDIS_PUBSUB = "redis"  # Redis pub/sub


class SecurityLevel(Enum):
    """Poziomy bezpieczeństwa"""
    NONE = "none"
    BASIC_AUTH = "basic_auth"
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    MUTUAL_TLS = "mutual_tls"


@dataclass
class EndpointConfig:
    """Konfiguracja endpointu komunikacji"""
    protocol: ProtocolType
    address: str
    port: Optional[int] = None
    path: Optional[str] = None
    security_level: SecurityLevel = SecurityLevel.NONE
    credentials: Optional[Dict[str, str]] = None
    ssl_context: Optional[ssl.SSLContext] = None
    timeout: int = 30
    retry_attempts: int = 3
    headers: Optional[Dict[str, str]] = None


@dataclass
class ConnectionStatus:
    """Status połączenia"""
    agent_id: str
    endpoint: EndpointConfig
    is_connected: bool
    last_ping: Optional[datetime] = None
    connection_time: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None


class BaseProtocolHandler(ABC):
    """Bazowa klasa dla handlerów protokołów"""

    def __init__(self, protocol_type: ProtocolType, messenger: AgentMessenger):
        self.protocol_type = protocol_type
        self.messenger = messenger
        self.connections: Dict[str, ConnectionStatus] = {}
        self.is_running = False

    @abstractmethod
    async def connect(self, agent_id: str, endpoint: EndpointConfig) -> bool:
        """Łączy się z agentem"""
        pass

    @abstractmethod
    async def disconnect(self, agent_id: str) -> bool:
        """Rozłącza się z agentem"""
        pass

    @abstractmethod
    async def send_message(self, agent_id: str, message: Message) -> bool:
        """Wysyła wiadomość do agenta"""
        pass

    @abstractmethod
    async def start_server(self, config: EndpointConfig) -> bool:
        """Uruchamia serwer dla incoming connections"""
        pass

    @abstractmethod
    async def stop_server(self) -> bool:
        """Zatrzymuje serwer"""
        pass


class LocalProtocolHandler(BaseProtocolHandler):
    """Handler dla lokalnej komunikacji (in-process)"""

    def __init__(self, messenger: AgentMessenger):
        super().__init__(ProtocolType.LOCAL, messenger)

    async def connect(self, agent_id: str, endpoint: EndpointConfig) -> bool:
        """Lokalne połączenie (zawsze dostępne)"""
        self.connections[agent_id] = ConnectionStatus(
            agent_id=agent_id,
            endpoint=endpoint,
            is_connected=True,
            connection_time=datetime.now()
        )
        return True

    async def disconnect(self, agent_id: str) -> bool:
        """Lokalne rozłączenie"""
        if agent_id in self.connections:
            self.connections[agent_id].is_connected = False
            return True
        return False

    async def send_message(self, agent_id: str, message: Message) -> bool:
        """Lokalne wysłanie wiadomości"""
        if agent_id in self.connections and self.connections[agent_id].is_connected:
            # Bezpośrednio przez messenger
            return await self.messenger._deliver_direct(message)
        return False

    async def start_server(self, config: EndpointConfig) -> bool:
        """Lokalny serwer nie wymaga uruchomienia"""
        self.is_running = True
        return True

    async def stop_server(self) -> bool:
        """Lokalny serwer nie wymaga zatrzymania"""
        self.is_running = False
        return True


class HTTPRestProtocolHandler(BaseProtocolHandler):
    """Handler dla REST API over HTTP"""

    def __init__(self, messenger: AgentMessenger):
        super().__init__(ProtocolType.HTTP_REST, messenger)
        self.session: Optional[aiohttp.ClientSession] = None
        self.server: Optional[aiohttp.web.Application] = None
        self.server_runner: Optional[aiohttp.web.AppRunner] = None

    async def connect(self, agent_id: str, endpoint: EndpointConfig) -> bool:
        """Łączy się z REST API agenta"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=endpoint.timeout))

            # Test connection
            url = self._build_url(endpoint, "/health")
            headers = self._build_headers(endpoint)

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    self.connections[agent_id] = ConnectionStatus(
                        agent_id=agent_id,
                        endpoint=endpoint,
                        is_connected=True,
                        connection_time=datetime.now()
                    )
                    logger.info(f"🔗 HTTP connection established with {agent_id}")
                    return True
                else:
                    logger.error(f"❌ HTTP connection failed with {agent_id}: {response.status}")
                    return False

        except Exception as e:
            logger.error(f"❌ HTTP connection error with {agent_id}: {str(e)}")
            return False

    async def disconnect(self, agent_id: str) -> bool:
        """Rozłącza HTTP połączenie"""
        if agent_id in self.connections:
            self.connections[agent_id].is_connected = False
            logger.info(f"🔌 HTTP connection closed with {agent_id}")
            return True
        return False

    async def send_message(self, agent_id: str, message: Message) -> bool:
        """Wysyła wiadomość przez REST API"""
        if agent_id not in self.connections or not self.connections[agent_id].is_connected:
            return False

        try:
            endpoint = self.connections[agent_id].endpoint
            url = self._build_url(endpoint, "/messages")
            headers = self._build_headers(endpoint)
            headers['Content-Type'] = 'application/json'

            payload = {
                'message_id': message.message_id,
                'message_type': message.message_type.value,
                'sender_id': message.sender_id,
                'subject': message.subject,
                'payload': message.payload,
                'priority': message.priority.value,
                'created_at': message.created_at.isoformat(),
                'requires_response': message.requires_response
            }

            async with self.session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    logger.info(f"📤 HTTP message sent to {agent_id}")
                    return True
                else:
                    logger.error(f"❌ HTTP message failed to {agent_id}: {response.status}")
                    return False

        except Exception as e:
            logger.error(f"❌ HTTP send error to {agent_id}: {str(e)}")
            return False

    async def start_server(self, config: EndpointConfig) -> bool:
        """Uruchamia HTTP server"""
        try:
            app = aiohttp.web.Application()

            # Routes
            app.router.add_get('/health', self._handle_health)
            app.router.add_post('/messages', self._handle_incoming_message)
            app.router.add_get('/status', self._handle_status)

            self.server = app
            self.server_runner = aiohttp.web.AppRunner(app)
            await self.server_runner.setup()

            site = aiohttp.web.TCPSite(
                self.server_runner,
                config.address,
                config.port or 8080
            )
            await site.start()

            self.is_running = True
            logger.info(f"🌐 HTTP server started on {config.address}:{config.port or 8080}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start HTTP server: {str(e)}")
            return False

    async def stop_server(self) -> bool:
        """Zatrzymuje HTTP server"""
        try:
            if self.server_runner:
                await self.server_runner.cleanup()
                self.server_runner = None
                self.server = None

            if self.session:
                await self.session.close()
                self.session = None

            self.is_running = False
            logger.info("🛑 HTTP server stopped")
            return True

        except Exception as e:
            logger.error(f"❌ Error stopping HTTP server: {str(e)}")
            return False

    def _build_url(self, endpoint: EndpointConfig, path: str) -> str:
        """Buduje URL dla endpointu"""
        scheme = "https" if endpoint.ssl_context else "http"
        port_part = f":{endpoint.port}" if endpoint.port else ""
        base_path = endpoint.path or ""
        return f"{scheme}://{endpoint.address}{port_part}{base_path}{path}"

    def _build_headers(self, endpoint: EndpointConfig) -> Dict[str, str]:
        """Buduje nagłówki HTTP"""
        headers = endpoint.headers.copy() if endpoint.headers else {}

        if endpoint.security_level == SecurityLevel.API_KEY and endpoint.credentials:
            api_key = endpoint.credentials.get('api_key')
            if api_key:
                headers['X-API-Key'] = api_key

        elif endpoint.security_level == SecurityLevel.JWT_TOKEN and endpoint.credentials:
            token = endpoint.credentials.get('token')
            if token:
                headers['Authorization'] = f'Bearer {token}'

        return headers

    async def _handle_health(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """Health check endpoint"""
        return aiohttp.web.json_response({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

    async def _handle_incoming_message(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """Obsługuje przychodzące wiadomości"""
        try:
            data = await request.json()

            # Stwórz Message object
            message = Message(
                message_id=data['message_id'],
                message_type=MessageType(data['message_type']),
                sender_id=data['sender_id'],
                recipient_id=None,  # Will be determined by local routing
                subject=data['subject'],
                payload=data['payload'],
                priority=MessagePriority(data['priority']),
                delivery_mode=data.get('delivery_mode', 'direct'),
                created_at=datetime.fromisoformat(data['created_at']),
                requires_response=data.get('requires_response', False)
            )

            # Forward to messenger
            success = await self.messenger.send_message(message)

            if success:
                return aiohttp.web.json_response({'status': 'received', 'message_id': message.message_id})
            else:
                return aiohttp.web.json_response({'status': 'failed'}, status=500)

        except Exception as e:
            logger.error(f"❌ Error handling incoming HTTP message: {str(e)}")
            return aiohttp.web.json_response({'status': 'error', 'message': str(e)}, status=400)

    async def _handle_status(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """Status endpoint"""
        status = {
            'protocol': self.protocol_type.value,
            'is_running': self.is_running,
            'connections': len(self.connections),
            'timestamp': datetime.now().isoformat()
        }
        return aiohttp.web.json_response(status)


class WebSocketProtocolHandler(BaseProtocolHandler):
    """Handler dla WebSocket connections"""

    def __init__(self, messenger: AgentMessenger):
        super().__init__(ProtocolType.WEBSOCKET, messenger)
        self.websockets: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.server = None

    async def connect(self, agent_id: str, endpoint: EndpointConfig) -> bool:
        """Łączy się przez WebSocket"""
        try:
            uri = self._build_websocket_uri(endpoint)

            websocket = await websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )

            self.websockets[agent_id] = websocket
            self.connections[agent_id] = ConnectionStatus(
                agent_id=agent_id,
                endpoint=endpoint,
                is_connected=True,
                connection_time=datetime.now()
            )

            # Start listening for messages
            asyncio.create_task(self._listen_websocket(agent_id, websocket))

            logger.info(f"🔗 WebSocket connection established with {agent_id}")
            return True

        except Exception as e:
            logger.error(f"❌ WebSocket connection failed with {agent_id}: {str(e)}")
            return False

    async def disconnect(self, agent_id: str) -> bool:
        """Rozłącza WebSocket"""
        try:
            if agent_id in self.websockets:
                websocket = self.websockets[agent_id]
                await websocket.close()
                del self.websockets[agent_id]

            if agent_id in self.connections:
                self.connections[agent_id].is_connected = False

            logger.info(f"🔌 WebSocket connection closed with {agent_id}")
            return True

        except Exception as e:
            logger.error(f"❌ WebSocket disconnect error with {agent_id}: {str(e)}")
            return False

    async def send_message(self, agent_id: str, message: Message) -> bool:
        """Wysyła wiadomość przez WebSocket"""
        if agent_id not in self.websockets:
            return False

        try:
            websocket = self.websockets[agent_id]

            payload = {
                'message_id': message.message_id,
                'message_type': message.message_type.value,
                'sender_id': message.sender_id,
                'subject': message.subject,
                'payload': message.payload,
                'priority': message.priority.value,
                'created_at': message.created_at.isoformat(),
                'requires_response': message.requires_response
            }

            await websocket.send(json.dumps(payload))
            logger.info(f"📤 WebSocket message sent to {agent_id}")
            return True

        except Exception as e:
            logger.error(f"❌ WebSocket send error to {agent_id}: {str(e)}")
            # Remove broken connection
            await self.disconnect(agent_id)
            return False

    async def start_server(self, config: EndpointConfig) -> bool:
        """Uruchamia WebSocket server"""
        try:
            self.server = await websockets.serve(
                self._handle_websocket_connection,
                config.address,
                config.port or 8765,
                ping_interval=20,
                ping_timeout=10
            )

            self.is_running = True
            logger.info(f"🌐 WebSocket server started on {config.address}:{config.port or 8765}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start WebSocket server: {str(e)}")
            return False

    async def stop_server(self) -> bool:
        """Zatrzymuje WebSocket server"""
        try:
            if self.server:
                self.server.close()
                await self.server.wait_closed()
                self.server = None

            # Close all connections
            for agent_id in list(self.websockets.keys()):
                await self.disconnect(agent_id)

            self.is_running = False
            logger.info("🛑 WebSocket server stopped")
            return True

        except Exception as e:
            logger.error(f"❌ Error stopping WebSocket server: {str(e)}")
            return False

    def _build_websocket_uri(self, endpoint: EndpointConfig) -> str:
        """Buduje URI dla WebSocket"""
        scheme = "wss" if endpoint.ssl_context else "ws"
        port_part = f":{endpoint.port}" if endpoint.port else ""
        path = endpoint.path or ""
        return f"{scheme}://{endpoint.address}{port_part}{path}"

    async def _handle_websocket_connection(self, websocket, path):
        """Obsługuje nowe połączenie WebSocket"""
        agent_id = f"ws_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            self.websockets[agent_id] = websocket
            self.connections[agent_id] = ConnectionStatus(
                agent_id=agent_id,
                endpoint=EndpointConfig(ProtocolType.WEBSOCKET, "incoming"),
                is_connected=True,
                connection_time=datetime.now()
            )

            logger.info(f"🔗 New WebSocket connection: {agent_id}")

            await self._listen_websocket(agent_id, websocket)

        except Exception as e:
            logger.error(f"❌ WebSocket connection error: {str(e)}")
        finally:
            await self.disconnect(agent_id)

    async def _listen_websocket(self, agent_id: str, websocket):
        """Nasłuchuje wiadomości z WebSocket"""
        try:
            async for message_data in websocket:
                try:
                    data = json.loads(message_data)

                    # Stwórz Message object
                    message = Message(
                        message_id=data['message_id'],
                        message_type=MessageType(data['message_type']),
                        sender_id=data['sender_id'],
                        recipient_id=None,
                        subject=data['subject'],
                        payload=data['payload'],
                        priority=MessagePriority(data['priority']),
                        delivery_mode='direct',
                        created_at=datetime.fromisoformat(data['created_at']),
                        requires_response=data.get('requires_response', False)
                    )

                    # Forward to messenger
                    await self.messenger.send_message(message)

                except json.JSONDecodeError:
                    logger.error(f"❌ Invalid JSON from WebSocket {agent_id}")
                except Exception as e:
                    logger.error(f"❌ Error processing WebSocket message from {agent_id}: {str(e)}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 WebSocket connection closed: {agent_id}")
        except Exception as e:
            logger.error(f"❌ WebSocket listen error for {agent_id}: {str(e)}")


class ProtocolHandler:
    """
    Główny handler protokołów - zarządza wszystkimi protokołami komunikacji
    """

    def __init__(self, messenger: AgentMessenger):
        self.messenger = messenger

        # Protocol handlers
        self.handlers: Dict[ProtocolType, BaseProtocolHandler] = {
            ProtocolType.LOCAL: LocalProtocolHandler(messenger),
            ProtocolType.HTTP_REST: HTTPRestProtocolHandler(messenger),
            ProtocolType.WEBSOCKET: WebSocketProtocolHandler(messenger)
        }

        # Agent -> Protocol mapping
        self.agent_protocols: Dict[str, ProtocolType] = {}

        # Configuration
        self.server_configs: Dict[ProtocolType, EndpointConfig] = {}
        self.is_running = False

    async def start_servers(self, configs: Dict[ProtocolType, EndpointConfig]):
        """Uruchamia serwery dla protokołów"""
        self.server_configs = configs

        for protocol_type, config in configs.items():
            if protocol_type in self.handlers:
                success = await self.handlers[protocol_type].start_server(config)
                if success:
                    logger.info(f"✅ {protocol_type.value} server started")
                else:
                    logger.error(f"❌ Failed to start {protocol_type.value} server")

        self.is_running = True

    async def stop_servers(self):
        """Zatrzymuje wszystkie serwery"""
        for protocol_type, handler in self.handlers.items():
            await handler.stop_server()

        self.is_running = False
        logger.info("🛑 All protocol servers stopped")

    async def connect_agent(self, agent_id: str, endpoint: EndpointConfig) -> bool:
        """Łączy się z agentem przez odpowiedni protokół"""
        protocol_type = endpoint.protocol

        if protocol_type not in self.handlers:
            logger.error(f"❌ Unsupported protocol: {protocol_type}")
            return False

        handler = self.handlers[protocol_type]
        success = await handler.connect(agent_id, endpoint)

        if success:
            self.agent_protocols[agent_id] = protocol_type
            logger.info(f"✅ Agent {agent_id} connected via {protocol_type.value}")

        return success

    async def disconnect_agent(self, agent_id: str) -> bool:
        """Rozłącza agenta"""
        if agent_id not in self.agent_protocols:
            return False

        protocol_type = self.agent_protocols[agent_id]
        handler = self.handlers[protocol_type]
        success = await handler.disconnect(agent_id)

        if success:
            del self.agent_protocols[agent_id]
            logger.info(f"✅ Agent {agent_id} disconnected")

        return success

    async def send_message_to_agent(self, agent_id: str, message: Message) -> bool:
        """Wysyła wiadomość do agenta przez odpowiedni protokół"""
        if agent_id not in self.agent_protocols:
            # Try local protocol as fallback
            return await self.handlers[ProtocolType.LOCAL].send_message(agent_id, message)

        protocol_type = self.agent_protocols[agent_id]
        handler = self.handlers[protocol_type]
        return await handler.send_message(agent_id, message)

    def get_protocol_status(self) -> Dict[str, Any]:
        """Zwraca status wszystkich protokołów"""
        status = {
            'is_running': self.is_running,
            'protocols': {},
            'connected_agents': len(self.agent_protocols),
            'agent_protocols': dict(self.agent_protocols)
        }

        for protocol_type, handler in self.handlers.items():
            status['protocols'][protocol_type.value] = {
                'is_running': handler.is_running,
                'connections': len(handler.connections),
                'connected_agents': [
                    agent_id for agent_id, conn in handler.connections.items()
                    if conn.is_connected
                ]
            }

        return status


# Test funkcji
async def test_protocol_handler():
    """Test protocol handler"""

    from communication.agent_messenger import AgentMessenger
    from core.agent_registry import AgentRegistry

    print("🔌 Testing Protocol Handler...")
    print("=" * 50)

    # Setup
    registry = AgentRegistry()
    messenger = AgentMessenger(registry)
    protocol_handler = ProtocolHandler(messenger)

    # Test 1: Start servers
    server_configs = {
        ProtocolType.LOCAL: EndpointConfig(ProtocolType.LOCAL, "localhost"),
        ProtocolType.HTTP_REST: EndpointConfig(ProtocolType.HTTP_REST, "localhost", 8080),
        ProtocolType.WEBSOCKET: EndpointConfig(ProtocolType.WEBSOCKET, "localhost", 8765)
    }

    await protocol_handler.start_servers(server_configs)

    # Test 2: Connect local agent
    local_endpoint = EndpointConfig(ProtocolType.LOCAL, "localhost")
    success = await protocol_handler.connect_agent("local_agent_1", local_endpoint)
    print(f"Local agent connection: {success}")

    # Test 3: Protocol status
    status = protocol_handler.get_protocol_status()
    print(f"\n📊 Protocol Status:")
    print(f"  Running: {status['is_running']}")
    print(f"  Connected agents: {status['connected_agents']}")

    for protocol, info in status['protocols'].items():
        print(f"  {protocol}: running={info['is_running']}, connections={info['connections']}")

    # Cleanup
    await protocol_handler.stop_servers()


if __name__ == "__main__":
    asyncio.run(test_protocol_handler())