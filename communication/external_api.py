#!/usr/bin/.env python3
"""
external_api.py - API dla zewnętrznych integracji
Plik: communication/external_api.py
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from aiohttp import web
import jwt
import uuid

from communication.agent_messenger import AgentMessenger, Message, MessageType
from core.agent_registry import AgentRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIEndpointType(Enum):
    """Typy endpointów API"""
    WEBHOOK = "webhook"
    REST = "rest"
    GRAPHQL = "graphql"
    WEBSOCKET = "websocket"


@dataclass
class APIRequest:
    """Request do API"""
    request_id: str
    endpoint: str
    method: str
    headers: Dict[str, str]
    body: Dict[str, Any]
    query_params: Dict[str, str]
    platform: str  # 'facebook', 'instagram', 'tiktok', 'external'
    timestamp: datetime


@dataclass
class APIResponse:
    """Response z API"""
    request_id: str
    status_code: int
    headers: Dict[str, str]
    body: Dict[str, Any]
    processing_time: float


class ExternalAPI:
    """
    External API dla integracji z różnymi platformami
    """

    def __init__(self, messenger: AgentMessenger, registry: AgentRegistry,
                 config: Dict[str, Any]):
        self.messenger = messenger
        self.registry = registry
        self.config = config

        # Security
        self.api_keys = config.get('api_keys', {})
        self.jwt_secret = config.get('jwt_secret', 'your-secret-key')

        # Rate limiting per platform
        self.rate_limits = {
            'facebook': {'requests_per_minute': 200, 'requests_per_hour': 10000},
            'instagram': {'requests_per_minute': 200, 'requests_per_hour': 5000},
            'tiktok': {'requests_per_minute': 100, 'requests_per_hour': 3000},
            'external': {'requests_per_minute': 60, 'requests_per_hour': 1000}
        }

        # Request tracking
        self.request_history = {}
        self.active_requests = {}

        # Platform-specific handlers
        self.platform_handlers = {
            'facebook': self._handle_facebook_request,
            'instagram': self._handle_instagram_request,
            'tiktok': self._handle_tiktok_request,
            'external': self._handle_external_request
        }

        self.app = None
        self.runner = None

    async def start_api_server(self, host: str = '0.0.0.0', port: int = 8080):
        """Uruchamia serwer API"""
        self.app = web.Application()
        self._setup_routes()
        self._setup_middleware()

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        site = web.TCPSite(self.runner, host, port)
        await site.start()

        logger.info(f"🌐 External API started on {host}:{port}")

    def _setup_routes(self):
        """Konfiguruje routing API"""
        # Health check
        self.app.router.add_get('/health', self._health_check)

        # Platform webhooks
        self.app.router.add_post('/webhook/{platform}', self._webhook_handler)
        self.app.router.add_get('/webhook/{platform}', self._webhook_verification)

        # REST API endpoints
        self.app.router.add_post('/api/v1/content', self._create_content)
        self.app.router.add_get('/api/v1/content/{content_id}', self._get_content)
        self.app.router.add_post('/api/v1/analyze', self._analyze_content)
        self.app.router.add_get('/api/v1/insights', self._get_insights)

        # Agent management
        self.app.router.add_get('/api/v1/agents', self._list_agents)
        self.app.router.add_post('/api/v1/agents/{agent_id}/task', self._assign_task)

        # Platform-specific endpoints
        self.app.router.add_post('/api/v1/{platform}/publish', self._publish_content)
        self.app.router.add_get('/api/v1/{platform}/analytics', self._get_platform_analytics)

    def _setup_middleware(self):
        """Konfiguruje middleware"""

        @web.middleware
        async def auth_middleware(request, handler):
            # Skip auth for health check and webhook verification
            if request.path in ['/health'] or request.method == 'GET':
                return await handler(request)

            # Check API key or JWT
            auth_header = request.headers.get('Authorization', '')

            if auth_header.startswith('Bearer '):
                # JWT auth
                token = auth_header[7:]
                if not self._verify_jwt(token):
                    return web.json_response({'error': 'Invalid token'}, status=401)
            elif auth_header.startswith('ApiKey '):
                # API key auth
                api_key = auth_header[7:]
                if not self._verify_api_key(api_key):
                    return web.json_response({'error': 'Invalid API key'}, status=401)
            else:
                return web.json_response({'error': 'Missing authentication'}, status=401)

            return await handler(request)

        @web.middleware
        async def rate_limit_middleware(request, handler):
            # Extract platform from path or header
            platform = request.match_info.get('platform', 'external')

            if not await self._check_rate_limit(request, platform):
                return web.json_response(
                    {'error': 'Rate limit exceeded'},
                    status=429
                )

            return await handler(request)

        self.app.middlewares.append(auth_middleware)
        self.app.middlewares.append(rate_limit_middleware)

    async def _webhook_handler(self, request: web.Request) -> web.Response:
        """Obsługuje webhook z platform"""
        platform = request.match_info['platform']

        if platform not in self.platform_handlers:
            return web.json_response({'error': 'Unknown platform'}, status=404)

        try:
            data = await request.json()

            # Zapisz request
            api_request = APIRequest(
                request_id=str(uuid.uuid4()),
                endpoint=f'/webhook/{platform}',
                method='POST',
                headers=dict(request.headers),
                body=data,
                query_params=dict(request.query),
                platform=platform,
                timestamp=datetime.now()
            )

            # Przetwórz przez odpowiedni handler
            handler = self.platform_handlers[platform]
            response = await handler(api_request)

            return web.json_response(response.body, status=response.status_code)

        except Exception as e:
            logger.error(f"Webhook error for {platform}: {str(e)}")
            return web.json_response({'error': 'Internal error'}, status=500)

    async def _webhook_verification(self, request: web.Request) -> web.Response:
        """Weryfikacja webhook (głównie dla Facebooka)"""
        platform = request.match_info['platform']

        if platform == 'facebook':
            # Facebook webhook verification
            mode = request.query.get('hub.mode')
            token = request.query.get('hub.verify_token')
            challenge = request.query.get('hub.challenge')

            if mode == 'subscribe' and token == self.config.get('fb_verify_token'):
                return web.Response(text=challenge)

        return web.Response(text='OK')

    async def _create_content(self, request: web.Request) -> web.Response:
        """Endpoint do tworzenia contentu"""
        try:
            data = await request.json()

            # Wyślij do Content Creator Agent
            message = Message(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.TASK_ASSIGNMENT,
                sender_id='external_api',
                recipient_id='content_creator',
                subject='Create content',
                payload={
                    'task_type': 'create_content',
                    'content_type': data.get('type', 'post'),
                    'platform': data.get('platform', 'facebook'),
                    'parameters': data
                },
                priority=data.get('priority', 'normal'),
                delivery_mode='direct',
                created_at=datetime.now()
            )

            success = await self.messenger.send_message(message)

            if success:
                return web.json_response({
                    'status': 'accepted',
                    'message_id': message.message_id,
                    'message': 'Content creation task submitted'
                })
            else:
                return web.json_response(
                    {'error': 'Failed to submit task'},
                    status=500
                )

        except Exception as e:
            logger.error(f"Create content error: {str(e)}")
            return web.json_response({'error': str(e)}, status=400)

    async def _analyze_content(self, request: web.Request) -> web.Response:
        """Endpoint do analizy contentu"""
        try:
            data = await request.json()
            content = data.get('content', '')

            # Wyślij do Zinzino Specialist dla analizy
            message = Message(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.REQUEST,
                sender_id='external_api',
                recipient_id='zinzino_specialist',
                subject='Analyze content',
                payload={
                    'task_type': 'analyze_content',
                    'content': content,
                    'check_scientific_validity': data.get('check_validity', True),
                    'check_zinzino_relevance': data.get('check_relevance', True)
                },
                priority='high',
                delivery_mode='direct',
                created_at=datetime.now(),
                requires_response=True
            )

            # Czekaj na odpowiedź
            response = await self.messenger.send_request(
                'external_api',
                'zinzino_specialist',
                'Analyze content',
                message.payload
            )

            if response:
                return web.json_response(response.payload)
            else:
                return web.json_response(
                    {'error': 'Analysis timeout'},
                    status=504
                )

        except Exception as e:
            logger.error(f"Analyze content error: {str(e)}")
            return web.json_response({'error': str(e)}, status=400)

    async def _get_insights(self, request: web.Request) -> web.Response:
        """Endpoint do pobierania insights"""
        try:
            platform = request.query.get('platform', 'all')
            days = int(request.query.get('days', 7))

            # Pobierz dane z różnych agentów
            insights = {
                'period': f'{days} days',
                'platform': platform,
                'content_performance': {},
                'engagement_metrics': {},
                'conversion_data': {},
                'recommendations': []
            }

            # TODO: Aggregate data from agents

            return web.json_response(insights)

        except Exception as e:
            logger.error(f"Get insights error: {str(e)}")
            return web.json_response({'error': str(e)}, status=400)

    async def _handle_facebook_request(self, request: APIRequest) -> APIResponse:
        """Handler dla Facebook webhooks"""
        data = request.body

        # Obsługa różnych typów eventów
        if 'entry' in data:
            for entry in data['entry']:
                if 'messaging' in entry:
                    # Wiadomości
                    for event in entry['messaging']:
                        await self._process_facebook_message(event)
                elif 'changes' in entry:
                    # Zmiany (komentarze, likes, etc.)
                    for change in entry['changes']:
                        await self._process_facebook_change(change)

        return APIResponse(
            request_id=request.request_id,
            status_code=200,
            headers={},
            body={'status': 'received'},
            processing_time=0.1
        )

    async def _handle_instagram_request(self, request: APIRequest) -> APIResponse:
        """Handler dla Instagram webhooks"""
        # Placeholder dla przyszłej implementacji
        return APIResponse(
            request_id=request.request_id,
            status_code=200,
            headers={},
            body={'status': 'instagram_handler_not_implemented'},
            processing_time=0.1
        )

    async def _handle_tiktok_request(self, request: APIRequest) -> APIResponse:
        """Handler dla TikTok webhooks"""
        # Placeholder dla przyszłej implementacji
        return APIResponse(
            request_id=request.request_id,
            status_code=200,
            headers={},
            body={'status': 'tiktok_handler_not_implemented'},
            processing_time=0.1
        )

    async def _handle_external_request(self, request: APIRequest) -> APIResponse:
        """Handler dla zewnętrznych requestów"""
        # Generic handler dla innych integracji
        return APIResponse(
            request_id=request.request_id,
            status_code=200,
            headers={},
            body={'status': 'processed'},
            processing_time=0.1
        )

    def _verify_jwt(self, token: str) -> bool:
        """Weryfikuje JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return True
        except jwt.InvalidTokenError:
            return False

    def _verify_api_key(self, api_key: str) -> bool:
        """Weryfikuje API key"""
        return api_key in self.api_keys.values()

    async def _check_rate_limit(self, request: web.Request, platform: str) -> bool:
        """Sprawdza rate limiting"""
        # TODO: Implementacja rate limiting
        return True

    async def _health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        agents_status = self.registry.get_registry_status()

        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'agents': agents_status['registry_stats'],
            'platforms': list(self.platform_handlers.keys())
        })

    async def _list_agents(self, request: web.Request) -> web.Response:
        """Lista aktywnych agentów"""
        agents = []
        for agent_id, agent in self.registry.active_agents.items():
            agents.append({
                'id': agent_id,
                'type': agent.agent_type,
                'status': agent.state.value,
                'capabilities': list(agent.capabilities.keys())
            })

        return web.json_response({'agents': agents})

    async def _assign_task(self, request: web.Request) -> web.Response:
        """Przypisuje zadanie do agenta"""
        agent_id = request.match_info['agent_id']

        try:
            data = await request.json()

            # Sprawdź czy agent istnieje
            if agent_id not in self.registry.active_agents:
                return web.json_response(
                    {'error': 'Agent not found'},
                    status=404
                )

            # Wyślij zadanie
            message = Message(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.TASK_ASSIGNMENT,
                sender_id='external_api',
                recipient_id=agent_id,
                subject=data.get('subject', 'External task'),
                payload=data,
                priority=data.get('priority', 'normal'),
                delivery_mode='direct',
                created_at=datetime.now()
            )

            success = await self.messenger.send_message(message)

            if success:
                return web.json_response({
                    'status': 'accepted',
                    'message_id': message.message_id
                })
            else:
                return web.json_response(
                    {'error': 'Failed to assign task'},
                    status=500
                )

        except Exception as e:
            logger.error(f"Assign task error: {str(e)}")
            return web.json_response({'error': str(e)}, status=400)

        async def _publish_content(self, request: web.Request) -> web.Response:
            """Publikuje content na platformie"""
            platform = request.match_info['platform']

            try:
                data = await request.json()

                # Wyślij do odpowiedniego agenta
                message = Message(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.TASK_ASSIGNMENT,
                    sender_id='external_api',
                    recipient_id='content_creator',
                    subject=f'Publish to {platform}',
                    payload={
                        'task_type': 'publish_content',
                        'platform': platform,
                        'content': data.get('content'),
                        'schedule_time': data.get('schedule_time'),
                        'targeting': data.get('targeting', {})
                    },
                    priority='high',
                    delivery_mode='direct',
                    created_at=datetime.now()
                )

                success = await self.messenger.send_message(message)

                if success:
                    return web.json_response({
                        'status': 'scheduled',
                        'message_id': message.message_id,
                        'platform': platform
                    })
                else:
                    return web.json_response(
                        {'error': 'Failed to schedule publishing'},
                        status=500
                    )

            except Exception as e:
                logger.error(f"Publish content error: {str(e)}")
                return web.json_response({'error': str(e)}, status=400)

        async def _get_platform_analytics(self, request: web.Request) -> web.Response:
            """Pobiera analytics dla platformy"""
            platform = request.match_info['platform']

            try:
                # Request analytics from analytics agent
                response = await self.messenger.send_request(
                    'external_api',
                    'analytics_agent',
                    f'Get {platform} analytics',
                    {
                        'platform': platform,
                        'metrics': request.query.getall('metrics', []),
                        'date_from': request.query.get('date_from'),
                        'date_to': request.query.get('date_to')
                    }
                )

                if response:
                    return web.json_response(response.payload)
                else:
                    return web.json_response(
                        {'error': 'Analytics timeout'},
                        status=504
                    )

            except Exception as e:
                logger.error(f"Get analytics error: {str(e)}")
                return web.json_response({'error': str(e)}, status=400)

        async def _process_facebook_message(self, event: Dict):
            """Przetwarza wiadomość z Facebooka"""
            sender_id = event['sender']['id']

            if 'message' in event:
                message_text = event['message'].get('text', '')

                # Wyślij do EngagementBot
                await self.messenger.send_message(Message(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.TASK_ASSIGNMENT,
                    sender_id='external_api',
                    recipient_id='engagement_bot',
                    subject='Handle Facebook message',
                    payload={
                        'task_type': 'handle_message',
                        'platform': 'facebook',
                        'sender_id': sender_id,
                        'message': message_text,
                        'timestamp': event.get('timestamp')
                    },
                    priority='high',
                    delivery_mode='direct',
                    created_at=datetime.now()
                ))

        async def _process_facebook_change(self, change: Dict):
            """Przetwarza zmianę z Facebooka (komentarz, like, etc.)"""
            if change['field'] == 'feed' and change['value'].get('item') == 'comment':
                # Nowy komentarz
                comment_id = change['value']['comment_id']

                await self.messenger.send_message(Message(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.TASK_ASSIGNMENT,
                    sender_id='external_api',
                    recipient_id='engagement_bot',
                    subject='Handle Facebook comment',
                    payload={
                        'task_type': 'respond_to_comment',
                        'platform': 'facebook',
                        'comment_id': comment_id,
                        'post_id': change['value'].get('post_id'),
                        'message': change['value'].get('message')
                    },
                    priority='high',
                    delivery_mode='direct',
                    created_at=datetime.now()
                ))

        async def stop_api_server(self):
            """Zatrzymuje serwer API"""
            if self.runner:
                await self.runner.cleanup()
                logger.info("🛑 External API stopped")