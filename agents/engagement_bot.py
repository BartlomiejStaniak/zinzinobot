"""
Engagement Bot Agent - Agent obsługujący interakcje na Facebook
Specjalizuje się w odpowiadaniu na komentarze i wiadomości
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import re
from dataclasses import dataclass
from enum import Enum

from core.agent_base import BaseAgent, AgentResult
from core.task_queue import Task
from core.memory_manager import MemoryManager
from core.facebook_client import FacebookClient


class InteractionType(Enum):
    """Typy interakcji"""
    COMMENT = "comment"
    REPLY = "reply"
    MESSAGE = "message"
    REACTION = "reaction"
    MENTION = "mention"


class SentimentType(Enum):
    """Typy sentymentu"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    QUESTION = "question"
    COMPLAINT = "complaint"


@dataclass
class InteractionContext:
    """Kontekst interakcji"""
    interaction_id: str
    interaction_type: InteractionType
    author_name: str
    author_id: str
    content: str
    post_id: Optional[str]
    timestamp: datetime
    sentiment: SentimentType
    requires_response: bool
    priority: int


@dataclass
class ResponseTemplate:
    """Szablon odpowiedzi"""
    trigger_keywords: List[str]
    response_templates: List[str]
    sentiment_type: SentimentType
    follow_up_required: bool
    escalate_to_human: bool


class EngagementBotAgent(BaseAgent):
    """
    Agent obsługujący interakcje na Facebook

    Funkcjonalności:
    - Automatyczne odpowiadanie na komentarze
    - Obsługa wiadomości prywatnych
    - Analiza sentymentu
    - Eskalacja złożonych przypadków
    - Personalizacja odpowiedzi
    - Monitoring mentions i reakcji
    """

    def __init__(self, name: str, config: Dict[str, Any],
                 facebook_client: FacebookClient, memory_manager: MemoryManager):
        super().__init__(name, config)
        self.facebook_client = facebook_client
        self.memory_manager = memory_manager

        # Konfiguracja bota
        self.response_delay = config.get('response_delay', 300)  # 5 minut
        self.max_interactions_per_hour = config.get('max_interactions_per_hour', 50)
        self.escalation_keywords = config.get('escalation_keywords', [])

        # Szablony odpowiedzi
        self.response_templates = self._load_response_templates()

        # Statystyki
        self.interactions_handled = 0
        self.escalations_created = 0
        self.response_times = []

        # Monitoring
        self.monitored_keywords = config.get('monitored_keywords', [])
        self.last_check_time = datetime.now()

        self.logger.info(f"Engagement Bot Agent {name} zainicjalizowany")

    def _load_response_templates(self) -> List[ResponseTemplate]:
        """Ładowanie szablonów odpowiedzi"""
        return [
            # Pozytywne komentarze
            ResponseTemplate(
                trigger_keywords=["świetnie", "super", "dziękuję", "brawo", "fantastycznie", "polecam"],
                response_templates=[
                    "🙏 Dziękujemy za miłe słowa! To dla nas ogromna motywacja!",
                    "✨ Cieszę się, że jesteś zadowolony/a! Dzięki za wsparcie!",
                    "❤️ Bardzo dziękujemy! Takie komentarze napędzają nas do działania!",
                    "🌟 To wspaniałe! Dzięki, że dzielisz się swoją opinią!"
                ],
                sentiment_type=SentimentType.POSITIVE,
                follow_up_required=False,
                escalate_to_human=False
            ),

            # Pytania o produkty
            ResponseTemplate(
                trigger_keywords=["jak", "kiedy", "ile", "gdzie", "czy", "co", "dlaczego", "?"],
                response_templates=[
                    "🤔 Świetne pytanie! Chętnie pomogę - {}",
                    "💡 Dzięki za pytanie! Oto odpowiedź: {}",
                    "📚 Cieszę się, że pytasz! {} Masz więcej pytań?",
                    "🎯 Odpowiadam na Twoje pytanie: {}"
                ],
                sentiment_type=SentimentType.QUESTION,
                follow_up_required=True,
                escalate_to_human=False
            ),

            # Skargi i problemy
            ResponseTemplate(
                trigger_keywords=["problem", "nie działa", "zły", "kiepski", "rozczarowany", "zwrot"],
                response_templates=[
                    "😔 Bardzo mi przykro, że masz problem. Napiszę do Ciebie prywatnie, żeby to rozwiązać!",
                    "🆘 Przepraszam za problemy! Skontaktuję się z Tobą bezpośrednio.",
                    "💪 Zależy nam na Twojej satysfakcji! Zaraz się tym zajmę osobiście.",
                    "🔧 Przykro mi z powodu problemów! Rozwiążemy to razem - napisz do mnie na priv!"
                ],
                sentiment_type=SentimentType.COMPLAINT,
                follow_up_required=True,
                escalate_to_human=True
            ),

            # Pytania o cenę i dostępność
            ResponseTemplate(
                trigger_keywords=["cena", "koszt", "ile kosztuje", "dostępność", "gdzie kupić", "zamówienie"],
                response_templates=[
                    "💰 Informacje o cenach i dostępności wyślę Ci na prywatną wiadomość!",
                    "🛒 Szczegóły dotyczące zamówienia otrzymasz w wiadomości prywatnej!",
                    "📦 Napisz do mnie prywatnie, a prześlę Ci aktualny cennik i warunki!",
                    "💝 Chętnie pomogę z zamówieniem - napisz do mnie na priv!"
                ],
                sentiment_type=SentimentType.QUESTION,
                follow_up_required=True,
                escalate_to_human=False
            ),

            # Prośby o informacje zdrowotne
            ResponseTemplate(
                trigger_keywords=["choroba", "lek", "medyczne", "leczenie", "diagnoza", "objawy"],
                response_templates=[
                    "⚕️ Dziękuję za pytanie! Ze względów bezpieczeństwa, sprawy medyczne omawiam tylko prywatnie z odpowiednim specjalistą.",
                    "🩺 To ważne pytanie medyczne. Napisz do mnie prywatnie, żeby skierować Cię do właściwej osoby.",
                    "💊 Kwestie zdrowotne wymagają indywidualnego podejścia. Skontaktuję się z Tobą prywatnie!",
                    "🔬 Sprawy medyczne omawiamy zawsze indywidualnie. Napisz do mnie na priv!"
                ],
                sentiment_type=SentimentType.QUESTION,
                follow_up_required=True,
                escalate_to_human=True
            ),

            # Testimonialsи опыт
            ResponseTemplate(
                trigger_keywords=["efekt", "rezultat", "poprawa", "zmiana", "doświadczenie", "testuje"],
                response_templates=[
                    "🌟 Fantastycznie, że dzielisz się swoim doświadczeniem! Inne osoby na pewno docenią!",
                    "💪 To wspaniałe! Twoje doświadczenie może zainspirować innych!",
                    "✨ Dziękuję za podzielenie się rezultatami! To motywuje całą społeczność!",
                    "🎯 Cudownie słyszeć o Twoich postępach! Trzymamy kciuki za dalsze sukcesy!"
                ],
                sentiment_type=SentimentType.POSITIVE,
                follow_up_required=False,
                escalate_to_human=False
            ),

            # Ogólne komentarze neutralne
            ResponseTemplate(
                trigger_keywords=["ok", "dzięki", "rozumiem", "dobra", "spoko"],
                response_templates=[
                    "👍 Dzięki za komentarz!",
                    "😊 Miło Cię słyszeć!",
                    "✅ Super, że jesteś z nami!",
                    "🙂 Dzięki za odzew!"
                ],
                sentiment_type=SentimentType.NEUTRAL,
                follow_up_required=False,
                escalate_to_human=False
            )
        ]

    async def execute_task(self, task: Task) -> AgentResult:
        """Wykonanie zadania obsługi interakcji"""
        try:
            if task.task_type == "respond_to_comment":
                return await self._respond_to_comment(task.data)
            elif task.task_type == "handle_message":
                return await self._handle_private_message(task.data)
            elif task.task_type == "monitor_interactions":
                return await self._monitor_interactions(task.data)
            elif task.task_type == "analyze_sentiment":
                return await self._analyze_interaction_sentiment(task.data)
            elif task.task_type == "escalate_issue":
                return await self._escalate_to_human(task.data)
            else:
                return AgentResult(success=False, error=f"Nieznany typ zadania: {task.task_type}")

        except Exception as e:
            self.logger.error(f"Błąd wykonania zadania {task.name}: {e}")
            return AgentResult(success=False, error=str(e))

    async def _monitor_interactions(self, data: Dict[str, Any]) -> AgentResult:
        """Monitorowanie nowych interakcji"""
        try:
            # Pobranie nowych komentarzy
            comments = await self.facebook_client.get_recent_comments(
                since=self.last_check_time
            )

            # Pobranie nowych wiadomości
            messages = await self.facebook_client.get_recent_messages(
                since=self.last_check_time
            )

            interactions_processed = 0
            escalations_created = 0

            # Przetwarzanie komentarzy
            for comment in comments.get('data', []):
                context = await self._create_interaction_context(comment, InteractionType.COMMENT)
                result = await self._process_interaction(context)

                if result.success:
                    interactions_processed += 1
                    if result.data.get('escalated'):
                        escalations_created += 1

            # Przetwarzanie wiadomości
            for message in messages.get('data', []):
                context = await self._create_interaction_context(message, InteractionType.MESSAGE)
                result = await self._process_interaction(context)

                if result.success:
                    interactions_processed += 1
                    if result.data.get('escalated'):
                        escalations_created += 1

            # Aktualizacja czasu ostatniej kontroli
            self.last_check_time = datetime.now()

            return AgentResult(
                success=True,
                data={
                    'interactions_processed': interactions_processed,
                    'escalations_created': escalations_created,
                    'monitoring_period': data.get('period', 'default'),
                    'timestamp': datetime.now().isoformat()
                }
            )

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    async def _create_interaction_context(self, interaction_data: Dict[str, Any],
                                          interaction_type: InteractionType) -> InteractionContext:
        """Utworzenie kontekstu interakcji"""

        # Analiza sentymentu
        sentiment = await self._analyze_sentiment(interaction_data.get('message', ''))

        # Określenie priorytetu
        priority = await self._calculate_interaction_priority(
            interaction_data.get('message', ''),
            sentiment,
            interaction_data.get('author_id')
        )

        return InteractionContext(
            interaction_id=interaction_data.get('id'),
            interaction_type=interaction_type,
            author_name=interaction_data.get('from', {}).get('name', 'Unknown'),
            author_id=interaction_data.get('from', {}).get('id', 'unknown'),
            content=interaction_data.get('message', ''),
            post_id=interaction_data.get('post_id'),
            timestamp=datetime.fromisoformat(interaction_data.get('created_time', datetime.now().isoformat())),
            sentiment=sentiment,
            requires_response=await self._requires_response(interaction_data.get('message', ''), sentiment),
            priority=priority
        )

    async def _analyze_sentiment(self, text: str) -> SentimentType:
        """Analiza sentymentu tekstu"""
        text_lower = text.lower()

        # Słowa kluczowe dla różnych sentymentów
        positive_keywords = ['świetnie', 'super', 'dziękuję', 'brawo', 'fantastycznie', 'polecam', 'doskonały', 'bomba']
        negative_keywords = ['źle', 'kiepski', 'problem', 'nie działa', 'rozczarowany', 'zwrot', 'oszustwo']
        question_keywords = ['jak', 'kiedy', 'ile', 'gdzie', 'czy', 'co', 'dlaczego', '?']
        complaint_keywords = ['skarga', 'reklamacja', 'problem', 'nie odpowiada', 'nie pomaga']

        # Sprawdzanie obecności słów kluczowych
        if any(keyword in text_lower for keyword in complaint_keywords):
            return SentimentType.COMPLAINT
        elif any(keyword in text_lower for keyword in question_keywords):
            return SentimentType.QUESTION
        elif any(keyword in text_lower for keyword in positive_keywords):
            return SentimentType.POSITIVE
        elif any(keyword in text_lower for keyword in negative_keywords):
            return SentimentType.NEGATIVE
        else:
            return SentimentType.NEUTRAL

    async def _calculate_interaction_priority(self, text: str, sentiment: SentimentType,
                                              author_id: str) -> int:
        """Obliczenie priorytetu interakcji (1-10, 10 = najwyższy)"""
        priority = 5  # Bazowy priorytet

        # Sentyment wpływa na priorytet
        if sentiment == SentimentType.COMPLAINT:
            priority += 4
        elif sentiment == SentimentType.NEGATIVE:
            priority += 2
        elif sentiment == SentimentType.QUESTION:
            priority += 1
        elif sentiment == SentimentType.POSITIVE:
            priority -= 1

        # Słowa kluczowe awaryjne
        emergency_keywords = ['zwrot pieniędzy', 'oszustwo', 'prawnik', 'sąd', 'UOKiK']
        if any(keyword in text.lower() for keyword in emergency_keywords):
            priority = 10

        # VIP klienci (sprawdzenie w historii)
        is_vip = await self._is_vip_customer(author_id)
        if is_vip:
            priority += 2

        # Ograniczenie do 1-10
        return max(1, min(10, priority))

    async def _is_vip_customer(self, author_id: str) -> bool:
        """Sprawdzenie czy klient jest VIP"""
        # Sprawdź historię interakcji
        customer_history = await self.memory_manager.get_data('customer_interactions', {'author_id': author_id})

        if customer_history:
            # VIP jeśli ma dużo pozytywnych interakcji lub jest częstym klientem
            total_interactions = len(customer_history)
            positive_interactions = sum(1 for interaction in customer_history
                                        if interaction.get('sentiment') == 'positive')

            return total_interactions > 10 or (positive_interactions / max(total_interactions, 1)) > 0.8

        return False

    async def _requires_response(self, text: str, sentiment: SentimentType) -> bool:
        """Sprawdzenie czy interakcja wymaga odpowiedzi"""
        # Zawsze odpowiadaj na pytania i skargi
        if sentiment in [SentimentType.QUESTION, SentimentType.COMPLAINT, SentimentType.NEGATIVE]:
            return True

        # Nie odpowiadaj na bardzo krótkie komentarze pozytywne
        if sentiment == SentimentType.POSITIVE and len(text.strip()) < 5:
            return False

        # Nie odpowiadaj na same emoji
        if re.match(r'^[\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+$', text):
            return False

        return True

    async def _process_interaction(self, context: InteractionContext) -> AgentResult:
        """Przetwarzanie pojedynczej interakcji"""
        try:
            # Sprawdź czy nie jest to spam lub duplikat
            if await self._is_spam_or_duplicate(context):
                return AgentResult(success=True, data={'action': 'ignored_spam'})

            # Zapisz interakcję do pamięci
            await self._save_interaction_to_memory(context)

            response_data = {'escalated': False, 'responded': False}

            # Sprawdź czy wymaga eskalacji
            if await self._should_escalate(context):
                escalation_result = await self._escalate_to_human({'context': context.__dict__})
                response_data['escalated'] = escalation_result.success
                self.escalations_created += 1

            # Jeśli wymaga odpowiedzi, wygeneruj ją
            if context.requires_response and not response_data['escalated']:
                response_result = await self._generate_and_send_response(context)
                response_data['responded'] = response_result.success
                response_data['response_content'] = response_result.data.get(
                    'content') if response_result.success else None

            self.interactions_handled += 1

            return AgentResult(success=True, data=response_data)

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    async def _is_spam_or_duplicate(self, context: InteractionContext) -> bool:
        """Sprawdzenie czy interakcja to spam lub duplikat"""
        # Sprawdź czy ten sam użytkownik nie napisał tego samego niedawno
        recent_interactions = await self.memory_manager.get_recent_data(
            'customer_interactions',
            hours=24,
            filters={'author_id': context.author_id}
        )

        for interaction in recent_interactions:
            if interaction.get('content', '').lower().strip() == context.content.lower().strip():
                return True

        # Sprawdź spam patterns
        spam_patterns = [
            r'^(.)\1{10,}$',  # Powtarzające się znaki
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # Linki
            r'^[A-Z\s!]{20,}$',  # Same wielkie litery
        ]

        for pattern in spam_patterns:
            if re.search(pattern, context.content):
                return True

        return False

    async def _should_escalate(self, context: InteractionContext) -> bool:
        """Sprawdzenie czy interakcja powinna być eskalowana"""
        # Automatyczna eskalacja dla skarg i problemów
        if context.sentiment in [SentimentType.COMPLAINT, SentimentType.NEGATIVE]:
            return True

        # Eskalacja dla słów kluczowych
        escalation_triggers = [
            'prawnik', 'sąd', 'uokik', 'oszustwo', 'policja',
            'zwrot pieniędzy', 'reklamacja', 'nie odpowiada',
            'lek', 'choroba', 'leczenie', 'diagnoza'
        ]

        return any(trigger in context.content.lower() for trigger in escalation_triggers)

    async def _generate_and_send_response(self, context: InteractionContext) -> AgentResult:
        """Generowanie i wysyłanie odpowiedzi"""
        try:
            # Znajdź odpowiedni szablon
            template = await self._find_best_template(context)

            if not template:
                # Użyj ogólnej odpowiedzi
                response_content = "Dziękuję za komentarz! Jeśli masz pytania, napisz do mnie prywatnie. 😊"
            else:
                # Wygeneruj odpowiedź na podstawie szablonu
                response_content = await self._generate_response_from_template(template, context)

            # Dodaj personalizację
            personalized_response = await self._personalize_response(response_content, context)

            # Wyślij odpowiedź
            if context.interaction_type == InteractionType.COMMENT:
                send_result = await self.facebook_client.reply_to_comment(
                    comment_id=context.interaction_id,
                    message=personalized_response
                )
            else:  # MESSAGE
                send_result = await self.facebook_client.send_private_message(
                    user_id=context.author_id,
                    message=personalized_response
                )

            if send_result['success']:
                # Zapisz odpowiedź do pamięci
                await self._save_response_to_memory(context, personalized_response)

                return AgentResult(
                    success=True,
                    data={
                        'content': personalized_response,
                        'template_used': template.sentiment_type.value if template else 'generic',
                        'response_time': (datetime.now() - context.timestamp).total_seconds()
                    }
                )
            else:
                return AgentResult(success=False, error=send_result.get('error'))

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    async def _find_best_template(self, context: InteractionContext) -> Optional[ResponseTemplate]:
        """Znajdowanie najlepszego szablonu odpowiedzi"""
        best_template = None
        best_score = 0

        for template in self.response_templates:
            score = 0

            # Dopasowanie sentymentu
            if template.sentiment_type == context.sentiment:
                score += 10

            # Dopasowanie słów kluczowych
            content_lower = context.content.lower()
            matching_keywords = sum(1 for keyword in template.trigger_keywords
                                    if keyword in content_lower)
            score += matching_keywords * 2

            # Preferuj szablon o najwyższym score
            if score > best_score:
                best_score = score
                best_template = template

        return best_template if best_score > 0 else None

    async def _generate_response_from_template(self, template: ResponseTemplate,
                                               context: InteractionContext) -> str:
        """Generowanie odpowiedzi z szablonu"""
        import random

        # Wybierz losowy szablon z listy
        base_response = random.choice(template.response_templates)

        # Jeśli szablon ma placeholder {}, wstaw specyficzną odpowiedź
        if '{}' in base_response:
            specific_answer = await self._generate_specific_answer(context)
            base_response = base_response.format(specific_answer)

        return base_response

    async def _generate_specific_answer(self, context: InteractionContext) -> str:
        """Generowanie specyficznej odpowiedzi na pytanie"""
        content_lower = context.content.lower()

        # FAQ - często zadawane pytania
        if 'ile kosztuje' in content_lower or 'cena' in content_lower:
            return "Aktualny cennik wyślę Ci na prywatną wiadomość wraz ze wszystkimi promocjami!"

        elif 'jak długo' in content_lower or 'kiedy efekt' in content_lower:
            return "Pierwsze efekty można odczuć już po 2-4 tygodniach regularnego stosowania. Pełny efekt po 3-4 miesiącach."

        elif 'jak stosować' in content_lower or 'dawka' in content_lower:
            return "Szczegółowe instrukcje stosowania otrzymasz wraz z produktem. Zawsze zgodnie z zaleceniami na opakowaniu!"

        elif 'gdzie kupić' in content_lower or 'jak zamówić' in content_lower:
            return "Możesz zamawiać bezpośrednio przez stronę Zinzino lub przez mnie. Napisz na priv po szczegóły!"

        elif 'test balance' in content_lower:
            return "Test Balance pokazuje Twój stosunek omega-3 do omega-6. Można go wykonać przed i po suplementacji!"

        elif 'skutki uboczne' in content_lower or 'bezpieczne' in content_lower:
            return "Produkty Zinzino są naturalne i bezpieczne. W razie wątpliwości skonsultuj się z lekarzem."

        else:
            return "Szczegółowe informacje chętnie przekażę na prywatnej wiadomości!"

    async def _personalize_response(self, response: str, context: InteractionContext) -> str:
        """Personalizacja odpowiedzi"""
        # Dodaj imię jeśli jest dostępne i stosowne
        if context.author_name and context.author_name != 'Unknown':
            first_name = context.author_name.split()[0]

            # Dodaj imię do niektórych odpowiedzi
            if context.sentiment in [SentimentType.QUESTION, SentimentType.POSITIVE]:
                if not any(greeting in response for greeting in ['Cześć', 'Dzień dobry', first_name]):
                    response = f"Cześć {first_name}! {response}"

        # Dodaj czas odpowiedzi
        current_hour = datetime.now().hour
        if current_hour < 12:
            time_greeting = "Dzień dobry"
        elif current_hour < 18:
            time_greeting = "Dzień dobry"
        else:
            time_greeting = "Dobry wieczór"

        # Dla formalnych odpowiedzi
        if context.sentiment == SentimentType.COMPLAINT:
            if not response.startswith(time_greeting):
                response = f"{time_greeting}! {response}"

        return response

    async def _save_interaction_to_memory(self, context: InteractionContext):
        """Zapisanie interakcji do pamięci"""
        interaction_data = {
            'interaction_id': context.interaction_id,
            'type': context.interaction_type.value,
            'author_name': context.author_name,
            'author_id': context.author_id,
            'content': context.content,
            'post_id': context.post_id,
            'timestamp': context.timestamp.isoformat(),
            'sentiment': context.sentiment.value,
            'priority': context.priority,
            'requires_response': context.requires_response,
            'agent': self.name
        }

        await self.memory_manager.store_data('customer_interactions', interaction_data)

    async def _save_response_to_memory(self, context: InteractionContext, response: str):
        """Zapisanie odpowiedzi do pamięci"""
        response_data = {
            'original_interaction_id': context.interaction_id,
            'response_content': response,
            'response_time': datetime.now().isoformat(),
            'agent': self.name,
            'template_used': True
        }

        await self.memory_manager.store_data('agent_responses', response_data)

    async def _respond_to_comment(self, data: Dict[str, Any]) -> AgentResult:
        """Odpowiedź na konkretny komentarz"""
        try:
            comment_id = data.get('comment_id')
            if not comment_id:
                return AgentResult(success=False, error="Brak ID komentarza")

            # Pobranie szczegółów komentarza
            comment_details = await self.facebook_client.get_comment_details(comment_id)

            if not comment_details['success']:
                return AgentResult(success=False, error="Nie można pobrać szczegółów komentarza")

            # Utworzenie kontekstu
            context = await self._create_interaction_context(
                comment_details['data'],
                InteractionType.COMMENT
            )

            # Przetworzenie interakcji
            return await self._process_interaction(context)

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    async def _handle_private_message(self, data: Dict[str, Any]) -> AgentResult:
        """Obsługa prywatnej wiadomości"""
        try:
            message_data = data.get('message_data')
            if not message_data:
                return AgentResult(success=False, error="Brak danych wiadomości")

            # Utworzenie kontekstu
            context = await self._create_interaction_context(
                message_data,
                InteractionType.MESSAGE
            )

            # Prywatne wiadomości mają wyższy priorytet
            context.priority = min(10, context.priority + 2)

            # Przetworzenie interakcji
            return await self._process_interaction(context)

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    async def _escalate_to_human(self, data: Dict[str, Any]) -> AgentResult:
        """Eskalacja sprawy do człowieka"""
        try:
            context_data = data.get('context', {})

            escalation_data = {
                'escalation_id': f"ESC_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'interaction_context': context_data,
                'escalation_reason': await self._determine_escalation_reason(context_data),
                'priority': context_data.get('priority', 5),
                'created_at': datetime.now().isoformat(),
                'status': 'pending',
                'assigned_to': None,
                'agent': self.name
            }

            # Zapisz eskalację
            await self.memory_manager.store_data('escalations', escalation_data)

            # Wyślij powiadomienie (w przyszłości można zintegrować z systemem ticketów)
            self.logger.warning(f"Eskalacja utworzona: {escalation_data['escalation_id']}")

            return AgentResult(
                success=True,
                data={
                    'escalation_id': escalation_data['escalation_id'],
                    'reason': escalation_data['escalation_reason']
                }
            )

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    async def _determine_escalation_reason(self, context_data: Dict[str, Any]) -> str:
        """Określenie powodu eskalacji"""
        content = context_data.get('content', '').lower()
        sentiment = context_data.get('sentiment', '')

        if sentiment == 'complaint':
            return "Skarga klienta wymagająca interwencji człowieka"
        elif any(word in content for word in ['prawnik', 'sąd', 'oszustwo']):
            return "Sprawa prawna - wymaga natychmiastowej uwagi"
        elif any(word in content for word in ['choroba', 'lek', 'leczenie']):
            return "Pytanie medyczne - wymaga konsultacji specjalisty"
        elif any(word in content for word in ['zwrot', 'reklamacja']):
            return "Sprawa zwrotu/reklamacji"
        else:
            return "Złożona sprawa wymagająca ludzkiej interwencji"

    async def _analyze_interaction_sentiment(self, data: Dict[str, Any]) -> AgentResult:
        """Analiza sentymentu interakcji"""
        try:
            text = data.get('text', '')
            if not text:
                return AgentResult(success=False, error="Brak tekstu do analizy")

            sentiment = await self._analyze_sentiment(text)

            # Dodatkowa analiza
            analysis = {
                'sentiment': sentiment.value,
                'confidence': await self._calculate_sentiment_confidence(text, sentiment),
                'keywords_found': await self._extract_keywords(text),
                'requires_immediate_attention': sentiment in [SentimentType.COMPLAINT, SentimentType.NEGATIVE],
                'suggested_response_type': await self._suggest_response_type(sentiment)
            }

            return AgentResult(success=True, data=analysis)

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    async def _calculate_sentiment_confidence(self, text: str, sentiment: SentimentType) -> float:
        """Obliczenie pewności analizy sentymentu"""
        text_lower = text.lower()

        # Liczenie słów kluczowych dla danego sentymentu
        keyword_mapping = {
            SentimentType.POSITIVE: ['świetnie', 'super', 'dziękuję', 'brawo', 'fantastycznie'],
            SentimentType.NEGATIVE: ['źle', 'kiepski', 'problem', 'nie działa', 'rozczarowany'],
            SentimentType.QUESTION: ['jak', 'kiedy', 'ile', 'gdzie', 'czy', '?'],
            SentimentType.COMPLAINT: ['skarga', 'reklamacja', 'problem', 'zwrot'],
            SentimentType.NEUTRAL: ['ok', 'dzięki', 'rozumiem']
        }

        relevant_keywords = keyword_mapping.get(sentiment, [])
        found_keywords = sum(1 for keyword in relevant_keywords if keyword in text_lower)

        # Oblicz pewność na podstawie liczby znalezionych słów kluczowych
        if found_keywords == 0:
            return 0.3  # Niska pewność
        elif found_keywords == 1:
            return 0.6  # Średnia pewność
        else:
            return 0.9  # Wysoka pewność

    async def _extract_keywords(self, text: str) -> List[str]:
        """Wyodrębnienie słów kluczowych z tekstu"""
        all_keywords = [
            'cena', 'koszt', 'zamówienie', 'dostępność', 'efekt', 'rezultat',
            'jak stosować', 'dawka', 'test balance', 'omega-3', 'balance oil',
            'problem', 'skarga', 'zwrot', 'reklamacja', 'choroba', 'lek'
        ]

        text_lower = text.lower()
        found_keywords = [keyword for keyword in all_keywords if keyword in text_lower]

        return found_keywords

    async def _suggest_response_type(self, sentiment: SentimentType) -> str:
        """Sugerowanie typu odpowiedzi"""
        response_types = {
            SentimentType.POSITIVE: "appreciation",
            SentimentType.NEGATIVE: "problem_solving",
            SentimentType.QUESTION: "informational",
            SentimentType.COMPLAINT: "escalation",
            SentimentType.NEUTRAL: "acknowledgment"
        }

        return response_types.get(sentiment, "generic")

    async def get_engagement_statistics(self) -> Dict[str, Any]:
        """Pobranie statystyk zaangażowania"""
        # Pobierz dane z ostatnich 30 dni
        recent_interactions = await self.memory_manager.get_recent_data(
            'customer_interactions',
            days=30
        )

        recent_responses = await self.memory_manager.get_recent_data(
            'agent_responses',
            days=30
        )

        if not recent_interactions:
            return {'message': 'Brak danych o interakcjach'}

        # Analiza sentymentów
        sentiment_counts = {}
        for interaction in recent_interactions:
            sentiment = interaction.get('sentiment', 'unknown')
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

        # Analiza czasów odpowiedzi
        response_times = []
        for response in recent_responses:
            if 'response_time' in response:
                response_times.append(float(response['response_time']))

        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        # Statystyki escalacji
        escalations = await self.memory_manager.get_recent_data('escalations', days=30)

        return {
            'period': '30 days',
            'total_interactions': len(recent_interactions),
            'total_responses': len(recent_responses),
            'response_rate': len(recent_responses) / len(recent_interactions) if recent_interactions else 0,
            'sentiment_breakdown': sentiment_counts,
            'average_response_time_seconds': round(avg_response_time, 2),
            'escalations_created': len(escalations),
            'escalation_rate': len(escalations) / len(recent_interactions) if recent_interactions else 0,
            'interactions_per_day': len(recent_interactions) / 30,
            'most_common_keywords': await self._get_most_common_keywords(recent_interactions)
        }

    async def _get_most_common_keywords(self, interactions: List[Dict]) -> Dict[str, int]:
        """Pobranie najczęstszych słów kluczowych"""
        keyword_counts = {}

        for interaction in interactions:
            content = interaction.get('content', '').lower()

            # Lista słów kluczowych do śledzenia
            keywords_to_track = [
                'cena', 'koszt', 'balance oil', 'omega-3', 'test balance',
                'efekt', 'rezultat', 'jak stosować', 'gdzie kupić', 'zamówienie',
                'problem', 'nie działa', 'skarga', 'zwrot', 'choroba'
            ]

            for keyword in keywords_to_track:
                if keyword in content:
                    keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1

        # Sortuj według częstotliwości
        sorted_keywords = dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True))

        # Zwróć top 10
        return dict(list(sorted_keywords.items())[:10])

    async def get_pending_escalations(self) -> List[Dict[str, Any]]:
        """Pobranie oczekujących eskalacji"""
        escalations = await self.memory_manager.get_data(
            'escalations',
            {'status': 'pending'}
        )

        # Sortuj według priorytetu i daty
        sorted_escalations = sorted(
            escalations,
            key=lambda x: (x.get('priority', 0), x.get('created_at', '')),
            reverse=True
        )

        return sorted_escalations

    async def resolve_escalation(self, escalation_id: str, resolution: str) -> AgentResult:
        """Rozwiązanie eskalacji"""
        try:
            # Aktualizuj status eskalacji
            escalation_update = {
                'status': 'resolved',
                'resolution': resolution,
                'resolved_at': datetime.now().isoformat(),
                'resolved_by': self.name
            }

            # W rzeczywistości tutaj byłaby aktualizacja w bazie danych
            # Dla uproszczenia logujemy rozwiązanie
            self.logger.info(f"Eskalacja {escalation_id} rozwiązana: {resolution}")

            return AgentResult(
                success=True,
                data={
                    'escalation_id': escalation_id,
                    'status': 'resolved',
                    'resolution': resolution
                }
            )

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    async def set_auto_response_mode(self, enabled: bool) -> AgentResult:
        """Włączenie/wyłączenie trybu automatycznych odpowiedzi"""
        try:
            self.config['auto_response_enabled'] = enabled

            mode_text = "włączony" if enabled else "wyłączony"
            self.logger.info(f"Tryb automatycznych odpowiedzi {mode_text}")

            return AgentResult(
                success=True,
                data={
                    'auto_response_enabled': enabled,
                    'timestamp': datetime.now().isoformat()
                }
            )

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    async def add_custom_response_template(self, template_data: Dict[str, Any]) -> AgentResult:
        """Dodanie niestandardowego szablonu odpowiedzi"""
        try:
            new_template = ResponseTemplate(
                trigger_keywords=template_data.get('trigger_keywords', []),
                response_templates=template_data.get('response_templates', []),
                sentiment_type=SentimentType(template_data.get('sentiment_type', 'neutral')),
                follow_up_required=template_data.get('follow_up_required', False),
                escalate_to_human=template_data.get('escalate_to_human', False)
            )

            self.response_templates.append(new_template)

            self.logger.info(f"Dodano nowy szablon odpowiedzi dla sentymentu: {new_template.sentiment_type.value}")

            return AgentResult(
                success=True,
                data={
                    'template_added': True,
                    'sentiment_type': new_template.sentiment_type.value,
                    'trigger_keywords': new_template.trigger_keywords
                }
            )

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    async def update_response_delay(self, delay_seconds: int) -> AgentResult:
        """Aktualizacja opóźnienia odpowiedzi"""
        try:
            self.response_delay = max(60, min(3600, delay_seconds))  # 1 min - 1 godzina

            self.logger.info(f"Opóźnienie odpowiedzi ustawione na {self.response_delay} sekund")

            return AgentResult(
                success=True,
                data={
                    'response_delay': self.response_delay,
                    'timestamp': datetime.now().isoformat()
                }
            )

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    async def get_customer_interaction_history(self, customer_id: str) -> Dict[str, Any]:
        """Pobranie historii interakcji z konkretnym klientem"""
        try:
            interactions = await self.memory_manager.get_data(
                'customer_interactions',
                {'author_id': customer_id}
            )

            responses = await self.memory_manager.get_data(
                'agent_responses',
                {'original_interaction_id': {'$in': [i.get('interaction_id') for i in interactions]}}
            )

            if not interactions:
                return {'message': f'Brak historii dla klienta {customer_id}'}

            # Analiza klienta
            sentiment_counts = {}
            for interaction in interactions:
                sentiment = interaction.get('sentiment', 'unknown')
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

            # Ostatnia interakcja
            last_interaction = max(interactions, key=lambda x: x.get('timestamp', ''))

            return {
                'customer_id': customer_id,
                'total_interactions': len(interactions),
                'total_responses': len(responses),
                'sentiment_breakdown': sentiment_counts,
                'last_interaction_date': last_interaction.get('timestamp'),
                'last_interaction_content': last_interaction.get('content', '')[:100] + '...',
                'customer_sentiment_trend': await self._analyze_customer_sentiment_trend(interactions),
                'is_vip': await self._is_vip_customer(customer_id)
            }

        except Exception as e:
            return {'error': str(e)}

    async def _analyze_customer_sentiment_trend(self, interactions: List[Dict]) -> str:
        """Analiza trendu sentymentu klienta"""
        if len(interactions) < 3:
            return "insufficient_data"

        # Sortuj po dacie
        sorted_interactions = sorted(interactions, key=lambda x: x.get('timestamp', ''))

        # Sprawdź trend w ostatnich interakcjach
        recent_sentiments = [i.get('sentiment') for i in sorted_interactions[-3:]]

        positive_count = recent_sentiments.count('positive')
        negative_count = recent_sentiments.count('negative')

        if positive_count > negative_count:
            return "improving"
        elif negative_count > positive_count:
            return "declining"
        else:
            return "stable"