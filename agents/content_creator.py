"""
Content Creator Agent - Agent tworzący treści na Facebook
Specjalizuje się w tworzeniu angażujących postów o Zinzino
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import random
from dataclasses import dataclass

from core.agent_base import BaseAgent, AgentResult
from core.task_queue import Task
from core.memory_manager import MemoryManager
from core.facebook_client import FacebookClient


@dataclass
class ContentTemplate:
    """Szablon treści"""
    category: str
    template: str
    hashtags: List[str]
    image_suggestions: List[str]
    target_audience: str
    best_time: str


@dataclass
class PostPerformance:
    """Wydajność posta"""
    post_id: str
    likes: int
    comments: int
    shares: int
    reach: int
    engagement_rate: float
    created_at: datetime


class ContentCreatorAgent(BaseAgent):
    """
    Agent tworzący treści na Facebook

    Funkcjonalności:
    - Tworzenie angażujących postów
    - Optymalizacja treści pod kątem algorytmu Facebook
    - Personalizacja treści dla różnych grup odbiorców
    - Analiza wydajności postów
    - Automatyczne planowanie publikacji
    """

    def __init__(self, name: str, config: Dict[str, Any],
                 facebook_client: FacebookClient, memory_manager: MemoryManager):
        super().__init__(name, config)
        self.facebook_client = facebook_client
        self.memory_manager = memory_manager

        # Konfiguracja specyficzna dla twórcy treści
        self.content_categories = config.get('content_categories', [])
        self.posting_schedule = config.get('posting_schedule', {})
        self.target_audiences = config.get('target_audiences', {})

        # Szablony treści
        self.content_templates = self._load_content_templates()

        # Statystyki wydajności
        self.post_performance_history = []

        # AI dla tworzenia treści
        self.content_ai_settings = config.get('content_ai', {})

        self.logger.info(f"Content Creator Agent {name} zainicjalizowany")

    def _load_content_templates(self) -> List[ContentTemplate]:
        """Ładowanie szablonów treści"""
        templates = [
            ContentTemplate(
                category="product_showcase",
                template="""🌟 Odkryj moc Zinzino {product_name}! 

{product_benefits}

✨ Dlaczego warto:
{reasons}

💪 Już ponad {testimonial_count} osób przekonało się o skuteczności!

👇 Podziel się swoim doświadczeniem w komentarzach!

#Zinzino #Zdrowie #Wellness {additional_hashtags}""",
                hashtags=["#Zinzino", "#Zdrowie", "#Wellness", "#Suplementy", "#ZdrowyStylŻycia"],
                image_suggestions=["product_photo", "before_after", "lifestyle_photo"],
                target_audience="health_conscious",
                best_time="18:00-20:00"
            ),

            ContentTemplate(
                category="testimonial",
                template="""💬 "Zinzino zmieniło moje życie!" - {customer_name}

{testimonial_text}

🎯 Rezultaty po {time_period}:
{results_list}

✅ To może być Twoja historia!

➡️ Sprawdź jak zacząć swoją transformację

#ZinzinoTestimonial #Transformacja #Sukces {additional_hashtags}""",
                hashtags=["#ZinzinoTestimonial", "#Transformacja", "#Sukces", "#Zdrowie"],
                image_suggestions=["customer_photo", "results_chart", "success_story"],
                target_audience="potential_customers",
                best_time="19:00-21:00"
            ),

            ContentTemplate(
                category="educational",
                template="""🧠 Czy wiesz, że {educational_fact}?

📚 Dzisiaj wyjaśniamy:
{explanation}

🔍 Ciekawostki:
{interesting_facts}

💡 Praktyczne zastosowanie:
{practical_tips}

👨‍⚕️ Zinzino wykorzystuje tę wiedzę w produktach takich jak {related_products}

💬 Jakie masz pytania? Zapytaj w komentarzach!

#EdukacjaZdrowotna #Zinzino #Wiedza {additional_hashtags}""",
                hashtags=["#EdukacjaZdrowotna", "#Zinzino", "#Wiedza", "#Nauka"],
                image_suggestions=["infographic", "educational_chart", "science_visual"],
                target_audience="knowledge_seekers",
                best_time="12:00-14:00"
            ),

            ContentTemplate(
                category="lifestyle",
                template="""🌅 Poranek z Zinzino = {morning_routine}

☀️ Moja rutyna:
{routine_steps}

💪 Efekty:
{benefits_felt}

🎯 Kluczowe produkty:
{key_products}

✨ Jak wygląda Twój poranek? Podziel się w komentarzach!

#PoraneZZinzino #ZdrowyPoranek #Rutyna {additional_hashtags}""",
                hashtags=["#PoraneZZinzino", "#ZdrowyPoranek", "#Rutyna", "#Lifestyle"],
                image_suggestions=["morning_routine", "product_in_use", "lifestyle_photo"],
                target_audience="lifestyle_enthusiasts",
                best_time="07:00-09:00"
            ),

            ContentTemplate(
                category="motivation",
                template="""💪 {motivational_quote}

🎯 Pamiętaj:
{motivation_points}

🌟 Zinzino wspiera Cię w drodze do:
{goals_list}

✅ Już dziś możesz zrobić pierwszy krok!

🔥 Co motywuje Ciebie? Napisz w komentarzach!

#Motywacja #Zinzino #CelZdrowotny {additional_hashtags}""",
                hashtags=["#Motywacja", "#Zinzino", "#CelZdrowotny", "#Inspiracja"],
                image_suggestions=["motivational_quote", "success_visual", "goal_achievement"],
                target_audience="goal_oriented",
                best_time="16:00-18:00"
            ),

            ContentTemplate(
                category="behind_scenes",
                template="""🎬 Za kulisami Zinzino: {behind_scenes_topic}

👀 Co dziś robimy:
{activities}

🔬 Ciekawostki z produkcji:
{production_facts}

👥 Zespół pracuje nad:
{current_projects}

💝 To wszystko dla Was - naszych klientów!

#ZaKulisamiZinzino #Zespół #Jakość {additional_hashtags}""",
                hashtags=["#ZaKulisamiZinzino", "#Zespół", "#Jakość", "#Transparentność"],
                image_suggestions=["team_photo", "production_process", "behind_scenes"],
                target_audience="brand_loyal",
                best_time="14:00-16:00"
            )
        ]

        return templates

    async def execute_task(self, task: Task) -> AgentResult:
        """Wykonanie zadania tworzenia treści"""
        try:
            if task.task_type == "create_post":
                return await self._create_single_post(task.data)
            elif task.task_type == "create_daily_posts":
                return await self._create_daily_posts(task.data)
            elif task.task_type == "optimize_post":
                return await self._optimize_existing_post(task.data)
            elif task.task_type == "analyze_performance":
                return await self._analyze_post_performance(task.data)
            else:
                return AgentResult(success=False, error=f"Nieznany typ zadania: {task.task_type}")

        except Exception as e:
            self.logger.error(f"Błąd wykonania zadania {task.name}: {e}")
            return AgentResult(success=False, error=str(e))

    async def _create_single_post(self, data: Dict[str, Any]) -> AgentResult:
        """Tworzenie pojedynczego posta"""
        try:
            # Określenie kategorii i szablonu
            category = data.get('category', 'product_showcase')
            template = self._get_template_by_category(category)

            if not template:
                return AgentResult(success=False, error=f"Brak szablonu dla kategorii: {category}")

            # Generowanie treści
            content = await self._generate_content(template, data)

            # Optymalizacja treści
            optimized_content = await self._optimize_content(content, template.target_audience)

            # Dodanie hashtag'ów
            final_content = await self._add_hashtags(optimized_content, template.hashtags,
                                                     data.get('additional_hashtags', []))

            # Publikacja na Facebook
            post_result = await self.facebook_client.create_post(
                content=final_content,
                image_url=data.get('image_url'),
                schedule_time=data.get('schedule_time')
            )

            if post_result['success']:
                # Zapisanie do pamięci
                await self._save_post_to_memory(post_result['post_id'], final_content, category)

                return AgentResult(
                    success=True,
                    data={
                        'post_id': post_result['post_id'],
                        'content': final_content,
                        'category': category,
                        'published_at': datetime.now().isoformat()
                    }
                )
            else:
                return AgentResult(success=False, error=post_result.get('error', 'Błąd publikacji'))

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    async def _create_daily_posts(self, data: Dict[str, Any]) -> AgentResult:
        """Tworzenie dziennych postów"""
        try:
            posts_to_create = data.get('count', 3)
            created_posts = []

            # Pobranie optymalnych kategorii na dziś
            daily_categories = await self._get_optimal_daily_categories()

            for i in range(posts_to_create):
                if i < len(daily_categories):
                    category = daily_categories[i]
                else:
                    category = random.choice(self.content_categories)

                # Dane dla pojedynczego posta
                post_data = {
                    'category': category,
                    'schedule_time': await self._calculate_optimal_posting_time(i),
                    'personalization_data': await self._get_personalization_data(category)
                }

                # Tworzenie posta
                result = await self._create_single_post(post_data)

                if result.success:
                    created_posts.append(result.data)
                    await asyncio.sleep(2)  # Krótka przerwa między postami
                else:
                    self.logger.warning(f"Nie udało się utworzyć posta {i + 1}: {result.error}")

            return AgentResult(
                success=True,
                data={
                    'posts_created': len(created_posts),
                    'posts': created_posts,
                    'date': datetime.now().date().isoformat()
                }
            )

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    def _get_template_by_category(self, category: str) -> Optional[ContentTemplate]:
        """Pobranie szablonu dla kategorii"""
        for template in self.content_templates:
            if template.category == category:
                return template
        return None

    async def _generate_content(self, template: ContentTemplate, data: Dict[str, Any]) -> str:
        """Generowanie treści na podstawie szablonu"""
        # Pobranie danych do personalizacji
        personalization_data = await self._get_personalization_data(template.category)

        # Łączenie danych
        content_data = {**personalization_data, **data.get('personalization_data', {})}

        # Wypełnienie szablonu
        try:
            content = template.template.format(**content_data)
        except KeyError as e:
            # Jeśli brakuje danych, użyj domyślnych wartości
            self.logger.warning(f"Brak danych personalizacji: {e}")
            content = await self._fill_template_with_defaults(template, content_data)

        return content

    async def _get_personalization_data(self, category: str) -> Dict[str, Any]:
        """Pobranie danych do personalizacji"""
        # Dane z pamięci systemu
        memory_data = await self.memory_manager.get_context('personalization')

        # Dane specyficzne dla kategorii
        category_data = {
            'product_showcase': {
                'product_name': 'Balance Oil+',
                'product_benefits': '🐟 Optymalne omega-3\n🌿 Naturalne składniki\n⚖️ Przywraca równowagę',
                'reasons': '✓ Klinicznie przetestowane\n✓ Certyfikowane składniki\n✓ Widoczne rezultaty',
                'testimonial_count': '10,000',
                'additional_hashtags': '#BalanceOil #Omega3'
            },
            'testimonial': {
                'customer_name': 'Anna K.',
                'testimonial_text': 'Po 3 miesiącach stosowania Balance Oil+ czuję się fantastycznie!',
                'time_period': '3 miesiące',
                'results_list': '• Więcej energii\n• Lepsza koncentracja\n• Spokojniejszy sen',
                'additional_hashtags': '#RealResults #CustomerLove'
            },
            'educational': {
                'educational_fact': 'omega-3 i omega-6 powinny być w równowadze 3:1',
                'explanation': 'Nowoczesna dieta często zawiera zbyt dużo omega-6, co prowadzi do nierównowagi.',
                'interesting_facts': '• 95% ludzi ma zaburzoną równowagę\n• Test Balance pokazuje dokładny stosunek\n• Zinzino pomaga przywrócić harmonię',
                'practical_tips': 'Sprawdź swój stosunek testem Balance i dostosuj suplementację.',
                'related_products': 'Balance Oil+ i Balance Test',
                'additional_hashtags': '#BalanceTest #Science'
            },
            'lifestyle': {
                'morning_routine': 'energia na cały dzień!',
                'routine_steps': '1️⃣ Szklanka wody\n2️⃣ Balance Oil+\n3️⃣ 10 minut medytacji\n4️⃣ Zdrowe śniadanie',
                'benefits_felt': '• Stały poziom energii\n• Lepsza koncentracja\n• Pozytywne nastawienie',
                'key_products': '🔸 Balance Oil+\n🔸 Protect+\n🔸 Xtend+',
                'additional_hashtags': '#MorningRoutine #HealthyLiving'
            },
            'motivation': {
                'motivational_quote': '"Zdrowie to nie wszystko, ale bez zdrowia wszystko to nic"',
                'motivation_points': '• Każdy dzień to nowa szansa\n• Małe kroki = wielkie zmiany\n• Twoje zdrowie = Twoja inwestycja',
                'goals_list': '💪 Lepszej kondycji\n🧠 Większej energii\n❤️ Optymalnego zdrowia',
                'additional_hashtags': '#MotivationMonday #HealthGoals'
            },
            'behind_scenes': {
                'behind_scenes_topic': 'Jak powstaje Balance Oil+',
                'activities': '🔬 Testowanie jakości\n📦 Pakowanie produktów\n✅ Kontrola standardów',
                'production_facts': '• 100% naturalny\n• Bez GMO\n• Certyfikowane surowce',
                'current_projects': '🌱 Nowe formuły\n🌍 Zrównoważone opakowania\n🔬 Badania kliniczne',
                'additional_hashtags': '#Quality #Production'
            }
        }

        return category_data.get(category, {})

    async def _fill_template_with_defaults(self, template: ContentTemplate, data: Dict[str, Any]) -> str:
        """Wypełnienie szablonu domyślnymi wartościami"""
        default_values = {
            'product_name': 'Balance Oil+',
            'customer_name': 'Klient Zinzino',
            'testimonial_count': '1000',
            'time_period': '30 dni',
            'additional_hashtags': '#Zinzino'
        }

        # Łączenie z domyślnymi wartościami
        merged_data = {**default_values, **data}

        try:
            return template.template.format(**merged_data)
        except KeyError:
            # Jeśli nadal brakuje danych, zwróć podstawową wersję
            return f"🌟 Odkryj moc Zinzino!\n\n{template.hashtags[0]} {template.hashtags[1]}"

    async def _optimize_content(self, content: str, target_audience: str) -> str:
        """Optymalizacja treści pod kątem grupy docelowej i algorytmu Facebook"""

        # Dodanie emoji dla lepszego engagement
        content = await self._enhance_with_emojis(content)

        # Optymalizacja długości
        content = await self._optimize_length(content)

        # Dodanie call-to-action
        content = await self._add_call_to_action(content, target_audience)

        return content

    async def _enhance_with_emojis(self, content: str) -> str:
        """Dodanie emoji dla lepszego zaangażowania"""
        # Prosta optymalizacja - dodanie emoji na początku jeśli brakuje
        if not any(char in content[:10] for char in ['🌟', '💪', '✨', '🎯', '💡']):
            content = f"✨ {content}"

        return content

    async def _optimize_length(self, content: str) -> str:
        """Optymalizacja długości posta"""
        # Facebook preferuje posty 40-80 znaków dla lepszego zasięgu
        # ale dla treści edukacyjnych może być dłużej
        if len(content) > 2000:
            # Skróć jeśli za długi
            content = content[:1900] + "...\n\n📖 Więcej informacji w komentarzach!"

        return content

    async def _add_call_to_action(self, content: str, target_audience: str) -> str:
        """Dodanie wezwania do działania"""
        cta_options = {
            'health_conscious': [
                "\n\n👇 Podziel się swoim doświadczeniem!",
                "\n\n💬 Jakie masz pytania o zdrowie?",
                "\n\n🎯 Sprawdź jak zacząć swoją podróż ze zdrowiem!"
            ],
            'potential_customers': [
                "\n\n📞 Napisz do mnie po więcej informacji!",
                "\n\n🎯 Gotowy na zmianę? Sprawdź nasze produkty!",
                "\n\n✅ Zamów test Balance już dziś!"
            ],
            'knowledge_seekers': [
                "\n\n🤔 Jakie masz pytania na ten temat?",
                "\n\n📚 Chcesz wiedzieć więcej? Zapytaj!",
                "\n\n💡 Podziel się swoją wiedzą w komentarzach!"
            ]
        }

        # Dodaj CTA jeśli nie ma już wezwania do działania
        if not any(phrase in content.lower() for phrase in ['podziel się', 'napisz', 'zapytaj', 'sprawdź']):
            cta_list = cta_options.get(target_audience, cta_options['health_conscious'])
            content += random.choice(cta_list)

        return content

    async def _add_hashtags(self, content: str, base_hashtags: List[str],
                            additional_hashtags: List[str]) -> str:
        """Dodanie hashtag'ów do posta"""
        # Sprawdź czy już są hashtagi
        if '#' not in content:
            all_hashtags = base_hashtags + additional_hashtags
            # Ogranicz do 10 hashtag'ów (optymalne dla Facebook)
            hashtags = all_hashtags[:10]
            content += f"\n\n{' '.join(hashtags)}"

        return content

    async def _get_optimal_daily_categories(self) -> List[str]:
        """Pobranie optymalnych kategorii na dziś"""
        # Analiza historii postów
        recent_posts = await self.memory_manager.get_recent_data('posts', days=7)

        # Zliczenie kategorii z ostatniego tygodnia
        category_counts = {}
        for post in recent_posts:
            category = post.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1

        # Wybór najmniej używanych kategorii
        available_categories = [t.category for t in self.content_templates]
        sorted_categories = sorted(available_categories,
                                   key=lambda x: category_counts.get(x, 0))

        # Dzień tygodnia wpływa na wybór kategorii
        weekday = datetime.now().weekday()

        if weekday == 0:  # Poniedziałek - motywacja
            return ['motivation', 'lifestyle', 'educational']
        elif weekday == 4:  # Piątek - za kulisami, lifestyle
            return ['behind_scenes', 'lifestyle', 'testimonial']
        else:  # Inne dni - mix
            return sorted_categories[:3]

    async def _calculate_optimal_posting_time(self, post_index: int) -> Optional[str]:
        """Obliczenie optymalnego czasu publikacji"""
        base_times = ['09:00', '13:00', '18:00', '20:00']

        if post_index < len(base_times):
            return base_times[post_index]

        return None

    async def _save_post_to_memory(self, post_id: str, content: str, category: str):
        """Zapisanie posta do pamięci"""
        post_data = {
            'post_id': post_id,
            'content': content,
            'category': category,
            'created_at': datetime.now().isoformat(),
            'agent': self.name
        }

        await self.memory_manager.store_data('posts', post_data)

    async def _analyze_post_performance(self, data: Dict[str, Any]) -> AgentResult:
        """Analiza wydajności postów"""
        try:
            post_id = data.get('post_id')
            if not post_id:
                return AgentResult(success=False, error="Brak ID posta")

            # Pobranie statystyk z Facebook
            stats = await self.facebook_client.get_post_insights(post_id)

            if stats['success']:
                performance = PostPerformance(
                    post_id=post_id,
                    likes=stats['data'].get('likes', 0),
                    comments=stats['data'].get('comments', 0),
                    shares=stats['data'].get('shares', 0),
                    reach=stats['data'].get('reach', 0),
                    engagement_rate=stats['data'].get('engagement_rate', 0.0),
                    created_at=datetime.now()
                )

                # Zapisanie do historii
                self.post_performance_history.append(performance)

                # Analiza i rekomendacje
                insights = await self._generate_content_insights(performance)

                return AgentResult(
                    success=True,
                    data={
                        'performance': performance.__dict__,
                        'insights': insights
                    }
                )
            else:
                return AgentResult(success=False, error=stats.get('error'))

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    async def _generate_content_insights(self, performance: PostPerformance) -> Dict[str, Any]:
        """Generowanie insights na podstawie wydajności"""
        insights = {
            'performance_rating': 'average',
            'recommendations': [],
            'optimal_posting_times': [],
            'content_suggestions': []
        }

        # Ocena wydajności
        if performance.engagement_rate > 0.05:  # 5%
            insights['performance_rating'] = 'excellent'
        elif performance.engagement_rate > 0.03:  # 3%
            insights['performance_rating'] = 'good'
        elif performance.engagement_rate < 0.01:  # 1%
            insights['performance_rating'] = 'poor'

        # Rekomendacje
        if performance.comments > performance.likes:
            insights['recommendations'].append("Post generuje dyskusję - więcej pytań otwartych")

        if performance.shares > performance.likes * 0.1:
            insights['recommendations'].append("Treść jest 'shareable' - więcej wartościowej edukacji")

        if performance.reach < 100:
            insights['recommendations'].append("Niska organiczna reach - rozważ boost lub inne hashtagi")

        return insights

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Podsumowanie wydajności agenta"""
        if not self.post_performance_history:
            return {'message': 'Brak danych o wydajności'}

        recent_posts = self.post_performance_history[-30:]  # Ostatnie 30 postów

        avg_engagement = sum(p.engagement_rate for p in recent_posts) / len(recent_posts)
        total_reach = sum(p.reach for p in recent_posts)
        total_interactions = sum(p.likes + p.comments + p.shares for p in recent_posts)

        return {
            'posts_analyzed': len(recent_posts),
            'average_engagement_rate': round(avg_engagement, 4),
            'total_reach': total_reach,
            'total_interactions': total_interactions,
            'best_performing_post': max(recent_posts, key=lambda p: p.engagement_rate).__dict__,
            'content_categories_performance': await self._analyze_category_performance(recent_posts)
        }

    async def _analyze_category_performance(self, posts: List[PostPerformance]) -> Dict[str, float]:
        """Analiza wydajności kategorii treści"""
        category_performance = {}

        # Pobierz kategorie z pamięci dla każdego posta
        for post in posts:
            post_data = await self.memory_manager.get_data('posts', {'post_id': post.post_id})
            if post_data:
                category = post_data[0].get('category', 'unknown')
                if category not in category_performance:
                    category_performance[category] = []
                category_performance[category].append(post.engagement_rate)

        # Oblicz średnią dla każdej kategorii
        return {
            category: sum(rates) / len(rates)
            for category, rates in category_performance.items()
        }