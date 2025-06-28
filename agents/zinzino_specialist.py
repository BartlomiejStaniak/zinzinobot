"""
Zinzino Specialist Agent - Ekspert ds. produktów i wiedzy o Zinzino
Specjalizuje się w udzielaniu szczegółowych informacji o produktach i firmie
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
from dataclasses import dataclass
from enum import Enum

from core.agent_base import BaseAgent, AgentResult
from core.task_queue import Task
from core.memory_manager import MemoryManager


class QuestionCategory(Enum):
    """Kategorie pytań"""
    PRODUCT_INFO = "product_info"
    USAGE_INSTRUCTIONS = "usage_instructions"
    HEALTH_BENEFITS = "health_benefits"
    SCIENTIFIC_RESEARCH = "scientific_research"
    PRICING = "pricing"
    ORDERING = "ordering"
    COMPANY_INFO = "company_info"
    TESTIMONIALS = "testimonials"
    SIDE_EFFECTS = "side_effects"
    COMPARISONS = "comparisons"


@dataclass
class ProductInfo:
    """Informacje o produkcie"""
    name: str
    category: str
    key_ingredients: List[str]
    benefits: List[str]
    usage_instructions: str
    scientific_backing: List[str]
    price_range: str
    target_audience: List[str]
    contraindications: List[str]


@dataclass
class KnowledgeEntry:
    """Wpis w bazie wiedzy"""
    question: str
    answer: str
    category: QuestionCategory
    keywords: List[str]
    confidence_level: float
    last_updated: datetime
    source: str


class ZinzinoSpecialistAgent(BaseAgent):
    """
    Agent ekspert ds. Zinzino

    Funkcjonalności:
    - Szczegółowe informacje o produktach
    - Odpowiadanie na pytania naukowe
    - Rekomendacje produktów
    - Analiza potrzeb klientów
    - Edukacja o zdrowiu i wellness
    - Wsparcie sprzedażowe
    """

    def __init__(self, name: str, config: Dict[str, Any], memory_manager: MemoryManager):
        super().__init__(name, config)
        self.memory_manager = memory_manager

        # Baza wiedzy o produktach
        self.product_database = self._initialize_product_database()

        # Baza wiedzy FAQ
        self.knowledge_base = self._initialize_knowledge_base()

        # Konfiguracja specjalisty
        self.expertise_areas = config.get('expertise_areas', [])
        self.recommendation_engine_config = config.get('recommendation_engine', {})

        # Statystyki
        self.questions_answered = 0
        self.recommendations_made = 0
        self.customer_consultations = 0

        self.logger.info(f"Zinzino Specialist Agent {name} zainicjalizowany")

    def _initialize_product_database(self) -> Dict[str, ProductInfo]:
        """Inicjalizacja bazy danych produktów"""
        return {
            "balance_oil_plus": ProductInfo(
                name="BalanceOil+ Premium",
                category="Omega-3 Supplements",
                key_ingredients=[
                    "Olej rybi (anchois, sardynki, makrela)",
                    "Olej z oliwek extra virgin",
                    "Witamina D3",
                    "Tokoferole (witamina E)"
                ],
                benefits=[
                    "Przywraca równowagę omega-3/omega-6",
                    "Wspiera funkcje serca, mózgu i wzroku",
                    "Wzmacnia system immunologiczny",
                    "Poprawia kondycję skóry i włosów",
                    "Wspiera funkcje kognitywne"
                ],
                usage_instructions="0,15ml na kg masy ciała dziennie. Najlepiej rano na czczo lub z posiłkiem.",
                scientific_backing=[
                    "Klinicznie przetestowane - wyniki w 120 dni",
                    "Certyfikat IFOS (5 gwiazdek)",
                    "Badania naukowe opublikowane w peer-reviewed journals",
                    "Test Balance - pomiar przed i po suplementacji"
                ],
                price_range="300-400 PLN",
                target_audience=[
                    "Osoby z niedoborem omega-3",
                    "Sportowcy i aktywni fizycznie",
                    "Osoby dbające o zdrowie serca",
                    "Kobiety w ciąży i karmiące"
                ],
                contraindications=[
                    "Alergia na ryby",
                    "Przyjmowanie antykoagulantów (konsultacja z lekarzem)",
                    "Dzieci poniżej 4 roku życia"
                ]
            ),

            "protect_plus": ProductInfo(
                name="Protect+ Premium",
                category="Immune Support",
                key_ingredients=[
                    "Beta-glukany z drożdży",
                    "Witamina C",
                    "Witamina D3",
                    "Cynk",
                    "Selenium"
                ],
                benefits=[
                    "Wzmacnia naturalną odporność",
                    "Wspiera regenerację organizmu",
                    "Chroni komórki przed stresem oksydacyjnym",
                    "Poprawia jakość snu",
                    "Zwiększa energię"
                ],
                usage_instructions="1-2 tabletki dziennie, najlepiej wieczorem.",
                scientific_backing=[
                    "Składniki potwierdzone przez EFSA",
                    "Beta-glukany 1,3/1,6 najwyższej jakości",
                    "Badania kliniczne nad beta-glukanami"
                ],
                price_range="200-300 PLN",
                target_audience=[
                    "Osoby z obniżoną odpornością",
                    "W okresie zwiększonego stresu",
                    "Sportowcy",
                    "Osoby starsze"
                ],
                contraindications=[
                    "Ciąża i karmienie (konsultacja z lekarzem)",
                    "Choroby autoimmunologiczne (konsultacja wymagana)"
                ]
            ),

            "xtend_plus": ProductInfo(
                name="Xtend+ Premium",
                category="Multivitamin",
                key_ingredients=[
                    "22 mikroskładniki",
                    "Koenzym Q10",
                    "Fitosterole",
                    "Alfa-karoten",
                    "Beta-karoten"
                ],
                benefits=[
                    "Kompleksowe wsparcie organizmu",
                    "Zmniejsza zmęczenie",
                    "Wspiera metabolizm energetyczny",
                    "Chroni przed stresem oksydacyjnym",
                    "Wspiera funkcje układu nerwowego"
                ],
                usage_instructions="2 miękkie kapsułki dziennie z posiłkiem.",
                scientific_backing=[
                    "Formuła oparta na badaniach naukowych",
                    "Składniki w bioaktywnych formach",
                    "Wysokie wskaźniki wchłaniania"
                ],
                price_range="250-350 PLN",
                target_audience=[
                    "Osoby aktywne zawodowo",
                    "W okresie zwiększonego wysiłku",
                    "Starsze osoby",
                    "Osoby na dietach redukcyjnych"
                ],
                contraindications=[
                    "Przedawkowanie witamin rozpuszczalnych w tłuszczach",
                    "Jednoczesne przyjmowanie innych multiwitamin"
                ]
            ),

            "balance_test": ProductInfo(
                name="Balance Test",
                category="Diagnostic Test",
                key_ingredients=["Test krwi z palca"],
                benefits=[
                    "Dokładny pomiar stosunku omega-3/omega-6",
                    "Indywidualne rekomendacje dawkowania",
                    "Monitoring postępów suplementacji",
                    "Weryfikacja skuteczności terapii"
                ],
                usage_instructions="Test wykonywany przed rozpoczęciem suplementacji i po 120 dniach.",
                scientific_backing=[
                    "Certyfikowane laboratorium",
                    "Metoda chromatografii gazowej",
                    "Walidacja naukowa testu"
                ],
                price_range="300-400 PLN",
                target_audience=[
                    "Wszyscy rozpoczynający suplementację",
                    "Osoby monitorujące zdrowie",
                    "Pacjenci z problemami kardiovascularnymi"
                ],
                contraindications=["Brak przeciwwskazań"]
            )
        }

    def _initialize_knowledge_base(self) -> List[KnowledgeEntry]:
        """Inicjalizacja bazy wiedzy FAQ"""
        return [
            KnowledgeEntry(
                question="Jak długo trzeba czekać na pierwsze efekty BalanceOil+?",
                answer="Pierwsze pozytywne efekty można odczuć już po 2-4 tygodniach regularnego stosowania. Pełne przywrócenie równowagi omega-3/omega-6 następuje po 120 dniach, co potwierdza test Balance.",
                category=QuestionCategory.PRODUCT_INFO,
                keywords=["efekty", "jak długo", "kiedy", "balance oil"],
                confidence_level=0.95,
                last_updated=datetime.now(),
                source="Clinical studies"
            ),

            KnowledgeEntry(
                question="Czy można przyjmować BalanceOil+ w ciąży?",
                answer="BalanceOil+ jest bezpieczny w ciąży i podczas karmienia piersią. Omega-3 są kluczowe dla rozwoju mózgu dziecka. Zalecamy jednak konsultację z lekarzem prowadzącym przed rozpoczęciem suplementacji.",
                category=QuestionCategory.SIDE_EFFECTS,
                keywords=["ciąża", "karmienie", "bezpieczny", "dziecko"],
                confidence_level=0.90,
                last_updated=datetime.now(),
                source="Medical guidelines"
            ),

            KnowledgeEntry(
                question="Jaka jest różnica między BalanceOil a BalanceOil+?",
                answer="BalanceOil+ to premium wersja z dodatkiem witaminy D3 oraz olejem z oliwek extra virgin najwyższej jakości. Ma lepszy smak i dodatkowe korzyści zdrowotne związane z witaminą D.",
                category=QuestionCategory.COMPARISONS,
                keywords=["różnica", "balance oil", "plus", "premium"],
                confidence_level=0.98,
                last_updated=datetime.now(),
                source="Product specifications"
            ),

            KnowledgeEntry(
                question="Czy produkty Zinzino mają certyfikaty jakości?",
                answer="Tak! Produkty Zinzino posiadają certyfikat IFOS (5 gwiazdek) dla czystości i potencji, są produkowane zgodnie z GMP, oraz regularnie testowane przez niezależne laboratoria. To gwarancja najwyższej jakości.",
                category=QuestionCategory.COMPANY_INFO,
                keywords=["certyfikaty", "jakość", "IFOS", "GMP"],
                confidence_level=0.99,
                last_updated=datetime.now(),
                source="Official certifications"
            ),

            KnowledgeEntry(
                question="Ile kosztuje zestaw startowy Zinzino?",
                answer="Ceny różnią się w zależności od zestawu. Podstawowy zestaw z BalanceOil+ i testem Balance to około 600-700 PLN. Dokładne ceny i aktualne promocje otrzymasz w wiadomości prywatnej.",
                category=QuestionCategory.PRICING,
                keywords=["cena", "koszt", "zestaw", "ile"],
                confidence_level=0.85,
                last_updated=datetime.now(),
                source="Current price list"
            ),

            KnowledgeEntry(
                question="Czy można łączyć produkty Zinzino z lekami?",
                answer="Większość produktów Zinzino można bezpiecznie łączyć z lekami. Wyjątek to antykoagulanty (leki rozrzedzające krew) - tutaj wymagana jest konsultacja lekarska. Zawsze informuj lekarza o suplementacji.",
                category=QuestionCategory.SIDE_EFFECTS,
                keywords=["leki", "interakcje", "antykoagulanty", "bezpieczeństwo"],
                confidence_level=0.88,
                last_updated=datetime.now(),
                source="Medical literature"
            )
        ]

    async def execute_task(self, task: Task) -> AgentResult:
        """Wykonanie zadania eksperta"""
        try:
            if task.task_type == "answer_question":
                return await self._answer_question(task.data)
            elif task.task_type == "product_recommendation":
                return await self._recommend_products(task.data)
            elif task.task_type == "analyze_customer_needs":
                return await self._analyze_customer_needs(task.data)
            elif task.task_type == "provide_scientific_info":
                return await self._provide_scientific_information(task.data)
            elif task.task_type == "create_consultation":
                return await self._create_consultation_plan(task.data)
            else:
                return AgentResult(success=False, error=f"Nieznany typ zadania: {task.task_type}")

        except Exception as e:
            self.logger.error(f"Błąd wykonania zadania {task.name}: {e}")
            return AgentResult(success=False, error=str(e))

    async def _answer_question(self, data: Dict[str, Any]) -> AgentResult:
        """Odpowiadanie na pytania o Zinzino"""
        try:
            question = data.get('question', '')
            customer_context = data.get('customer_context', {})

            if not question:
                return AgentResult(success=False, error="Brak pytania")

            # Analiza pytania
            question_analysis = await self._analyze_question(question)

            # Wyszukanie odpowiedzi w bazie wiedzy
            knowledge_answer = await self._search_knowledge_base(question, question_analysis['category'])

            # Jeśli nie ma w bazie, generuj odpowiedź
            if not knowledge_answer:
                generated_answer = await self._generate_answer(question, question_analysis)
                answer_data = generated_answer
            else:
                answer_data = {
                    'answer': knowledge_answer.answer,
                    'confidence': knowledge_answer.confidence_level,
                    'source': knowledge_answer.source,
                    'category': knowledge_answer.category.value
                }

            # Personalizacja odpowiedzi
            personalized_answer = await self._personalize_answer(
                answer_data['answer'],
                customer_context,
                question_analysis
            )

            # Dodanie rekomendacji produktów jeśli stosowne
            product_recommendations = await self._get_relevant_product_recommendations(
                question_analysis['category']
            )

            self.questions_answered += 1

            return AgentResult(
                success=True,
                data={
                    'question': question,
                    'answer': personalized_answer,
                    'category': question_analysis['category'].value,
                    'confidence': answer_data.get('confidence', 0.8),
                    'product_recommendations': product_recommendations,
                    'follow_up_suggestions': await self._generate_follow_up_questions(question_analysis),
                    'requires_consultation': await self._requires_personal_consultation(question_analysis)
                }
            )

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    async def _analyze_question(self, question: str) -> Dict[str, Any]:
        """Analiza pytania klienta"""
        question_lower = question.lower()

        # Kategoryzacja pytania
        category = QuestionCategory.PRODUCT_INFO  # domyślna

        if any(word in question_lower for word in ['cena', 'koszt', 'ile kosztuje', 'płatność']):
            category = QuestionCategory.PRICING
        elif any(word in question_lower for word in ['jak stosować', 'dawka', 'ile brać']):
            category = QuestionCategory.USAGE_INSTRUCTIONS
        elif any(word in question_lower for word in ['efekt', 'korzyść', 'pomaga', 'działa']):
            category = QuestionCategory.HEALTH_BENEFITS
        elif any(word in question_lower for word in ['badania', 'nauka', 'dowód', 'klinicznie']):
            category = QuestionCategory.SCIENTIFIC_RESEARCH
        elif any(word in question_lower for word in ['zamówić', 'kupić', 'gdzie', 'dostępność']):
            category = QuestionCategory.ORDERING
        elif any(word in question_lower for word in ['skutki uboczne', 'bezpieczny', 'przeciwwskazania']):
            category = QuestionCategory.SIDE_EFFECTS
        elif any(word in question_lower for word in ['porównanie', 'różnica', 'lepszy']):
            category = QuestionCategory.COMPARISONS
        elif any(word in question_lower for word in ['opinia', 'recenzja', 'doświadczenie']):
            category = QuestionCategory.TESTIMONIALS
        elif any(word in question_lower for word in ['firma', 'zinzino', 'historia', 'o firmie']):
            category = QuestionCategory.COMPANY_INFO

        # Analiza intencji
        intent = self._analyze_question_intent(question_lower)

        # Ekstraktowanie produktów wspomnianych w pytaniu
        mentioned_products = self._extract_mentioned_products(question_lower)

        return {
            'category': category,
            'intent': intent,
            'mentioned_products': mentioned_products,
            'urgency': self._assess_question_urgency(question_lower),
            'complexity': self._assess_question_complexity(question_lower)
        }

    def _analyze_question_intent(self, question: str) -> str:
        """Analiza intencji pytania"""
        if any(word in question for word in ['polecasz', 'rekomendacja', 'co wybrać']):
            return 'seeking_recommendation'
        elif any(word in question for word in ['czy mogę', 'czy można', 'bezpieczne']):
            return 'seeking_safety_confirmation'
        elif any(word in question for word in ['jak', 'kiedy', 'ile']):
            return 'seeking_instructions'
        elif any(word in question for word in ['dlaczego', 'czemu', 'mechanizm']):
            return 'seeking_explanation'
        elif any(word in question for word in ['zamówić', 'kupić', 'gdzie']):
            return 'seeking_purchase_info'
        elif any(word in question for word in ['problem', 'nie działa', 'nie pomaga']):
            return 'reporting_issue'
        else:
            return 'general_inquiry'

    def _extract_mentioned_products(self, question: str) -> List[str]:
        """Ekstraktowanie produktów wspomnianych w pytaniu"""
        mentioned = []

        product_keywords = {
            'balance_oil_plus': ['balance oil', 'balanceoil', 'omega'],
            'protect_plus': ['protect', 'odporność', 'immunitet'],
            'xtend_plus': ['xtend', 'witaminy', 'multiwitamina'],
            'balance_test': ['test', 'balance test', 'badanie']
        }

        for product, keywords in product_keywords.items():
            if any(keyword in question for keyword in keywords):
                mentioned.append(product)

        return mentioned

    def _assess_question_urgency(self, question: str) -> str:
        """Ocena pilności pytania"""
        urgent_keywords = ['pilne', 'natychmiast', 'problem', 'źle się czuję', 'niepokojące']

        if any(keyword in question for keyword in urgent_keywords):
            return 'high'
        elif any(word in question for word in ['kiedy', 'jak szybko']):
            return 'medium'
        else:
            return 'low'

    def _assess_question_complexity(self, question: str) -> str:
        """Ocena złożoności pytania"""
        complex_keywords = ['mechanizm', 'biochemia', 'interakcje', 'metabolizm', 'farmakokinetyka']

        if any(keyword in question for keyword in complex_keywords):
            return 'high'
        elif len(question.split()) > 15:
            return 'medium'
        else:
            return 'low'

    async def _search_knowledge_base(self, question: str, category: QuestionCategory) -> Optional[KnowledgeEntry]:
        """Wyszukiwanie w bazie wiedzy"""
        question_lower = question.lower()
        best_match = None
        best_score = 0

        for entry in self.knowledge_base:
            score = 0

            # Bonus za kategorię
            if entry.category == category:
                score += 5

            # Punkty za słowa kluczowe
            for keyword in entry.keywords:
                if keyword in question_lower:
                    score += 2

            # Punkty za podobieństwo pytania
            if any(word in question_lower for word in entry.question.lower().split()):
                score += 1

            if score > best_score:
                best_score = score
                best_match = entry

        # Zwróć tylko jeśli wynik jest wystarczająco dobry
        return best_match if best_score >= 3 else None

    async def _generate_answer(self, question: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generowanie odpowiedzi jeśli nie ma w bazie wiedzy"""
        category = analysis['category']

        # Szablon odpowiedzi na podstawie kategorii
        if category == QuestionCategory.PRODUCT_INFO:
            answer = await self._generate_product_info_answer(question, analysis)
        elif category == QuestionCategory.PRICING:
            answer = "Informacje o cenach są regularne aktualizowane. Aby otrzymać aktualny cennik z ewentualnymi promocjami, napisz do mnie na prywatną wiadomość!"
        elif category == QuestionCategory.USAGE_INSTRUCTIONS:
            answer = await self._generate_usage_answer(question, analysis)
        elif category == QuestionCategory.HEALTH_BENEFITS:
            answer = await self._generate_benefits_answer(question, analysis)
        elif category == QuestionCategory.ORDERING:
            answer = "Produkty Zinzino można zamawiać przez oficjalną stronę internetową lub bezpośrednio przez mnie. Napisz na priv, a pomogę Ci złożyć zamówienie!"
        else:
            answer = "To bardzo dobre pytanie! Aby udzielić Ci szczegółowej odpowiedzi, napisz do mnie prywatnie - będę mógł lepiej pomóc!"

        return {
            'answer': answer,
            'confidence': 0.7,
            'source': 'Generated response',
            'category': category.value
        }

    async def _generate_product_info_answer(self, question: str, analysis: Dict[str, Any]) -> str:
        """Generowanie odpowiedzi o produktach"""
        mentioned_products = analysis['mentioned_products']

        if mentioned_products:
            product_key = mentioned_products[0]
            if product_key in self.product_database:
                product = self.product_database[product_key]
                return f"""
📌 {product.name}

🔹 Główne składniki: {', '.join(product.key_ingredients[:3])}
🔹 Kluczowe korzyści: {', '.join(product.benefits[:3])}
🔹 Grupa docelowa: {', '.join(product.target_audience[:2])}

💡 Chcesz wiedzieć więcej? Napisz do mnie prywatnie po szczegółowe informacje!
                """.strip()

        return "Zinzino oferuje szeroki wybór produktów premium dla zdrowia i wellness. Napisz do mnie prywatnie, a dopasujemy najlepsze produkty do Twoich potrzeb!"

    async def _generate_usage_answer(self, question: str, analysis: Dict[str, Any]) -> str:
        """Generowanie odpowiedzi o sposobie użycia"""
        mentioned_products = analysis['mentioned_products']

        if mentioned_products:
            product_key = mentioned_products[0]
            if product_key in self.product_database:
                product = self.product_database[product_key]
                return f"""
📋 {product.name} - sposób stosowania:

{product.usage_instructions}

⚠️ Zawsze czytaj instrukcję na opakowaniu.
💡 Masz dodatkowe pytania? Napisz prywatnie!
                """.strip()

        return "Każdy produkt Zinzino ma szczegółowe instrukcje stosowania. Napisz do mnie prywatnie z pytaniem o konkretny produkt!"

    async def _generate_benefits_answer(self, question: str, analysis: Dict[str, Any]) -> str:
        """Generowanie odpowiedzi o korzyściach"""
        mentioned_products = analysis['mentioned_products']

        if mentioned_products:
            product_key = mentioned_products[0]
            if product_key in self.product_database:
                product = self.product_database[product_key]
                benefits_text = '\n'.join([f"✅ {benefit}" for benefit in product.benefits])
                return f"""
🌟 {product.name} - potwierdzone korzyści:

{benefits_text}

🔬 Wszystkie korzyści poparte badaniami naukowymi!
                """.strip()

        return "Produkty Zinzino oferują szerokie spektrum korzyści zdrowotnych. Napisz prywatnie po szczegółowe informacje!"

    async def _personalize_answer(self, answer: str, customer_context: Dict[str, Any],
                                  analysis: Dict[str, Any]) -> str:
        """Personalizacja odpowiedzi"""

        # Dodaj personalizację na podstawie kontekstu klienta
        customer_name = customer_context.get('name', '')
        customer_age = customer_context.get('age')
        customer_health_goals = customer_context.get('health_goals', [])

        personalized = answer

        # Dodaj imię jeśli dostępne
        if customer_name and not customer_name.startswith('Cześć'):
            personalized = f"Cześć {customer_name.split()[0]}! {personalized}"

        # Dostosuj do wieku
        if customer_age:
            if customer_age > 60:
                personalized += "\n\n👥 Dla osób w Twoim wieku szczególnie ważne jest regularne monitorowanie zdrowia."
            elif customer_age < 30:
                personalized += "\n\n🌱 W młodym wieku inwestycja w zdrowie przynosi największe korzyści!"

        # Dostosuj do celów zdrowotnych
        if customer_health_goals:
            if 'heart_health' in customer_health_goals:
                personalized += "\n\n❤️ Pamiętając o Twoich celach dotyczących zdrowia serca, omega-3 będą kluczowe."
            if 'immunity' in customer_health_goals:
                personalized += "\n\n🛡️ Dla wzmocnienia odporności polecam również Protect+."

        return personalized

    async def _get_relevant_product_recommendations(self, category: QuestionCategory) -> List[Dict[str, Any]]:
        """Pobranie relevantnych rekomendacji produktów"""
        recommendations = []

        if category in [QuestionCategory.HEALTH_BENEFITS, QuestionCategory.PRODUCT_INFO]:
            recommendations.append({
                'product': 'balance_oil_plus',
                'reason': 'Podstawa zdrowia - równowaga omega-3/omega-6',
                'priority': 1
            })

        if category == QuestionCategory.USAGE_INSTRUCTIONS:
            recommendations.append({
                'product': 'balance_test',
                'reason': 'Pozwala określić optymalne dawkowanie',
                'priority': 2
            })

        return recommendations

    async def _generate_follow_up_questions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generowanie pytań uzupełniających"""
        category = analysis['category']

        follow_ups = {
            QuestionCategory.PRODUCT_INFO: [
                "Czy masz jakieś konkretne cele zdrowotne?",
                "Czy przyjmujesz obecnie jakieś suplementy?",
                "Czy wykonywałeś kiedyś test Balance?"
            ],
            QuestionCategory.HEALTH_BENEFITS: [
                "Jak długo planujesz suplementację?",
                "Czy masz jakieś problemy zdrowotne?",
                "Czy jesteś aktywny fizycznie?"
            ],
            QuestionCategory.USAGE_INSTRUCTIONS: [
                "Czy masz doświadczenie z suplementami omega-3?",
                "Czy przyjmujesz jakieś leki?",
                "Jaka jest Twoja dieta?"
            ]
        }

        return follow_ups.get(category, [
            "Czy masz jeszcze jakieś pytania?",
            "Potrzebujesz pomocy z wyborem produktów?"
        ])

    async def _requires_personal_consultation(self, analysis: Dict[str, Any]) -> bool:
        """Sprawdzenie czy wymaga konsultacji osobistej"""

        # Wysokie priorytety dla konsultacji
        if analysis['urgency'] == 'high':
            return True

        if analysis['complexity'] == 'high':
            return True

        if analysis['category'] in [QuestionCategory.SIDE_EFFECTS, QuestionCategory.SCIENTIFIC_RESEARCH]:
            return True

        if analysis['intent'] == 'reporting_issue':
            return True

        return False

    async def _recommend_products(self, data: Dict[str, Any]) -> AgentResult:
        """Rekomendacja produktów na podstawie potrzeb klienta"""
        try:
            customer_profile = data.get('customer_profile', {})
            goals = data.get('health_goals', [])
            budget = data.get('budget_range', 'medium')

            # Analiza potrzeb klienta
            needs_analysis = await self._analyze_customer_needs({'customer_profile': customer_profile})

            # Generowanie rekomendacji
            recommendations = []

            # Podstawowa rekomendacja - Balance Oil+ dla wszystkich
            recommendations.append({
                'product': 'balance_oil_plus',
                'priority': 1,
                'reason': 'Podstawa zdrowia - 95% ludzi ma niedobór omega-3',
                'confidence': 0.95,
                'monthly_cost_range': '100-133 PLN'
            })

            # Dodatkowe rekomendacje na podstawie celów
            if 'immunity' in goals or 'immune_support' in goals:
                recommendations.append({
                    'product': 'protect_plus',
                    'priority': 2,
                    'reason': 'Wzmocnienie odporności, szczególnie w okresie jesienno-zimowym',
                    'confidence': 0.85,
                    'monthly_cost_range': '67-100 PLN'
                })

            if 'energy' in goals or 'fatigue' in goals:
                recommendations.append({
                    'product': 'xtend_plus',
                    'priority': 2,
                    'reason': 'Kompleksowe wsparcie energetyczne i redukcja zmęczenia',
                    'confidence': 0.80,
                    'monthly_cost_range': '83-117 PLN'
                })

            # Test Balance zawsze polecany
            recommendations.append({
                'product': 'balance_test',
                'priority': 1,
                'reason': 'Niezbędny do określenia stanu omega-3/omega-6 i monitorowania postępów',
                'confidence': 0.90,
                'monthly_cost_range': 'Jednorazowo 300-400 PLN'
            })

            # Personalizacja na podstawie budżetu
            recommendations = await self._filter_by_budget(recommendations, budget)

            # Utworzenie planu suplementacji
            supplementation_plan = await self._create_supplementation_plan(recommendations, customer_profile)

            self.recommendations_made += 1

            return AgentResult(
                success=True,
                data={
                    'recommendations': recommendations,
                    'supplementation_plan': supplementation_plan,
                    'total_monthly_cost': await self._calculate_total_cost(recommendations),
                    'customer_profile_analysis': needs_analysis.data if needs_analysis.success else {},
                    'next_steps': await self._generate_next_steps(recommendations)
                }
            )

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    async def _filter_by_budget(self, recommendations: List[Dict], budget: str) -> List[Dict]:
        """Filtrowanie rekomendacji według budżetu"""
        if budget == 'low':
            # Tylko najważniejsze produkty
            return [r for r in recommendations if r['priority'] == 1]
        elif budget == 'high':
            # Wszystkie rekomendacje
            return recommendations
        else:  # medium
            # Priorytet 1 i wybrane z priorytetu 2
            return [r for r in recommendations if r['priority'] <= 2][:3]

    async def _create_supplementation_plan(self, recommendations: List[Dict],
                                           customer_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Utworzenie planu suplementacji"""

        plan_phases = []

        # Faza 1: Start (pierwsze 30 dni)
        phase1_products = [r['product'] for r in recommendations if
                           r['product'] in ['balance_oil_plus', 'balance_test']]
        plan_phases.append({
            'phase': 1,
            'duration': '30 dni',
            'products': phase1_products,
            'goals': 'Rozpoczęcie suplementacji, wykonanie testu bazowego',
            'expected_effects': 'Pierwsze pozytywne sygnały, lepszy sen'
        })

        # Faza 2: Rozbudowa (dni 31-120)
        phase2_products = [r['product'] for r in recommendations]
        plan_phases.append({
            'phase': 2,
            'duration': '90 dni',
            'products': phase2_products,
            'goals': 'Pełna suplementacja, przywrócenie równowagi',
            'expected_effects': 'Znaczna poprawa energii, lepsze samopoczucie'
        })

        # Faza 3: Kontrola (dzień 120)
        plan_phases.append({
            'phase': 3,
            'duration': '1 dzień',
            'products': ['balance_test'],
            'goals': 'Kontrola postępów, weryfikacja skuteczności',
            'expected_effects': 'Potwierdzenie przywrócenia równowagi omega-3/omega-6'
        })

        return {
            'phases': plan_phases,
            'total_duration': '120 dni',
            'monitoring_schedule': 'Test Balance: dzień 0 i 120',
            'adjustment_points': 'Dzień 30 i 60 - możliwość dostosowania dawkowania'
        }

    async def _calculate_total_cost(self, recommendations: List[Dict]) -> Dict[str, str]:
        """Obliczenie całkowitego kosztu"""

        # Przybliżone koszty miesięczne
        monthly_costs = {
            'balance_oil_plus': 120,
            'protect_plus': 80,
            'xtend_plus': 100,
            'balance_test': 0  # Jednorazowo
        }

        one_time_costs = {
            'balance_test': 350
        }

        monthly_total = sum(monthly_costs.get(r['product'], 0) for r in recommendations)
        one_time_total = sum(one_time_costs.get(r['product'], 0) for r in recommendations)

        return {
            'monthly': f"{monthly_total} PLN",
            'one_time': f"{one_time_total} PLN",
            'first_month_total': f"{monthly_total + one_time_total} PLN"
        }

    async def _generate_next_steps(self, recommendations: List[Dict]) -> List[str]:
        """Generowanie kolejnych kroków dla klienta"""
        steps = [
            "1. Wykonaj test Balance aby poznać swój aktualny stan",
            "2. Rozpocznij od BalanceOil+ zgodnie z wynikami testu",
            "3. Monitoruj samopoczucie i prowadź dziennik zmian"
        ]

        if any(r['product'] == 'protect_plus' for r in recommendations):
            steps.append("4. Dodaj Protect+ szczególnie w okresie jesienno-zimowym")

        if any(r['product'] == 'xtend_plus' for r in recommendations):
            steps.append("5. Rozważ Xtend+ jeśli czujesz przewlekłe zmęczenie")

        steps.extend([
            "6. Po 120 dniach powtórz test Balance",
            "7. Dostosuj dawkowanie na podstawie wyników kontrolnych"
        ])

        return steps

    async def _analyze_customer_needs(self, data: Dict[str, Any]) -> AgentResult:
        """Analiza potrzeb klienta"""
        try:
            customer_profile = data.get('customer_profile', {})

            analysis = {
                'age_group': self._categorize_age(customer_profile.get('age')),
                'lifestyle_factors': self._analyze_lifestyle(customer_profile),
                'health_priorities': self._identify_health_priorities(customer_profile),
                'risk_factors': self._identify_risk_factors(customer_profile),
                'supplement_history': customer_profile.get('supplement_history', 'unknown')
            }

            # Określenie poziomu potrzeb
            needs_level = self._calculate_needs_level(analysis)

            # Rekomendacje w kontekście analizy
            contextual_recommendations = self._generate_contextual_recommendations(analysis)

            return AgentResult(
                success=True,
                data={
                    'analysis': analysis,
                    'needs_level': needs_level,
                    'priority_areas': contextual_recommendations['priority_areas'],
                    'recommended_approach': contextual_recommendations['approach'],
                    'consultation_recommendation': needs_level >= 8  # Wysoki poziom potrzeb
                }
            )

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    def _categorize_age(self, age: Optional[int]) -> str:
        """Kategoryzacja wieku"""
        if not age:
            return 'unknown'
        elif age < 25:
            return 'young_adult'
        elif age < 45:
            return 'adult'
        elif age < 65:
            return 'middle_aged'
        else:
            return 'senior'

    def _analyze_lifestyle(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza stylu życia"""
        lifestyle = {
            'activity_level': profile.get('activity_level', 'unknown'),
            'stress_level': profile.get('stress_level', 'unknown'),
            'diet_quality': profile.get('diet_quality', 'unknown'),
            'sleep_quality': profile.get('sleep_quality', 'unknown')
        }

        # Oblicz ogólny score lifestyle'u (1-10)
        score_mapping = {
            'poor': 2, 'low': 3, 'fair': 5, 'good': 7, 'excellent': 9, 'unknown': 5
        }

        scores = [score_mapping.get(value, 5) for value in lifestyle.values()]
        lifestyle['overall_score'] = sum(scores) / len(scores)

        return lifestyle

    def _identify_health_priorities(self, profile: Dict[str, Any]) -> List[str]:
        """Identyfikacja priorytetów zdrowotnych"""
        priorities = []

        # Na podstawie podanych celów
        goals = profile.get('health_goals', [])
        if 'heart_health' in goals:
            priorities.append('cardiovascular_health')
        if 'brain_health' in goals:
            priorities.append('cognitive_function')
        if 'immunity' in goals:
            priorities.append('immune_support')
        if 'energy' in goals:
            priorities.append('energy_metabolism')

        # Na podstawie problemów zdrowotnych
        issues = profile.get('health_issues', [])
        if 'fatigue' in issues:
            priorities.append('energy_metabolism')
        if 'frequent_infections' in issues:
            priorities.append('immune_support')
        if 'concentration_problems' in issues:
            priorities.append('cognitive_function')

        return list(set(priorities))  # Remove duplicates

    def _identify_risk_factors(self, profile: Dict[str, Any]) -> List[str]:
        """Identyfikacja czynników ryzyka"""
        risk_factors = []

        # Wiek
        age = profile.get('age', 0)
        if age > 50:
            risk_factors.append('age_related_decline')

        # Styl życia
        if profile.get('activity_level') == 'low':
            risk_factors.append('sedentary_lifestyle')
        if profile.get('stress_level') == 'high':
            risk_factors.append('chronic_stress')
        if profile.get('diet_quality') == 'poor':
            risk_factors.append('poor_nutrition')

        # Historia rodzinna
        family_history = profile.get('family_history', [])
        if 'heart_disease' in family_history:
            risk_factors.append('cardiovascular_genetic_risk')
        if 'diabetes' in family_history:
            risk_factors.append('metabolic_genetic_risk')

        return risk_factors

    def _calculate_needs_level(self, analysis: Dict[str, Any]) -> int:
        """Obliczenie poziomu potrzeb (1-10)"""
        base_score = 5  # Każdy może skorzystać z omega-3

        # Zwiększ score na podstawie czynników ryzyka
        base_score += len(analysis['risk_factors'])

        # Zwiększ na podstawie priorytetów zdrowotnych
        base_score += len(analysis['health_priorities'])

        # Dostosuj na podstawie lifestyle'u
        lifestyle_score = analysis['lifestyle_factors']['overall_score']
        if lifestyle_score < 5:
            base_score += 2  # Gorszy lifestyle = większe potrzeby

        # Ogranicz do 1-10
        return max(1, min(10, base_score))

    def _generate_contextual_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generowanie rekomendacji kontekstowych"""

        priority_areas = []
        approach = "standard"

        # Na podstawie grup wiekowych
        age_group = analysis['age_group']
        if age_group == 'senior':
            priority_areas.extend(['cognitive_support', 'joint_health', 'immune_support'])
            approach = "comprehensive"
        elif age_group == 'young_adult':
            priority_areas.extend(['foundation_building', 'energy_optimization'])
            approach = "preventive"

        # Na podstawie priorytetów zdrowotnych
        priority_areas.extend(analysis['health_priorities'])

        # Na podstawie czynników ryzyka
        if len(analysis['risk_factors']) > 3:
            approach = "intensive"
        elif len(analysis['risk_factors']) == 0:
            approach = "maintenance"

        return {
            'priority_areas': list(set(priority_areas)),
            'approach': approach
        }

    async def get_specialist_statistics(self) -> Dict[str, Any]:
        """Pobranie statystyk specjalisty"""

        return {
            'questions_answered': self.questions_answered,
            'recommendations_made': self.recommendations_made,
            'customer_consultations': self.customer_consultations,
            'knowledge_base_entries': len(self.knowledge_base),
            'product_database_entries': len(self.product_database),
            'expertise_areas': self.expertise_areas,
            'most_common_question_categories': await self._get_question_category_stats(),
            'average_response_confidence': await self._calculate_average_confidence()
        }

    async def _get_question_category_stats(self) -> Dict[str, int]:
        """Statystyki kategorii pytań"""
        # W rzeczywistej implementacji pobierałoby to dane z memory_manager
        return {
            'product_info': 45,
            'usage_instructions': 23,
            'health_benefits': 18,
            'pricing': 12,
            'scientific_research': 8
        }

    async def _calculate_average_confidence(self) -> float:
        """Obliczenie średniej pewności odpowiedzi"""
        # W rzeczywistej implementacji obliczałoby to na podstawie historycznych danych
        return 0.87