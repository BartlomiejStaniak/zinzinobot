"""
adaptive_learning.py - System adaptacyjnego uczenia się
Plik: learning_engine/adaptive_learning.py
"""

import asyncio
import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import logging
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LearningPattern:
    """Wzorzec uczenia wykryty przez system"""
    pattern_type: str  # 'content_preference', 'timing_pattern', 'audience_segment'
@dataclass
class LearningPattern:
    """Wzorzec uczenia wykryty przez system"""
    pattern_type: str  # 'content_preference', 'timing_pattern', 'audience_segment'
    pattern_data: Dict[str, Any]
    confidence: float  # 0.0 - 1.0
    impact_score: float  # Wpływ na performance
    discovered_at: str
    last_validated: str
    usage_count: int


@dataclass
class AdaptationStrategy:
    """Strategia adaptacji na podstawie wzorców"""
    strategy_name: str
    target_categories: List[str]
    adjustments: Dict[str, float]  # Jakie zmiany zastosować
    expected_improvement: float
    confidence: float
    implementation_priority: int  # 1-10


@dataclass
class LearningInsight:
    """Insight wygenerowany przez system uczenia"""
    insight_type: str
    description: str
    evidence: Dict[str, Any]
    actionable_recommendations: List[str]
    potential_impact: float
    confidence: float


class AdaptiveLearningSystem:
    """
    System adaptacyjnego uczenia się - analizuje wzorce i dostosowuje strategię
    """

    def __init__(self, db_path: str = "data/learning_materials.db"):
        self.db_path = db_path

        # Konfiguracja uczenia
        self.learning_config = {
            'pattern_detection_window': 30,  # dni dla wykrywania wzorców
            'min_samples_for_pattern': 10,  # min. próbek dla wzorca
            'adaptation_aggressiveness': 0.2,  # jak szybko adaptować (0.1-0.5)
            'confidence_threshold': 0.7,  # min. pewność dla zastosowania wzorca
            'performance_improvement_target': 0.15  # cel poprawy (15%)
        }

        # Typy wzorców do wykrywania
        self.pattern_types = {
            'content_preference': {
                'analyzer': self._analyze_content_preferences,
                'weight': 0.3
            },
            'timing_optimization': {
                'analyzer': self._analyze_timing_patterns,
                'weight': 0.2
            },
            'audience_segmentation': {
                'analyzer': self._analyze_audience_segments,
                'weight': 0.25
            },
            'zinzino_conversion_factors': {
                'analyzer': self._analyze_zinzino_conversion_patterns,
                'weight': 0.25
            }
        }

        # Metastrategije adaptacji
        self.adaptation_strategies = {
            'content_optimization': {
                'focus': 'Optymalizacja typów treści',
                'metrics': ['engagement_rate', 'conversion_rate'],
                'adjustments': ['boost_top_performers', 'retire_poor_performers']
            },
            'temporal_optimization': {
                'focus': 'Optymalizacja czasowa postów',
                'metrics': ['reach', 'engagement_rate'],
                'adjustments': ['optimal_posting_times', 'frequency_adjustment']
            },
            'zinzino_focus_optimization': {
                'focus': 'Optymalizacja dla leadów Zinzino',
                'metrics': ['conversion_rate', 'zinzino_leads'],
                'adjustments': ['increase_zinzino_content', 'optimize_call_to_action']
            }
        }

        self._init_adaptive_learning_tables()

    def _init_adaptive_learning_tables(self):
        """Inicjalizuje tabele dla adaptacyjnego uczenia"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tabela wykrytych wzorców
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                confidence REAL NOT NULL,
                impact_score REAL NOT NULL,
                discovered_at TEXT NOT NULL,
                last_validated TEXT NOT NULL,
                usage_count INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT 1
            )
        ''')

        # Tabela strategii adaptacji
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS adaptation_strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                target_categories TEXT NOT NULL,
                adjustments TEXT NOT NULL,
                expected_improvement REAL NOT NULL,
                confidence REAL NOT NULL,
                implementation_priority INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                last_applied TEXT,
                success_rate REAL DEFAULT 0.0
            )
        ''')

        # Tabela insightów
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                insight_type TEXT NOT NULL,
                description TEXT NOT NULL,
                evidence TEXT NOT NULL,
                actionable_recommendations TEXT NOT NULL,
                potential_impact REAL NOT NULL,
                confidence REAL NOT NULL,
                generated_at TEXT NOT NULL,
                applied BOOLEAN DEFAULT 0
            )
        ''')

        conn.commit()
        conn.close()

    async def update_knowledge_from_performance(self) -> List[LearningInsight]:
        """
        Główna metoda - aktualizuje wiedzę na podstawie performance
        """
        logger.info("🧠 Starting adaptive learning update...")

        insights = []

        # 1. Wykryj nowe wzorce
        new_patterns = await self._detect_learning_patterns()

        # 2. Waliduj istniejące wzorce
        await self._validate_existing_patterns()

        # 3. Generuj strategie adaptacji
        strategies = await self._generate_adaptation_strategies(new_patterns)

        # 4. Generuj actionable insights
        insights = await self._generate_learning_insights(new_patterns, strategies)

        # 5. Zastosuj najlepsze strategie automatycznie
        await self._auto_apply_high_confidence_strategies(strategies)

        logger.info(f"🎯 Adaptive learning complete: {len(insights)} insights generated")
        return insights

    async def _detect_learning_patterns(self) -> List[LearningPattern]:
        """Wykrywa nowe wzorce uczenia"""
        patterns = []

        for pattern_type, config in self.pattern_types.items():
            try:
                analyzer = config['analyzer']
                detected_patterns = await analyzer()

                for pattern_data in detected_patterns:
                    pattern = LearningPattern(
                        pattern_type=pattern_type,
                        pattern_data=pattern_data,
                        confidence=pattern_data.get('confidence', 0.5),
                        impact_score=pattern_data.get('impact_score', 0.5),
                        discovered_at=datetime.now().isoformat(),
                        last_validated=datetime.now().isoformat(),
                        usage_count=0
                    )

                    # Zapisz wzorzec jeśli ma wystarczającą pewność
                    if pattern.confidence >= self.learning_config['confidence_threshold']:
                        await self._save_learning_pattern(pattern)
                        patterns.append(pattern)

                        logger.info(f"🔍 New pattern detected: {pattern_type} (confidence: {pattern.confidence:.2f})")

            except Exception as e:
                logger.error(f"Error detecting {pattern_type} patterns: {str(e)}")

        return patterns

    async def _analyze_content_preferences(self) -> List[Dict]:
        """Analizuje preferencje treści audience"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Pobierz performance data z ostatnich 30 dni
        cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()

        cursor.execute('''
            SELECT kp.category, kp.zinzino_specific, kp.title,
                   AVG(cp.engagement_rate) as avg_engagement,
                   AVG(cp.conversion_rate) as avg_conversion,
                   COUNT(*) as sample_count,
                   AVG(cp.viral_potential) as avg_viral
            FROM content_performance cp
            JOIN knowledge_points kp ON cp.knowledge_point_id = kp.id
            WHERE cp.timestamp > ?
            GROUP BY kp.category, kp.zinzino_specific
            HAVING sample_count >= ?
            ORDER BY avg_conversion DESC
        ''', (cutoff_date, self.learning_config['min_samples_for_pattern']))

        results = cursor.fetchall()
        conn.close()

        patterns = []

        if results:
            # Znajdź top performing categories
            top_performers = results[:3]  # Top 3 categories
            poor_performers = results[-2:]  # Bottom 2 categories

            for row in top_performers:
                patterns.append({
                    'category': row[0],
                    'zinzino_specific': bool(row[1]),
                    'avg_engagement': row[3],
                    'avg_conversion': row[4],
                    'sample_count': row[5],
                    'performance_tier': 'high',
                    'confidence': min(row[5] / 20.0, 1.0),  # Higher confidence with more samples
                    'impact_score': row[4] * 2,  # Conversion rate * 2
                    'recommendation': 'increase_usage'
                })

            for row in poor_performers:
                patterns.append({
                    'category': row[0],
                    'zinzino_specific': bool(row[1]),
                    'avg_engagement': row[3],
                    'avg_conversion': row[4],
                    'sample_count': row[5],
                    'performance_tier': 'low',
                    'confidence': min(row[5] / 15.0, 1.0),
                    'impact_score': (0.1 - row[4]) * 2,  # Impact of removing poor content
                    'recommendation': 'decrease_usage_or_modify'
                })

        return patterns

    async def _analyze_timing_patterns(self) -> List[Dict]:
        """Analizuje optymalne czasy publikacji"""
        # Symulacja analizy timing patterns - w rzeczywistości analizowałbyś dane z social media

        patterns = []

        # Symulowane odkrycie: posty o wellness działają lepiej rano
        patterns.append({
            'optimal_time': '08:00-10:00',
            'category': 'mental_health',
            'avg_engagement_boost': 0.25,
            'days_of_week': ['monday', 'tuesday', 'wednesday'],
            'confidence': 0.8,
            'impact_score': 0.6,
            'recommendation': 'schedule_morning_wellness_content'
        })

        # Symulowane odkrycie: treści Zinzino działają lepiej wieczorem
        patterns.append({
            'optimal_time': '19:00-21:00',
            'category': 'zinzino_products',
            'avg_engagement_boost': 0.35,
            'days_of_week': ['thursday', 'friday', 'saturday'],
            'confidence': 0.75,
            'impact_score': 0.8,
            'recommendation': 'schedule_evening_zinzino_content'
        })

        return patterns

    async def _analyze_audience_segments(self) -> List[Dict]:
        """Analizuje segmenty audience i ich preferencje"""
        patterns = []

        # Symulowane segmenty audience - w rzeczywistości analizowałbyś dane Facebook Insights

        # Segment 1: Young professionals (25-35)
        patterns.append({
            'segment_name': 'young_professionals',
            'age_range': '25-35',
            'preferred_content': ['productivity', 'mental_health'],
            'best_conversion_content': 'stress_management',
            'avg_conversion_rate': 0.08,
            'zinzino_interest': 0.6,
            'confidence': 0.7,
            'impact_score': 0.5,
            'size_estimate': 0.4,  # 40% of audience
            'recommendation': 'create_productivity_stress_content'
        })

        # Segment 2: Health-conscious adults (35-50)
        patterns.append({
            'segment_name': 'health_conscious_adults',
            'age_range': '35-50',
            'preferred_content': ['nutrition_wellness', 'zinzino_products'],
            'best_conversion_content': 'omega_balance',
            'avg_conversion_rate': 0.12,
            'zinzino_interest': 0.85,
            'confidence': 0.85,
            'impact_score': 0.9,
            'size_estimate': 0.35,  # 35% of audience
            'recommendation': 'focus_on_zinzino_nutrition_content'
        })

        return patterns

    async def _analyze_zinzino_conversion_patterns(self) -> List[Dict]:
        """Analizuje wzorce konwersji dla Zinzino"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()

        # Pobierz dane o konwersjach Zinzino
        cursor.execute('''
            SELECT kp.category, kp.zinzino_specific,
                   AVG(cp.conversion_rate) as avg_conversion,
                   AVG(cp.zinzino_leads) as avg_leads,
                   COUNT(*) as sample_count
            FROM content_performance cp
            JOIN knowledge_points kp ON cp.knowledge_point_id = kp.id
            WHERE cp.timestamp > ? AND cp.zinzino_leads > 0
            GROUP BY kp.category, kp.zinzino_specific
            HAVING sample_count >= 3
            ORDER BY avg_conversion DESC
        ''', (cutoff_date,))

        results = cursor.fetchall()
        conn.close()

        patterns = []

        for row in results:
            # Identyfikuj high-converting patterns
            if row[2] > 0.05:  # 5% conversion rate
                patterns.append({
                    'category': row[0],
                    'zinzino_specific': bool(row[1]),
                    'avg_conversion_rate': row[2],
                    'avg_leads_per_post': row[3],
                    'sample_count': row[4],
                    'pattern_type': 'high_converter',
                    'confidence': min(row[4] / 10.0, 1.0),
                    'impact_score': row[2] * 5,  # High impact for conversions
                    'recommendation': 'prioritize_this_content_type'
                })

        # Dodaj specjalne wzorce dla Zinzino
        patterns.append({
            'content_elements': ['balance_test', 'omega_ratio', 'scientific_evidence'],
            'avg_conversion_boost': 0.4,
            'pattern_type': 'zinzino_specific_elements',
            'confidence': 0.8,
            'impact_score': 0.8,
            'recommendation': 'include_balance_test_mentions'
        })

        return patterns

    async def _validate_existing_patterns(self):
        """Waliduje istniejące wzorce"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Pobierz aktywne wzorce
        cursor.execute('''
            SELECT id, pattern_type, pattern_data, confidence, last_validated
            FROM learning_patterns
            WHERE is_active = 1
        ''')

        patterns = cursor.fetchall()

        for pattern_row in patterns:
            pattern_id, pattern_type, pattern_data_json, old_confidence, last_validated = pattern_row

            # Sprawdź czy wzorzec nadal jest ważny (walidacja co 7 dni)
            last_validated_date = datetime.fromisoformat(last_validated)
            if (datetime.now() - last_validated_date).days >= 7:

                # Re-analizuj wzorzec
                pattern_data = json.loads(pattern_data_json)
                new_confidence = await self._revalidate_pattern(pattern_type, pattern_data)

                # Aktualizuj confidence
                cursor.execute('''
                    UPDATE learning_patterns
                    SET confidence = ?, last_validated = ?, usage_count = usage_count + 1
                    WHERE id = ?
                ''', (new_confidence, datetime.now().isoformat(), pattern_id))

                # Deaktywuj jeśli confidence spadła zbyt nisko
                if new_confidence < self.learning_config['confidence_threshold'] * 0.7:
                    cursor.execute('''
                        UPDATE learning_patterns SET is_active = 0 WHERE id = ?
                    ''', (pattern_id,))

                    logger.info(f"🔄 Deactivated pattern {pattern_id} due to low confidence: {new_confidence:.2f}")

        conn.commit()
        conn.close()

    async def _revalidate_pattern(self, pattern_type: str, pattern_data: Dict) -> float:
        """Re-waliduje wzorzec na podstawie recent performance"""
        # Uproszczona re-walidacja - sprawdź czy wzorzec nadal działa

        if pattern_type == 'content_preference':
            category = pattern_data.get('category')
            if category:
                # Sprawdź recent performance tej kategorii
                recent_performance = await self._get_recent_category_performance(category)
                if recent_performance and recent_performance > 0.05:
                    return min(pattern_data.get('confidence', 0.5) * 1.1, 1.0)
                else:
                    return pattern_data.get('confidence', 0.5) * 0.8

        # Default: slight confidence decay
        return pattern_data.get('confidence', 0.5) * 0.95

    async def _get_recent_category_performance(self, category: str) -> Optional[float]:
        """Pobiera recent performance dla kategorii"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_date = (datetime.now() - timedelta(days=7)).isoformat()

        cursor.execute('''
            SELECT AVG(cp.conversion_rate)
            FROM content_performance cp
            JOIN knowledge_points kp ON cp.knowledge_point_id = kp.id
            WHERE kp.category = ? AND cp.timestamp > ?
        ''', (category, cutoff_date))

        result = cursor.fetchone()
        conn.close()

        return result[0] if result and result[0] else None

    async def _generate_adaptation_strategies(self, patterns: List[LearningPattern]) -> List[AdaptationStrategy]:
        """Generuje strategie adaptacji na podstawie wzorców"""
        strategies = []

        for pattern in patterns:
            if pattern.pattern_type == 'content_preference':
                strategy = await self._create_content_adaptation_strategy(pattern)
                if strategy:
                    strategies.append(strategy)

            elif pattern.pattern_type == 'timing_optimization':
                strategy = await self._create_timing_adaptation_strategy(pattern)
                if strategy:
                    strategies.append(strategy)

            elif pattern.pattern_type == 'zinzino_conversion_factors':
                strategy = await self._create_zinzino_adaptation_strategy(pattern)
                if strategy:
                    strategies.append(strategy)

        # Zapisz strategie do bazy danych
        for strategy in strategies:
            await self._save_adaptation_strategy(strategy)

        return strategies

    async def _create_content_adaptation_strategy(self, pattern: LearningPattern) -> Optional[AdaptationStrategy]:
        """Tworzy strategię adaptacji treści"""
        pattern_data = pattern.pattern_data

        if pattern_data.get('performance_tier') == 'high':
            return AdaptationStrategy(
                strategy_name=f"boost_{pattern_data['category']}_content",
                target_categories=[pattern_data['category']],
                adjustments={
                    'content_frequency_multiplier': 1.5,
                    'priority_boost': 0.3,
                    'zinzino_focus': 1.2 if pattern_data.get('zinzino_specific') else 1.0
                },
                expected_improvement=pattern_data.get('impact_score', 0.5) * 0.3,
                confidence=pattern.confidence,
                implementation_priority=8
            )

        elif pattern_data.get('performance_tier') == 'low':
            return AdaptationStrategy(
                strategy_name=f"reduce_{pattern_data['category']}_content",
                target_categories=[pattern_data['category']],
                adjustments={
                    'content_frequency_multiplier': 0.6,
                    'priority_reduction': -0.2,
                    'modification_needed': True
                },
                expected_improvement=pattern_data.get('impact_score', 0.3) * 0.2,
                confidence=pattern.confidence,
                implementation_priority=6
            )

        return None

    async def _create_timing_adaptation_strategy(self, pattern: LearningPattern) -> Optional[AdaptationStrategy]:
        """Tworzy strategię adaptacji timing"""
        pattern_data = pattern.pattern_data

        return AdaptationStrategy(
            strategy_name=f"optimize_timing_{pattern_data.get('category', 'general')}",
            target_categories=[pattern_data.get('category', 'all')],
            adjustments={
                'optimal_posting_time': pattern_data.get('optimal_time'),
                'engagement_boost_expected': pattern_data.get('avg_engagement_boost', 0.2),
                'days_of_week': pattern_data.get('days_of_week', [])
            },
            expected_improvement=pattern_data.get('avg_engagement_boost', 0.2),
            confidence=pattern.confidence,
            implementation_priority=7
        )

    async def _create_zinzino_adaptation_strategy(self, pattern: LearningPattern) -> Optional[AdaptationStrategy]:
        """Tworzy strategię adaptacji dla Zinzino"""
        pattern_data = pattern.pattern_data

        if pattern_data.get('pattern_type') == 'high_converter':
            return AdaptationStrategy(
                strategy_name=f"prioritize_zinzino_converters",
                target_categories=[pattern_data['category']],
                adjustments={
                    'zinzino_content_boost': 2.0,
                    'conversion_optimization': True,
                    'lead_generation_focus': True
                },
                expected_improvement=pattern_data.get('avg_conversion_rate', 0.05) * 0.5,
                confidence=pattern.confidence,
                implementation_priority=9  # High priority for conversions
            )

        return None

    async def _save_learning_pattern(self, pattern: LearningPattern):
        """Zapisuje wzorzec uczenia do bazy danych"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO learning_patterns
            (pattern_type, pattern_data, confidence, impact_score, 
             discovered_at, last_validated, usage_count, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern.pattern_type,
            json.dumps(pattern.pattern_data),
            pattern.confidence,
            pattern.impact_score,
            pattern.discovered_at,
            pattern.last_validated,
            pattern.usage_count,
            1  # is_active
        ))

        conn.commit()
        conn.close()

    async def _save_adaptation_strategy(self, strategy: AdaptationStrategy):
        """Zapisuje strategię adaptacji"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO adaptation_strategies
            (strategy_name, target_categories, adjustments, expected_improvement,
             confidence, implementation_priority, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            strategy.strategy_name,
            json.dumps(strategy.target_categories),
            json.dumps(strategy.adjustments),
            strategy.expected_improvement,
            strategy.confidence,
            strategy.implementation_priority,
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

    async def _generate_learning_insights(self, patterns: List[LearningPattern],
                                          strategies: List[AdaptationStrategy]) -> List[LearningInsight]:
        """Generuje actionable insights"""
        insights = []

        # Insight 1: Top performing content types
        if patterns:
            top_pattern = max(patterns, key=lambda p: p.impact_score)

            insight = LearningInsight(
                insight_type="content_optimization",
                description=f"Content type '{top_pattern.pattern_data.get('category', 'unknown')}' shows highest performance",
                evidence={
                    'avg_conversion': top_pattern.pattern_data.get('avg_conversion', 0),
                    'impact_score': top_pattern.impact_score,
                    'confidence': top_pattern.confidence
                },
                actionable_recommendations=[
                    f"Increase production of {top_pattern.pattern_data.get('category')} content by 50%",
                    "Analyze successful posts in this category for common elements",
                    "Create templates based on high-performing content structure"
                ],
                potential_impact=top_pattern.impact_score,
                confidence=top_pattern.confidence
            )
            insights.append(insight)

        # Insight 2: Zinzino optimization opportunities
        zinzino_patterns = [p for p in patterns if 'zinzino' in str(p.pattern_data).lower()]
        if zinzino_patterns:
            insight = LearningInsight(
                insight_type="zinzino_optimization",
                description="Specific Zinzino content elements drive higher conversions",
                evidence={'patterns_found': len(zinzino_patterns)},
                actionable_recommendations=[
                    "Include BalanceTest mentions in 80% of health-related posts",
                    "Focus on omega-3 ratio education content",
                    "Create before/after transformation stories",
                    "Emphasize scientific backing and test results"
                ],
                potential_impact=0.6,
                confidence=0.8
            )
            insights.append(insight)

        # Insight 3: Timing optimization
        timing_strategies = [s for s in strategies if 'timing' in s.strategy_name]
        if timing_strategies:
            insight = LearningInsight(
                insight_type="timing_optimization",
                description="Optimal posting times identified for different content types",
                evidence={'strategies_count': len(timing_strategies)},
                actionable_recommendations=[
                    "Schedule wellness content for 8-10 AM on weekdays",
                    "Post Zinzino content during 7-9 PM on Thu-Sat",
                    "Implement automated scheduling based on content type"
                ],
                potential_impact=0.4,
                confidence=0.75
            )
            insights.append(insight)

        # Zapisz insights do bazy danych
        for insight in insights:
            await self._save_learning_insight(insight)

        return insights

    async def _save_learning_insight(self, insight: LearningInsight):
        """Zapisuje insight do bazy danych"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO learning_insights
            (insight_type, description, evidence, actionable_recommendations,
             potential_impact, confidence, generated_at, applied)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            insight.insight_type,
            insight.description,
            json.dumps(insight.evidence),
            json.dumps(insight.actionable_recommendations),
            insight.potential_impact,
            insight.confidence,
            datetime.now().isoformat(),
            0  # not applied yet
        ))

        conn.commit()
        conn.close()

    async def _auto_apply_high_confidence_strategies(self, strategies: List[AdaptationStrategy]):
        """Automatycznie zastosowuje strategie o wysokiej pewności"""
        auto_apply_threshold = 0.8

        for strategy in strategies:
            if (strategy.confidence >= auto_apply_threshold and
                    strategy.implementation_priority >= 8):
                logger.info(f"🤖 Auto-applying high-confidence strategy: {strategy.strategy_name}")

                # Tutaj implementowałbyś faktyczne zastosowanie strategii
                # np. aktualizacja wag w systemie generowania contentu
                await self._apply_strategy_adjustments(strategy)

                # Oznacz jako zastosowaną
                await self._mark_strategy_as_applied(strategy)

    async def _apply_strategy_adjustments(self, strategy: AdaptationStrategy):
        """Stosuje adjustments strategii do systemu"""
        # Przykład zastosowania - aktualizacja priorytetów kategorii

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        adjustments = strategy.adjustments

        if 'content_frequency_multiplier' in adjustments:
            multiplier = adjustments['content_frequency_multiplier']

            for category in strategy.target_categories:
                # Aktualizuj priority w learning_objectives
                cursor.execute('''
                    UPDATE learning_objectives
                    SET priority = priority * ?,
                        last_updated = ?
                    WHERE category = ?
                ''', (multiplier, datetime.now().isoformat(), category))

        if 'priority_boost' in adjustments:
            boost = adjustments['priority_boost']

            for category in strategy.target_categories:
                cursor.execute('''
                    UPDATE processed_materials
                    SET performance_score = performance_score + ?
                    WHERE id IN (
                        SELECT pm.id FROM processed_materials pm
                        JOIN knowledge_points kp ON kp.material_id = pm.id
                        WHERE kp.category = ?
                    )
                ''', (boost, category))

        conn.commit()
        conn.close()

        logger.info(f"✅ Applied adjustments for {strategy.strategy_name}")

    async def _mark_strategy_as_applied(self, strategy: AdaptationStrategy):
        """Oznacza strategię jako zastosowaną"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE adaptation_strategies
            SET last_applied = ?
            WHERE strategy_name = ?
        ''', (datetime.now().isoformat(), strategy.strategy_name))

        conn.commit()
        conn.close()

    async def integrate_new_knowledge(self, knowledge_points: List[Dict], category: str):
        """Integruje nową wiedzę do systemu adaptacyjnego uczenia"""
        logger.info(f"🔗 Integrating {len(knowledge_points)} new knowledge points for {category}")

        # Analizuj nową wiedzę pod kątem istniejących wzorców
        await self._analyze_new_knowledge_against_patterns(knowledge_points, category)

        # Aktualizuj kategorie performance targets jeśli potrzeba
        await self._update_category_targets_if_needed(category, len(knowledge_points))

    async def _analyze_new_knowledge_against_patterns(self, knowledge_points: List[Dict], category: str):
        """Analizuje nową wiedzę względem istniejących wzorców"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Sprawdź czy kategoria ma aktywne wzorce
        cursor.execute('''
            SELECT pattern_type, pattern_data, confidence
            FROM learning_patterns
            WHERE is_active = 1 AND pattern_data LIKE ?
        ''', (f'%{category}%',))

        patterns = cursor.fetchall()
        conn.close()

        for pattern_row in patterns:
            pattern_type, pattern_data_json, confidence = pattern_row
            pattern_data = json.loads(pattern_data_json)

            # Jeśli dodajemy wiedzę do high-performing category
            if (pattern_data.get('category') == category and
                    pattern_data.get('performance_tier') == 'high'):
                logger.info(f"📈 Adding knowledge to high-performing category: {category}")

                # Zwiększ confidence wzorca (więcej danych = większa pewność)
                new_confidence = min(confidence * 1.05, 1.0)

                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE learning_patterns
                    SET confidence = ?, last_validated = ?
                    WHERE pattern_type = ? AND pattern_data = ?
                ''', (new_confidence, datetime.now().isoformat(), pattern_type, pattern_data_json))
                conn.commit()
                conn.close()

    async def _update_category_targets_if_needed(self, category: str, new_knowledge_count: int):
        """Aktualizuje cele kategorii jeśli potrzeba"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Sprawdź obecne cele
        cursor.execute('''
            SELECT current_knowledge_points, target_knowledge_points
            FROM learning_objectives
            WHERE category = ?
        ''', (category,))

        result = cursor.fetchone()

        if result:
            current, target = result
            new_total = current + new_knowledge_count

            # Jeśli przekroczyliśmy cel, zwiększ go
            if new_total > target:
                new_target = int(target * 1.2)  # Zwiększ o 20%

                cursor.execute('''
                    UPDATE learning_objectives
                    SET target_knowledge_points = ?, last_updated = ?
                    WHERE category = ?
                ''', (new_target, datetime.now().isoformat(), category))

                logger.info(f"🎯 Updated target for {category}: {target} → {new_target}")

        conn.commit()
        conn.close()

    async def get_adaptive_learning_status(self) -> Dict:
        """Zwraca status systemu adaptacyjnego uczenia"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Policz aktywne wzorce
        cursor.execute('''
            SELECT pattern_type, COUNT(*), AVG(confidence), AVG(impact_score)
            FROM learning_patterns
            WHERE is_active = 1
            GROUP BY pattern_type
        ''')
        patterns_stats = cursor.fetchall()

        # Policz strategie
        cursor.execute('''
            SELECT COUNT(*) as total_strategies,
                   COUNT(CASE WHEN last_applied IS NOT NULL THEN 1 END) as applied_strategies,
                   AVG(confidence) as avg_confidence,
                   AVG(expected_improvement) as avg_expected_improvement
            FROM adaptation_strategies
        ''')
        strategies_stats = cursor.fetchone()

        # Policz insights
        cursor.execute('''
            SELECT COUNT(*) as total_insights,
                   COUNT(CASE WHEN applied = 1 THEN 1 END) as applied_insights,
                   AVG(potential_impact) as avg_impact
            FROM learning_insights
        ''')
        insights_stats = cursor.fetchone()

        conn.close()

        return {
            'patterns': {
                'by_type': [
                    {
                        'type': row[0],
                        'count': row[1],
                        'avg_confidence': row[2],
                        'avg_impact': row[3]
                    } for row in patterns_stats
                ],
                'total_active': sum(row[1] for row in patterns_stats)
            },
            'strategies': {
                'total': strategies_stats[0],
                'applied': strategies_stats[1],
                'avg_confidence': strategies_stats[2] or 0,
                'avg_expected_improvement': strategies_stats[3] or 0
            },
            'insights': {
                'total': insights_stats[0],
                'applied': insights_stats[1],
                'avg_potential_impact': insights_stats[2] or 0
            },
            'system_learning_rate': self.learning_config['adaptation_aggressiveness'],
            'last_update': datetime.now().isoformat()
        }

    async def get_recommendations_for_content_generation(self, category: str) -> Dict:
        """Zwraca rekomendacje dla generowania contentu w danej kategorii"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Pobierz aktywne wzorce dla kategorii
        cursor.execute('''
            SELECT pattern_type, pattern_data, confidence, impact_score
            FROM learning_patterns
            WHERE is_active = 1 AND pattern_data LIKE ?
            ORDER BY impact_score DESC, confidence DESC
        ''', (f'%{category}%',))

        patterns = cursor.fetchall()

        # Pobierz najnowsze insights
        cursor.execute('''
            SELECT insight_type, actionable_recommendations, confidence
            FROM learning_insights
            WHERE applied = 0
            ORDER BY potential_impact DESC, generated_at DESC
            LIMIT 5
        ''')

        insights = cursor.fetchall()
        conn.close()

        recommendations = {
            'category': category,
            'priority_adjustments': {},
            'content_elements_to_include': [],
            'timing_recommendations': {},
            'audience_targeting': {},
            'zinzino_optimization': {}
        }

        # Przetwórz wzorce
        for pattern_row in patterns:
            pattern_type, pattern_data_json, confidence, impact_score = pattern_row
            pattern_data = json.loads(pattern_data_json)

            if pattern_type == 'content_preference':
                if pattern_data.get('performance_tier') == 'high':
                    recommendations['priority_adjustments']['boost_multiplier'] = 1.5
                elif pattern_data.get('performance_tier') == 'low':
                    recommendations['priority_adjustments']['reduction_multiplier'] = 0.7

            elif pattern_type == 'timing_optimization':
                recommendations['timing_recommendations'] = {
                    'optimal_time': pattern_data.get('optimal_time'),
                    'best_days': pattern_data.get('days_of_week', []),
                    'expected_boost': pattern_data.get('avg_engagement_boost', 0)
                }

            elif pattern_type == 'zinzino_conversion_factors':
                if 'content_elements' in pattern_data:
                    recommendations['zinzino_optimization']['key_elements'] = pattern_data['content_elements']

                if pattern_data.get('pattern_type') == 'high_converter':
                    recommendations['zinzino_optimization']['conversion_focus'] = True
                    recommendations['zinzino_optimization']['lead_generation_priority'] = True

        # Przetwórz insights
        actionable_items = []
        for insight_row in insights:
            insight_type, recommendations_json, confidence = insight_row
            insight_recommendations = json.loads(recommendations_json)

            for rec in insight_recommendations:
                if category.lower() in rec.lower() or 'zinzino' in rec.lower():
                    actionable_items.append({
                        'recommendation': rec,
                        'confidence': confidence,
                        'source': insight_type
                    })

        recommendations['actionable_items'] = actionable_items

        return recommendations


# Test funkcji
async def test_adaptive_learning():
    """Test systemu adaptacyjnego uczenia"""

    learning_system = AdaptiveLearningSystem()

    print("🧠 Testing Adaptive Learning System...")
    print("=" * 50)

    # Test 1: Uruchom główny cykl uczenia
    print("🔄 Running adaptive learning cycle...")
    insights = await learning_system.update_knowledge_from_performance()

    print(f"✅ Generated {len(insights)} learning insights")

    for insight in insights:
        print(f"\n📋 Insight: {insight.insight_type}")
        print(f"   Description: {insight.description}")
        print(f"   Confidence: {insight.confidence:.2f}")
        print(f"   Potential impact: {insight.potential_impact:.2f}")
        print(f"   Recommendations: {len(insight.actionable_recommendations)}")

    # Test 2: Sprawdź status systemu
    print(f"\n📊 Adaptive Learning Status:")
    status = await learning_system.get_adaptive_learning_status()

    print(f"   Active patterns: {status['patterns']['total_active']}")
    print(f"   Total strategies: {status['strategies']['total']}")
    print(f"   Applied strategies: {status['strategies']['applied']}")
    print(f"   Total insights: {status['insights']['total']}")
    print(f"   Applied insights: {status['insights']['applied']}")

    # Test 3: Pobierz rekomendacje dla contentu
    print(f"\n🎯 Content Generation Recommendations for 'mental_health':")
    recommendations = await learning_system.get_recommendations_for_content_generation('mental_health')

    print(f"   Priority adjustments: {recommendations['priority_adjustments']}")
    print(f"   Timing recommendations: {recommendations['timing_recommendations']}")
    print(f"   Actionable items: {len(recommendations['actionable_items'])}")

    for item in recommendations['actionable_items'][:3]:
        print(f"   - {item['recommendation']} (confidence: {item['confidence']:.2f})")


if __name__ == "__main__":
    asyncio.run(test_adaptive_learning())  # !/usr/bin/env python3
