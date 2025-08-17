#!/usr/bin/.env python3
"""
knowledge_updater.py - Aktualizacja bazy wiedzy na podstawie performance
Plik: learning_engine/knowledge_updater.py
"""

import asyncio
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import logging
import statistics
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Metryki performance dla treści"""
    knowledge_point_id: int
    engagement_rate: float  # Likes, shares, comments na FB
    click_through_rate: float  # CTR do strony/Zinzino
    conversion_rate: float  # Rejestracje w Zinzino
    reach: int  # Ile osób widziało
    zinzino_leads_generated: int
    content_quality_score: float  # Ocena jakości treści
    viral_potential: float  # Czy treść ma potencjał viralowy
    timestamp: str
    platform: str  # facebook, instagram, etc.


@dataclass
class KnowledgeUpdate:
    """Aktualizacja punktu wiedzy"""
    knowledge_point_id: int
    old_score: float
    new_score: float
    update_reason: str
    performance_trend: str  # 'improving', 'declining', 'stable'
    recommended_action: str  # 'boost', 'modify', 'retire', 'maintain'


class KnowledgeUpdater:
    """
    Klasa odpowiedzialna za aktualizację bazy wiedzy na podstawie performance
    """

    def __init__(self, db_path: str = "data/learning_materials.db"):
        self.db_path = db_path

        # Konfiguracja aktualizacji
        self.update_config = {
            'performance_window_days': 30,  # Okno analizy performance
            'min_data_points': 5,  # Min. punktów danych dla aktualizacji
            'score_adjustment_rate': 0.1,  # Jak szybko aktualizować scores
            'viral_threshold': 0.05,  # Próg content viralowy (5% reach)
            'low_performance_threshold': 0.3,  # Próg niskiej wydajności
            'high_performance_threshold': 0.8  # Próg wysokiej wydajności
        }

        # Wagi dla różnych metryk
        self.metric_weights = {
            'engagement_rate': 0.25,  # Interakcje z postem
            'click_through_rate': 0.20,  # Clicks do strony
            'conversion_rate': 0.30,  # Rejestracje Zinzino (najważniejsze!)
            'reach': 0.10,  # Zasięg
            'viral_potential': 0.15  # Potencjał viralowy
        }

        # Kategorie treści i ich cele performance
        self.category_targets = {
            'zinzino_products': {
                'target_conversion_rate': 0.08,  # 8% conversion dla Zinzino
                'target_engagement': 0.06,
                'priority_multiplier': 1.5
            },
            'mental_health': {
                'target_conversion_rate': 0.05,  # 5% conversion
                'target_engagement': 0.08,  # Wyższe zaangażowanie
                'priority_multiplier': 1.2
            },
            'productivity': {
                'target_conversion_rate': 0.04,
                'target_engagement': 0.07,
                'priority_multiplier': 1.0
            },
            'nutrition_wellness': {
                'target_conversion_rate': 0.06,
                'target_engagement': 0.06,
                'priority_multiplier': 1.1
            }
        }

        self._init_performance_tracking()

    def _init_performance_tracking(self):
        """Inicjalizuje tabele do trackingu performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tabela już istnieje w material_processor, ale sprawdźmy strukturę
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                knowledge_point_id INTEGER REFERENCES knowledge_points(id),
                content_generated TEXT NOT NULL,
                engagement_rate REAL DEFAULT 0.0,
                click_through_rate REAL DEFAULT 0.0,
                conversion_rate REAL DEFAULT 0.0,
                reach INTEGER DEFAULT 0,
                zinzino_leads INTEGER DEFAULT 0,
                content_quality_score REAL DEFAULT 0.5,
                viral_potential REAL DEFAULT 0.0,
                timestamp TEXT NOT NULL,
                platform TEXT DEFAULT 'facebook'
            )
        ''')

        # Tabela historii aktualizacji
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_updates_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                knowledge_point_id INTEGER REFERENCES knowledge_points(id),
                old_performance_score REAL NOT NULL,
                new_performance_score REAL NOT NULL,
                update_reason TEXT NOT NULL,
                performance_trend TEXT NOT NULL,
                recommended_action TEXT NOT NULL,
                metrics_summary TEXT NOT NULL,
                update_timestamp TEXT NOT NULL
            )
        ''')

        conn.commit()
        conn.close()

    async def record_content_performance(self, metrics: PerformanceMetrics):
        """Zapisuje metryki performance dla treści"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO content_performance 
            (knowledge_point_id, content_generated, engagement_rate, click_through_rate,
             conversion_rate, reach, zinzino_leads, content_quality_score, 
             viral_potential, timestamp, platform)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.knowledge_point_id,
            f"Content based on knowledge point {metrics.knowledge_point_id}",
            metrics.engagement_rate,
            metrics.click_through_rate,
            metrics.conversion_rate,
            metrics.reach,
            metrics.zinzino_leads_generated,
            metrics.content_quality_score,
            metrics.viral_potential,
            metrics.timestamp,
            metrics.platform
        ))

        conn.commit()
        conn.close()

        logger.info(f"📊 Recorded performance for knowledge point {metrics.knowledge_point_id}")

    async def update_knowledge_scores(self) -> List[KnowledgeUpdate]:
        """
        Główna metoda - aktualizuje scores na podstawie recent performance
        """
        logger.info("🔄 Starting knowledge score updates based on performance...")

        updates = []

        # Pobierz punkty wiedzy z wystarczającą ilością danych
        knowledge_points = await self._get_knowledge_points_for_update()

        for kp_id, kp_data in knowledge_points.items():
            # Pobierz metryki performance
            performance_data = await self._get_performance_data(kp_id)

            if len(performance_data) >= self.update_config['min_data_points']:
                # Oblicz nowy score
                update = await self._calculate_score_update(kp_id, kp_data, performance_data)

                if update:
                    # Zastosuj aktualizację
                    await self._apply_knowledge_update(update)
                    updates.append(update)

                    logger.info(f"✅ Updated knowledge point {kp_id}: {update.old_score:.3f} → {update.new_score:.3f}")

        # Identyfikuj trendy i rekomendacje
        await self._analyze_trends_and_recommendations(updates)

        logger.info(f"🎯 Updated {len(updates)} knowledge points")
        return updates

    async def _get_knowledge_points_for_update(self) -> Dict[int, Dict]:
        """Pobiera punkty wiedzy gotowe do aktualizacji"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Pobierz punkty wiedzy które mają recent performance data
        cutoff_date = (datetime.now() - timedelta(days=self.update_config['performance_window_days'])).isoformat()

        cursor.execute('''
            SELECT DISTINCT kp.id, kp.category, kp.title, kp.confidence_score, 
                   pm.performance_score, kp.zinzino_specific
            FROM knowledge_points kp
            JOIN processed_materials pm ON kp.material_id = pm.id
            WHERE EXISTS (
                SELECT 1 FROM content_performance cp 
                WHERE cp.knowledge_point_id = kp.id 
                AND cp.timestamp > ?
            )
            AND pm.is_active = 1
        ''', (cutoff_date,))

        results = cursor.fetchall()
        conn.close()

        knowledge_points = {}
        for row in results:
            knowledge_points[row[0]] = {
                'category': row[1],
                'title': row[2],
                'confidence_score': row[3],
                'current_performance_score': row[4],
                'zinzino_specific': bool(row[5])
            }

        return knowledge_points

    async def _get_performance_data(self, knowledge_point_id: int) -> List[Dict]:
        """Pobiera dane performance dla punktu wiedzy"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_date = (datetime.now() - timedelta(days=self.update_config['performance_window_days'])).isoformat()

        cursor.execute('''
            SELECT engagement_rate, click_through_rate, conversion_rate, reach,
                   zinzino_leads, content_quality_score, viral_potential, timestamp, platform
            FROM content_performance
            WHERE knowledge_point_id = ? AND timestamp > ?
            ORDER BY timestamp DESC
        ''', (knowledge_point_id, cutoff_date))

        results = cursor.fetchall()
        conn.close()

        performance_data = []
        for row in results:
            performance_data.append({
                'engagement_rate': row[0],
                'click_through_rate': row[1],
                'conversion_rate': row[2],
                'reach': row[3],
                'zinzino_leads': row[4],
                'content_quality_score': row[5],
                'viral_potential': row[6],
                'timestamp': row[7],
                'platform': row[8]
            })

        return performance_data

    async def _calculate_score_update(self, kp_id: int, kp_data: Dict,
                                      performance_data: List[Dict]) -> Optional[KnowledgeUpdate]:
        """Oblicza nowy score na podstawie performance"""

        # Oblicz średnie metryki
        avg_metrics = self._calculate_average_metrics(performance_data)

        # Pobierz target values dla kategorii
        category = kp_data['category']
        targets = self.category_targets.get(category, self.category_targets['nutrition_wellness'])

        # Oblicz composite performance score
        composite_score = self._calculate_composite_score(avg_metrics, targets)

        # Uwzględnij specjalne przypadki
        if kp_data['zinzino_specific']:
            composite_score *= 1.2  # Bonus dla Zinzino-specific content

        # Określ trend
        trend = self._determine_performance_trend(performance_data)

        # Oblicz nowy score z uwzględnieniem learning rate
        current_score = kp_data['current_performance_score']
        score_adjustment = (composite_score - current_score) * self.update_config['score_adjustment_rate']
        new_score = max(0.0, min(1.0, current_score + score_adjustment))

        # Sprawdź czy zmiana jest znacząca
        if abs(new_score - current_score) < 0.02:  # Mniej niż 2% change
            return None

        # Określ rekomendowaną akcję
        recommended_action = self._determine_recommended_action(composite_score, trend)

        # Przygotuj update reason
        update_reason = self._generate_update_reason(avg_metrics, targets, trend)

        return KnowledgeUpdate(
            knowledge_point_id=kp_id,
            old_score=current_score,
            new_score=new_score,
            update_reason=update_reason,
            performance_trend=trend,
            recommended_action=recommended_action
        )

    def _calculate_average_metrics(self, performance_data: List[Dict]) -> Dict[str, float]:
        """Oblicza średnie metryki z danych performance"""
        if not performance_data:
            return {}

        # Oblicz średnie z wagami czasowymi (nowsze dane ważniejsze)
        total_weight = 0
        weighted_sums = defaultdict(float)

        for i, data in enumerate(performance_data):
            # Waga maleje dla starszych danych
            weight = 1.0 / (i + 1)
            total_weight += weight

            for metric in ['engagement_rate', 'click_through_rate', 'conversion_rate', 'viral_potential']:
                weighted_sums[metric] += data[metric] * weight

        # Oblicz średnie ważone
        avg_metrics = {}
        for metric, weighted_sum in weighted_sums.items():
            avg_metrics[metric] = weighted_sum / total_weight if total_weight > 0 else 0

        # Dodaj specjalne metryki
        avg_metrics['total_reach'] = sum(d['reach'] for d in performance_data)
        avg_metrics['total_zinzino_leads'] = sum(d['zinzino_leads'] for d in performance_data)
        avg_metrics['avg_quality'] = statistics.mean(d['content_quality_score'] for d in performance_data)

        return avg_metrics

    def _calculate_composite_score(self, metrics: Dict[str, float], targets: Dict[str, float]) -> float:
        """Oblicza composite performance score"""
        score = 0.0

        # Performance względem targets
        if 'conversion_rate' in metrics:
            conversion_performance = min(metrics['conversion_rate'] / targets['target_conversion_rate'], 2.0)
            score += conversion_performance * self.metric_weights['conversion_rate']

        if 'engagement_rate' in metrics:
            engagement_performance = min(metrics['engagement_rate'] / targets['target_engagement'], 2.0)
            score += engagement_performance * self.metric_weights['engagement_rate']

        # Inne metryki
        score += metrics.get('click_through_rate', 0) * self.metric_weights['click_through_rate'] * 10  # Scale up CTR
        score += min(metrics.get('viral_potential', 0) / self.update_config['viral_threshold'], 1.0) * \
                 self.metric_weights['viral_potential']

        # Bonus za reach (scaled)
        reach_score = min(metrics.get('total_reach', 0) / 1000, 1.0)  # Max bonus at 1000 reach
        score += reach_score * self.metric_weights['reach']

        # Uwzględnij priority multiplier
        score *= targets.get('priority_multiplier', 1.0)

        return min(score, 1.0)

    def _determine_performance_trend(self, performance_data: List[Dict]) -> str:
        """Określa trend performance (improving, declining, stable)"""
        if len(performance_data) < 3:
            return 'stable'

        # Porównaj pierwszą i drugą połowę danych
        mid_point = len(performance_data) // 2
        recent_data = performance_data[:mid_point]  # Nowsze dane (DESC order)
        older_data = performance_data[mid_point:]

        recent_avg = statistics.mean(d['conversion_rate'] for d in recent_data)
        older_avg = statistics.mean(d['conversion_rate'] for d in older_data)

        change_ratio = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0

        if change_ratio > 0.1:  # 10% improvement
            return 'improving'
        elif change_ratio < -0.1:  # 10% decline
            return 'declining'
        else:
            return 'stable'

    def _determine_recommended_action(self, composite_score: float, trend: str) -> str:
        """Określa rekomendowaną akcję dla punktu wiedzy"""

        if composite_score > self.update_config['high_performance_threshold']:
            if trend == 'improving':
                return 'boost'  # Zwiększ użycie tego typu contentu
            else:
                return 'maintain'  # Utrzymaj obecny poziom

        elif composite_score < self.update_config['low_performance_threshold']:
            if trend == 'declining':
                return 'retire'  # Przestań używać
            else:
                return 'modify'  # Spróbuj zmodyfikować approach

        else:  # Medium performance
            if trend == 'improving':
                return 'boost'
            elif trend == 'declining':
                return 'modify'
            else:
                return 'maintain'

    def _generate_update_reason(self, metrics: Dict[str, float], targets: Dict[str, float], trend: str) -> str:
        """Generuje opis powodu aktualizacji"""
        reasons = []

        # Sprawdź conversion rate
        if 'conversion_rate' in metrics:
            conv_rate = metrics['conversion_rate']
            target_conv = targets['target_conversion_rate']

            if conv_rate > target_conv * 1.2:
                reasons.append(f"Excellent conversion rate: {conv_rate:.1%} (target: {target_conv:.1%})")
            elif conv_rate < target_conv * 0.8:
                reasons.append(f"Low conversion rate: {conv_rate:.1%} (target: {target_conv:.1%})")

        # Sprawdź engagement
        if 'engagement_rate' in metrics:
            eng_rate = metrics['engagement_rate']
            target_eng = targets['target_engagement']

            if eng_rate > target_eng * 1.3:
                reasons.append(f"High engagement: {eng_rate:.1%}")
            elif eng_rate < target_eng * 0.7:
                reasons.append(f"Low engagement: {eng_rate:.1%}")

        # Sprawdź viral potential
        if metrics.get('viral_potential', 0) > self.update_config['viral_threshold']:
            reasons.append(f"Viral content detected: {metrics['viral_potential']:.1%} viral rate")

        # Dodaj trend
        reasons.append(f"Performance trend: {trend}")

        return "; ".join(reasons) if reasons else "Regular performance update"

    async def _apply_knowledge_update(self, update: KnowledgeUpdate):
        """Zastosowuje aktualizację do bazy danych"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Aktualizuj performance_score w processed_materials
        cursor.execute('''
            UPDATE processed_materials 
            SET performance_score = ?, last_updated = ?
            WHERE id = (
                SELECT material_id FROM knowledge_points 
                WHERE id = ?
            )
        ''', (update.new_score, datetime.now().isoformat(), update.knowledge_point_id))

        # Zapisz historię aktualizacji
        cursor.execute('''
            INSERT INTO knowledge_updates_history
            (knowledge_point_id, old_performance_score, new_performance_score,
             update_reason, performance_trend, recommended_action, 
             metrics_summary, update_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            update.knowledge_point_id,
            update.old_score,
            update.new_score,
            update.update_reason,
            update.performance_trend,
            update.recommended_action,
            json.dumps({'score_change': update.new_score - update.old_score}),
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

    async def _analyze_trends_and_recommendations(self, updates: List[KnowledgeUpdate]):
        """Analizuje trendy i generuje rekomendacje na poziomie systemu"""
        if not updates:
            return

        # Analiza trendów
        improving_count = sum(1 for u in updates if u.performance_trend == 'improving')
        declining_count = sum(1 for u in updates if u.performance_trend == 'declining')

        # Analiza akcji
        boost_count = sum(1 for u in updates if u.recommended_action == 'boost')
        retire_count = sum(1 for u in updates if u.recommended_action == 'retire')

        logger.info(f"📈 Performance Analysis:")
        logger.info(f"  - Improving: {improving_count}/{len(updates)}")
        logger.info(f"  - Declining: {declining_count}/{len(updates)}")
        logger.info(f"  - Recommended to boost: {boost_count}")
        logger.info(f"  - Recommended to retire: {retire_count}")

        # Identyfikuj top performers
        top_performers = sorted(updates, key=lambda x: x.new_score, reverse=True)[:5]
        logger.info("🏆 Top performing knowledge points:")
        for i, update in enumerate(top_performers, 1):
            logger.info(f"  {i}. ID {update.knowledge_point_id}: {update.new_score:.3f}")

    async def get_performance_insights(self, days: int = 30) -> Dict:
        """Zwraca insights o performance systemu"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        # Ogólne statystyki
        cursor.execute('''
            SELECT 
                AVG(conversion_rate) as avg_conversion,
                AVG(engagement_rate) as avg_engagement,
                SUM(zinzino_leads) as total_leads,
                SUM(reach) as total_reach,
                COUNT(*) as total_posts
            FROM content_performance 
            WHERE timestamp > ?
        ''', (cutoff_date,))

        stats = cursor.fetchone()

        # Top performing categories
        cursor.execute('''
            SELECT kp.category, 
                   AVG(cp.conversion_rate) as avg_conversion,
                   COUNT(*) as post_count,
                   SUM(cp.zinzino_leads) as total_leads
            FROM content_performance cp
            JOIN knowledge_points kp ON cp.knowledge_point_id = kp.id
            WHERE cp.timestamp > ?
            GROUP BY kp.category
            ORDER BY avg_conversion DESC
        ''', (cutoff_date,))

        categories = cursor.fetchall()
        conn.close()

        return {
            'period_days': days,
            'overall_stats': {
                'avg_conversion_rate': stats[0] or 0,
                'avg_engagement_rate': stats[1] or 0,
                'total_zinzino_leads': stats[2] or 0,
                'total_reach': stats[3] or 0,
                'total_posts': stats[4] or 0
            },
            'category_performance': [
                {
                    'category': cat[0],
                    'avg_conversion': cat[1],
                    'post_count': cat[2],
                    'total_leads': cat[3]
                } for cat in categories
            ]
        }


# Test funkcji
async def test_knowledge_updater():
    """Test knowledge updater"""

    updater = KnowledgeUpdater()

    print("🔄 Testing Knowledge Updater...")
    print("=" * 50)

    # Test 1: Symuluj performance data
    print("📊 Simulating performance data...")

    # Stwórz fake performance metrics
    fake_metrics = [
        PerformanceMetrics(
            knowledge_point_id=1,
            engagement_rate=0.08,
            click_through_rate=0.03,
            conversion_rate=0.06,  # Good conversion for Zinzino
            reach=1500,
            zinzino_leads_generated=90,
            content_quality_score=0.8,
            viral_potential=0.02,
            timestamp=datetime.now().isoformat(),
            platform="facebook"
        ),
        PerformanceMetrics(
            knowledge_point_id=1,
            engagement_rate=0.09,
            click_through_rate=0.04,
            conversion_rate=0.07,  # Improving
            reach=1800,
            zinzino_leads_generated=126,
            content_quality_score=0.85,
            viral_potential=0.03,
            timestamp=(datetime.now() - timedelta(days=1)).isoformat(),
            platform="facebook"
        )
    ]

    # Record fake metrics
    for metrics in fake_metrics:
        await updater.record_content_performance(metrics)

    print("✅ Performance data recorded")

    # Test 2: Sprawdź insights
    insights = await updater.get_performance_insights(days=30)

    print(f"\n📈 Performance Insights:")
    print(f"  - Total posts: {insights['overall_stats']['total_posts']}")
    print(f"  - Avg conversion rate: {insights['overall_stats']['avg_conversion_rate']:.1%}")
    print(f"  - Total Zinzino leads: {insights['overall_stats']['total_zinzino_leads']}")
    print(f"  - Total reach: {insights['overall_stats']['total_reach']:,}")


if __name__ == "__main__":
    asyncio.run(test_knowledge_updater())