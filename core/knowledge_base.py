#!/usr/bin/.env python3
"""
knowledge_base.py - Centralna baza wiedzy systemu
Plik: core/knowledge_base.py
"""

import asyncio
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import numpy as np
from collections import defaultdict

# Import dla vector database (ChromaDB)
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None
    logging.warning("ChromaDB not installed. Vector search will be limited.")

from core.scientific_validator import ScientificValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KnowledgeItem:
    """Pojedynczy element wiedzy"""
    id: str
    content: str
    category: str
    source: str
    platform_tags: List[str]  # ['facebook', 'instagram', 'tiktok']
    credibility_score: float
    usage_count: int
    performance_score: float
    created_at: datetime
    last_used: Optional[datetime]
    metadata: Dict[str, Any]


@dataclass
class KnowledgeQuery:
    """Zapytanie do bazy wiedzy"""
    query_text: str
    category: Optional[str]
    platform: Optional[str]
    min_credibility: float = 0.7
    limit: int = 10


class KnowledgeBase:
    """
    Centralna baza wiedzy z obsługą multi-platform
    """

    def __init__(self, db_path: str = "data/knowledge_base.db",
                 vector_db_path: str = "data/vector_db"):
        self.db_path = db_path
        self.vector_db_path = vector_db_path

        # Platform-specific configurations
        self.platform_configs = {
            'facebook': {
                'max_content_length': 63206,
                'preferred_categories': ['education', 'engagement', 'product_info'],
                'content_style': 'detailed'
            },
            'instagram': {
                'max_content_length': 2200,
                'preferred_categories': ['visual', 'lifestyle', 'quick_tips'],
                'content_style': 'concise'
            },
            'tiktok': {
                'max_content_length': 150,
                'preferred_categories': ['trending', 'quick_tips', 'entertainment'],
                'content_style': 'snappy'
            }
        }

        self.scientific_validator = ScientificValidator()
        self._init_database()
        self._init_vector_db()

    def _init_database(self):
        """Inicjalizuje bazę SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_items (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                source TEXT NOT NULL,
                platform_tags TEXT NOT NULL,
                credibility_score REAL NOT NULL,
                usage_count INTEGER DEFAULT 0,
                performance_score REAL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                last_used TEXT,
                metadata TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item1_id TEXT NOT NULL,
                item2_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                strength REAL DEFAULT 0.5,
                FOREIGN KEY (item1_id) REFERENCES knowledge_items(id),
                FOREIGN KEY (item2_id) REFERENCES knowledge_items(id)
            )
        ''')

        conn.commit()
        conn.close()

    def _init_vector_db(self):
        """Inicjalizuje vector database"""
        if chromadb:
            self.chroma_client = chromadb.PersistentClient(
                path=self.vector_db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name="knowledge_vectors"
            )
        else:
            self.chroma_client = None
            self.collection = None

    async def add_knowledge(self, content: str, source: str,
                            category: str, platforms: List[str]) -> KnowledgeItem:
        """Dodaje wiedzę do bazy z walidacją"""
        # Walidacja naukowa
        validation = await self.scientific_validator.validate_content(content)

        if not validation.should_use_content:
            raise ValueError(f"Content rejected: {validation.agent_summary}")

        # Tworzenie knowledge item
        item = KnowledgeItem(
            id=self._generate_id(),
            content=content,
            category=category,
            source=source,
            platform_tags=platforms,
            credibility_score=validation.overall_credibility,
            usage_count=0,
            performance_score=0.5,
            created_at=datetime.now(),
            last_used=None,
            metadata={
                'validation_result': validation.agent_summary,
                'red_flags': validation.red_flags
            }
        )

        # Zapisz do SQLite
        await self._save_to_db(item)

        # Zapisz do vector DB
        if self.collection:
            await self._save_to_vector_db(item)

        return item

    async def query_knowledge(self, query: KnowledgeQuery) -> List[KnowledgeItem]:
        """Wyszukuje wiedzę z filtrowaniem per-platform"""
        if self.collection and query.query_text:
            # Vector search
            results = self.collection.query(
                query_texts=[query.query_text],
                n_results=query.limit,
                where=self._build_where_clause(query)
            )

            item_ids = results['ids'][0] if results['ids'] else []
            items = await self._get_items_by_ids(item_ids)
        else:
            # Fallback to SQL search
            items = await self._sql_search(query)

        # Filtruj i sortuj według platform
        if query.platform:
            items = self._filter_for_platform(items, query.platform)

        return items

    async def update_performance(self, item_id: str, performance_data: Dict):
        """Aktualizuje performance score na podstawie użycia"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Pobierz obecne dane
        cursor.execute(
            "SELECT usage_count, performance_score FROM knowledge_items WHERE id = ?",
            (item_id,)
        )
        result = cursor.fetchone()

        if result:
            usage_count, old_score = result

            # Oblicz nowy score
            engagement_rate = performance_data.get('engagement_rate', 0)
            conversion_rate = performance_data.get('conversion_rate', 0)
            platform = performance_data.get('platform', 'facebook')

            # Platform-specific weights
            weights = {
                'facebook': {'engagement': 0.4, 'conversion': 0.6},
                'instagram': {'engagement': 0.6, 'conversion': 0.4},
                'tiktok': {'engagement': 0.7, 'conversion': 0.3}
            }

            w = weights.get(platform, weights['facebook'])
            new_score = (
                    w['engagement'] * engagement_rate +
                    w['conversion'] * conversion_rate
            )

            # Weighted average with old score
            final_score = (old_score * usage_count + new_score) / (usage_count + 1)

            # Update
            cursor.execute('''
                UPDATE knowledge_items 
                SET performance_score = ?, usage_count = ?, last_used = ?
                WHERE id = ?
            ''', (final_score, usage_count + 1, datetime.now().isoformat(), item_id))

        conn.commit()
        conn.close()

    def _filter_for_platform(self, items: List[KnowledgeItem],
                             platform: str) -> List[KnowledgeItem]:
        """Filtruje i dostosowuje content dla platformy"""
        filtered = []
        config = self.platform_configs.get(platform, self.platform_configs['facebook'])

        for item in items:
            if platform in item.platform_tags:
                # Dostosuj długość contentu
                if len(item.content) > config['max_content_length']:
                    item.content = item.content[:config['max_content_length']] + "..."

                # Preferuj odpowiednie kategorie
                if item.category in config['preferred_categories']:
                    item.performance_score *= 1.2  # Boost

                filtered.append(item)

        # Sortuj według dopasowania do platformy
        return sorted(filtered, key=lambda x: x.performance_score, reverse=True)

    def _generate_id(self) -> str:
        """Generuje unikalny ID"""
        import uuid
        return f"kb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    async def _save_to_db(self, item: KnowledgeItem):
        """Zapisuje do SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO knowledge_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item.id,
            item.content,
            item.category,
            item.source,
            json.dumps(item.platform_tags),
            item.credibility_score,
            item.usage_count,
            item.performance_score,
            item.created_at.isoformat(),
            item.last_used.isoformat() if item.last_used else None,
            json.dumps(item.metadata)
        ))

        conn.commit()
        conn.close()

    async def _save_to_vector_db(self, item: KnowledgeItem):
        """Zapisuje do vector database"""
        if self.collection:
            self.collection.add(
                documents=[item.content],
                metadatas=[{
                    'category': item.category,
                    'source': item.source,
                    'platforms': ','.join(item.platform_tags),
                    'credibility': item.credibility_score
                }],
                ids=[item.id]
            )

    def _build_where_clause(self, query: KnowledgeQuery) -> Dict:
        """Buduje where clause dla ChromaDB"""
        where = {}

        if query.category:
            where['category'] = query.category

        if query.platform:
            where['platforms'] = {'$contains': query.platform}

        where['credibility'] = {'$gte': query.min_credibility}

        return where

    async def _get_items_by_ids(self, ids: List[str]) -> List[KnowledgeItem]:
        """Pobiera items po ID z SQLite"""
        if not ids:
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        placeholders = ','.join('?' * len(ids))
        cursor.execute(
            f"SELECT * FROM knowledge_items WHERE id IN ({placeholders})",
            ids
        )

        items = []
        for row in cursor.fetchall():
            items.append(self._row_to_item(row))

        conn.close()
        return items

    async def _sql_search(self, query: KnowledgeQuery) -> List[KnowledgeItem]:
        """Fallback SQL search"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        sql = '''
            SELECT * FROM knowledge_items 
            WHERE credibility_score >= ?
        '''
        params = [query.min_credibility]

        if query.category:
            sql += " AND category = ?"
            params.append(query.category)

        if query.platform:
            sql += " AND platform_tags LIKE ?"
            params.append(f'%{query.platform}%')

        if query.query_text:
            sql += " AND content LIKE ?"
            params.append(f'%{query.query_text}%')

        sql += " ORDER BY performance_score DESC LIMIT ?"
        params.append(query.limit)

        cursor.execute(sql, params)

        items = []
        for row in cursor.fetchall():
            items.append(self._row_to_item(row))

        conn.close()
        return items

    def _row_to_item(self, row) -> KnowledgeItem:
        """Konwertuje row z SQL na KnowledgeItem"""
        return KnowledgeItem(
            id=row[0],
            content=row[1],
            category=row[2],
            source=row[3],
            platform_tags=json.loads(row[4]),
            credibility_score=row[5],
            usage_count=row[6],
            performance_score=row[7],
            created_at=datetime.fromisoformat(row[8]),
            last_used=datetime.fromisoformat(row[9]) if row[9] else None,
            metadata=json.loads(row[10]) if row[10] else {}
        )

    async def get_platform_statistics(self) -> Dict[str, Any]:
        """Zwraca statystyki per-platform"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}
        for platform in ['facebook', 'instagram', 'tiktok']:
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    AVG(performance_score) as avg_performance,
                    AVG(credibility_score) as avg_credibility,
                    SUM(usage_count) as total_usage
                FROM knowledge_items
                WHERE platform_tags LIKE ?
            ''', (f'%{platform}%',))

            result = cursor.fetchone()
            stats[platform] = {
                'total_items': result[0] or 0,
                'avg_performance': result[1] or 0,
                'avg_credibility': result[2] or 0,
                'total_usage': result[3] or 0
            }

        conn.close()
        return stats