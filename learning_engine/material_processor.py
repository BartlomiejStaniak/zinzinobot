#!/usr/bin/.env python3
"""
material_processor.py - Główny procesor materiałów uczących
Plik: learning_engine/material_processor.py
"""

import asyncio
import os
import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json
import logging

# Import lokalnych modułów
from core.scientific_validator import ScientificValidator, ValidationResult
from learning_engine.content_extractor import ContentExtractor, ExtractedContent
from learning_engine.knowledge_updater import KnowledgeUpdater
from learning_engine.adaptive_learning import AdaptiveLearningSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessedMaterial:
    """Reprezentuje przetworzone materiały uczące"""
    file_path: str
    material_type: str  # 'pdf', 'video', 'audio', 'image', 'text'
    extracted_content: ExtractedContent
    validation_result: ValidationResult
    knowledge_points: List[Dict]
    zinzino_relevance: float  # 0.0 - 1.0
    wellness_category: str
    processing_timestamp: str
    content_hash: str
    performance_score: float  # Aktualizowane na podstawie użycia


@dataclass
class LearningObjective:
    """Cel uczenia się systemu"""
    category: str  # 'zinzino_products', 'wellness_tips', 'mental_health', 'productivity'
    priority: int  # 1-10 (10 = najwyższy)
    target_knowledge_points: int
    current_knowledge_points: int
    success_metrics: Dict[str, float]
    improvement_needed: bool


class MaterialProcessor:
    """
    Główny procesor materiałów uczących - serce Learning Engine
    """

    def __init__(self, knowledge_base_path: str = "knowledge_base/",
                 data_path: str = "data/"):

        self.knowledge_base_path = Path(knowledge_base_path)
        self.data_path = Path(data_path)
        self.db_path = self.data_path / "learning_materials.db"

        # Inicjalizuj komponenty
        self.scientific_validator = ScientificValidator()
        self.content_extractor = ContentExtractor()
        self.knowledge_updater = KnowledgeUpdater()
        self.adaptive_learning = AdaptiveLearningSystem()

        # Kategorie wiedzy (dostosowane do Zinzino + wellness)
        self.knowledge_categories = {
            'zinzino_products': {
                'keywords': ['zinzino', 'balance oil', 'omega-3', 'polyphenol', 'balance test'],
                'priority': 10,
                'target_performance': 0.85
            },
            'mental_health': {
                'keywords': ['stress', 'anxiety', 'depression', 'mental health', 'wellbeing', 'mindfulness'],
                'priority': 9,
                'target_performance': 0.80
            },
            'productivity': {
                'keywords': ['productivity', 'focus', 'concentration', 'energy', 'motivation', 'performance'],
                'priority': 8,
                'target_performance': 0.75
            },
            'nutrition_wellness': {
                'keywords': ['nutrition', 'diet', 'supplements', 'vitamins', 'minerals', 'health'],
                'priority': 7,
                'target_performance': 0.70
            },
            'lifestyle_optimization': {
                'keywords': ['sleep', 'exercise', 'habits', 'routine', 'lifestyle', 'wellness'],
                'priority': 6,
                'target_performance': 0.65
            }
        }

        # Kryteria jakości materiałów
        self.quality_thresholds = {
            'minimum_scientific_credibility': 0.60,
            'minimum_zinzino_relevance': 0.30,
            'maximum_corporate_bias': 0.40,
            'minimum_content_length': 100,  # minimum znaków
            'preferred_source_types': ['peer_reviewed', 'medical_institution', 'university']
        }

        # Obsługiwane formaty
        self.supported_formats = {
            'documents': ['.pdf', '.txt', '.docx', '.md'],
            'videos': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
            'audio': ['.mp3', '.wav', '.flac', '.m4a'],
            'images': ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        }

        self._init_database()
        self._setup_directories()

    def _init_database(self):
        """Inicjalizuje bazę danych materiałów uczących"""
        self.data_path.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tabela przetworzonych materiałów
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_materials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                material_type TEXT NOT NULL,
                content_hash TEXT UNIQUE NOT NULL,
                scientific_credibility REAL NOT NULL,
                zinzino_relevance REAL NOT NULL,
                wellness_category TEXT NOT NULL,
                corporate_bias_score REAL NOT NULL,
                knowledge_points_count INTEGER NOT NULL,
                processing_timestamp TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                performance_score REAL DEFAULT 0.5,
                usage_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                is_active BOOLEAN DEFAULT 1
            )
        ''')

        # Tabela punktów wiedzy
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                material_id INTEGER REFERENCES processed_materials(id),
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                zinzino_specific BOOLEAN DEFAULT 0,
                performance_impact REAL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                last_used TEXT,
                usage_count INTEGER DEFAULT 0
            )
        ''')

        # Tabela celów uczenia
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_objectives (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT UNIQUE NOT NULL,
                priority INTEGER NOT NULL,
                target_knowledge_points INTEGER NOT NULL,
                current_knowledge_points INTEGER DEFAULT 0,
                success_metrics TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                improvement_needed BOOLEAN DEFAULT 1
            )
        ''')

        # Tabela performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                knowledge_point_id INTEGER REFERENCES knowledge_points(id),
                content_generated TEXT NOT NULL,
                engagement_score REAL DEFAULT 0.0,
                conversion_score REAL DEFAULT 0.0,
                zinzino_leads INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL,
                platform TEXT NOT NULL
            )
        ''')

        conn.commit()
        conn.close()

        # Inicjalizuj cele uczenia
        self._initialize_learning_objectives()

    def _setup_directories(self):
        """Tworzy strukturę katalogów"""
        directories = [
            self.knowledge_base_path / "documents",
            self.knowledge_base_path / "videos",
            self.knowledge_base_path / "audio",
            self.knowledge_base_path / "images",
            self.knowledge_base_path / "processed",
            self.knowledge_base_path / "embeddings",
            self.data_path / "vector_db"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _initialize_learning_objectives(self):
        """Inicjalizuje cele uczenia w bazie danych"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for category, details in self.knowledge_categories.items():
            cursor.execute('''
                INSERT OR REPLACE INTO learning_objectives 
                (category, priority, target_knowledge_points, success_metrics, last_updated)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                category,
                details['priority'],
                100,  # Target: 100 knowledge points per category
                json.dumps({'target_performance': details['target_performance']}),
                datetime.now().isoformat()
            ))

        conn.commit()
        conn.close()

    async def process_new_materials(self, scan_directories: bool = True) -> List[ProcessedMaterial]:
        """
        Główna metoda - przetwarza nowe materiały
        """
        logger.info("🧠 Starting material processing...")

        processed_materials = []

        if scan_directories:
            # Skanuj katalogi w poszukiwaniu nowych plików
            new_files = await self._scan_for_new_files()
            logger.info(f"Found {len(new_files)} new files to process")

            for file_path in new_files:
                try:
                    processed = await self.process_single_material(file_path)
                    if processed:
                        processed_materials.append(processed)
                        logger.info(f"✅ Processed: {file_path.name}")
                    else:
                        logger.warning(f"❌ Skipped: {file_path.name}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")

        # Aktualizuj cele uczenia
        await self._update_learning_objectives()

        # Uruchom adaptacyjne uczenie
        await self.adaptive_learning.update_knowledge_from_performance()

        logger.info(f"🎯 Processing complete. {len(processed_materials)} materials added to knowledge base")
        return processed_materials

    async def process_single_material(self, file_path: Path) -> Optional[ProcessedMaterial]:
        """
        Przetwarza pojedynczy materiał
        """
        logger.info(f"Processing: {file_path}")

        # 1. Sprawdź czy już przetworzony
        if await self._is_already_processed(file_path):
            logger.info(f"Already processed: {file_path}")
            return None

        # 2. Określ typ materiału
        material_type = self._determine_material_type(file_path)
        if not material_type:
            logger.warning(f"Unsupported file type: {file_path}")
            return None

        # 3. Wyciągnij treść
        extracted_content = await self.content_extractor.extract_content(file_path, material_type)
        if not extracted_content or len(extracted_content.main_text) < self.quality_thresholds[
            'minimum_content_length']:
            logger.warning(f"Insufficient content extracted from: {file_path}")
            return None

        # 4. Walidacja naukowa
        validation_result = await self.scientific_validator.validate_content(extracted_content.main_text)

        # 5. Sprawdź kryteria jakości
        if not self._meets_quality_criteria(validation_result, extracted_content):
            logger.warning(f"Content does not meet quality criteria: {file_path}")
            return None

        # 6. Oceń relevance dla Zinzino
        zinzino_relevance = self._calculate_zinzino_relevance(extracted_content.main_text)

        # 7. Kategoryzuj wellness
        wellness_category = self._categorize_wellness_content(extracted_content.main_text)

        # 8. Wyciągnij punkty wiedzy
        knowledge_points = await self._extract_knowledge_points(extracted_content, wellness_category)

        # 9. Oblicz hash zawartości
        content_hash = self._calculate_content_hash(extracted_content.main_text)

        # 10. Stwórz ProcessedMaterial object
        processed_material = ProcessedMaterial(
            file_path=str(file_path),
            material_type=material_type,
            extracted_content=extracted_content,
            validation_result=validation_result,
            knowledge_points=knowledge_points,
            zinzino_relevance=zinzino_relevance,
            wellness_category=wellness_category,
            processing_timestamp=datetime.now().isoformat(),
            content_hash=content_hash,
            performance_score=0.5  # Początkowy neutralny score
        )

        # 11. Zapisz do bazy danych
        await self._save_processed_material(processed_material)

        # 12. Aktualizuj system adaptacyjnego uczenia
        await self.adaptive_learning.integrate_new_knowledge(knowledge_points, wellness_category)

        logger.info(f"✅ Successfully processed {file_path} -> {len(knowledge_points)} knowledge points")
        return processed_material

    async def _scan_for_new_files(self) -> List[Path]:
        """Skanuje katalogi w poszukiwaniu nowych plików"""
        new_files = []

        for category, extensions in self.supported_formats.items():
            category_path = self.knowledge_base_path / category.rstrip('s')  # documents -> document

            if category_path.exists():
                for ext in extensions:
                    files = list(category_path.glob(f"*{ext}"))
                    for file_path in files:
                        if not await self._is_already_processed(file_path):
                            new_files.append(file_path)

        return new_files

    async def _is_already_processed(self, file_path: Path) -> bool:
        """Sprawdza czy plik już został przetworzony"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT id FROM processed_materials WHERE file_path = ?', (str(file_path),))
        result = cursor.fetchone()
        conn.close()

        return result is not None

    def _determine_material_type(self, file_path: Path) -> Optional[str]:
        """Określa typ materiału na podstawie rozszerzenia"""
        suffix = file_path.suffix.lower()

        for category, extensions in self.supported_formats.items():
            if suffix in extensions:
                return category.rstrip('s')  # documents -> document

        return None

    def _meets_quality_criteria(self, validation_result: ValidationResult, extracted_content: ExtractedContent) -> bool:
        """Sprawdza czy materiał spełnia kryteria jakości"""

        # Sprawdź wiarygodność naukową
        if validation_result.overall_credibility < self.quality_thresholds['minimum_scientific_credibility']:
            logger.warning(f"Low scientific credibility: {validation_result.overall_credibility:.2f}")
            return False

        # Sprawdź corporate bias
        if validation_result.bias_score > self.quality_thresholds['maximum_corporate_bias']:
            logger.warning(f"High corporate bias: {validation_result.bias_score:.2f}")
            return False

        # Sprawdź długość treści
        if len(extracted_content.main_text) < self.quality_thresholds['minimum_content_length']:
            logger.warning(f"Content too short: {len(extracted_content.main_text)} chars")
            return False

        # Dodatkowe sprawdzenie: czy nie promuje corporate bias (radioterapia vs ketoza)
        content_lower = extracted_content.main_text.lower()
        radiotherapy_positive = any(
            term in content_lower for term in ['radiotherapy gold standard', 'radiation therapy proven'])
        ketosis_negative = any(term in content_lower for term in ['ketogenic dangerous', 'ketosis unproven'])

        if radiotherapy_positive and ketosis_negative:
            logger.warning("Corporate bias detected: radiotherapy promoted over ketosis")
            return False

        return True

    def _calculate_zinzino_relevance(self, content: str) -> float:
        """Oblicza relevance dla Zinzino (0.0 - 1.0)"""
        content_lower = content.lower()
        relevance_score = 0.0

        # Bezpośrednie wzmianki o Zinzino
        zinzino_terms = ['zinzino', 'balance oil', 'balance test', 'polyphenol omega']
        for term in zinzino_terms:
            if term in content_lower:
                relevance_score += 0.3

        # Tematy związane z produktami Zinzino
        product_related = ['omega-3', 'omega 3', 'fish oil', 'fatty acids', 'inflammation', 'polyphenol']
        for term in product_related:
            if term in content_lower:
                relevance_score += 0.15

        # Tematy wellness (target audience Zinzino)
        wellness_terms = ['wellness', 'health optimization', 'nutritional balance', 'immune support']
        for term in wellness_terms:
            if term in content_lower:
                relevance_score += 0.10

        # Tematy biznesowe (MLM/network marketing)
        business_terms = ['network marketing', 'direct sales', 'business opportunity', 'entrepreneurship']
        for term in business_terms:
            if term in content_lower:
                relevance_score += 0.20

        return min(relevance_score, 1.0)

    def _categorize_wellness_content(self, content: str) -> str:
        """Kategoryzuje treść wellness"""
        content_lower = content.lower()
        category_scores = {}

        for category, details in self.knowledge_categories.items():
            score = 0
            for keyword in details['keywords']:
                if keyword in content_lower:
                    score += 1

            if score > 0:
                category_scores[category] = score

        if category_scores:
            return max(category_scores, key=category_scores.get)
        else:
            return 'general_wellness'

    async def _extract_knowledge_points(self, extracted_content: ExtractedContent, category: str) -> List[Dict]:
        """Wyciąga punkty wiedzy z treści"""
        knowledge_points = []

        # Podziel treść na sekcje
        sections = self._split_content_into_sections(extracted_content.main_text)

        for i, section in enumerate(sections):
            if len(section.strip()) < 50:  # Skip short sections
                continue

            # Sprawdź czy sekcja zawiera użyteczną wiedzę
            if self._is_valuable_knowledge(section, category):
                # Wyciągnij tytuł sekcji
                title = self._extract_section_title(section)

                # Oceń confidence
                confidence = self._assess_knowledge_confidence(section)

                # Sprawdź czy specific dla Zinzino
                zinzino_specific = self._is_zinzino_specific(section)

                knowledge_point = {
                    'title': title,
                    'content': section.strip(),
                    'category': category,
                    'confidence_score': confidence,
                    'zinzino_specific': zinzino_specific,
                    'source_section': i,
                    'extracted_at': datetime.now().isoformat()
                }

                knowledge_points.append(knowledge_point)

        return knowledge_points

    def _split_content_into_sections(self, content: str) -> List[str]:
        """Dzieli treść na logiczne sekcje"""
        # Prosta implementacja - można rozszerzyć o bardziej zaawansowaną segmentację

        # Split by double newlines (paragraphs)
        paragraphs = content.split('\n\n')

        # Group related paragraphs
        sections = []
        current_section = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Start new section if paragraph looks like a header
            if (len(paragraph) < 100 and
                    (paragraph.isupper() or
                     paragraph.count('.') == 0 or
                     any(indicator in paragraph.lower() for indicator in
                         ['chapter', 'section', 'benefits', 'how to']))):

                if current_section:
                    sections.append(current_section)
                current_section = paragraph + "\n\n"
            else:
                current_section += paragraph + "\n\n"

        # Add final section
        if current_section:
            sections.append(current_section)

        return sections

    def _is_valuable_knowledge(self, section: str, category: str) -> bool:
        """Sprawdza czy sekcja zawiera wartościową wiedzę"""
        section_lower = section.lower()

        # Sprawdź czy zawiera actionable information
        actionable_indicators = ['how to', 'steps to', 'method', 'technique', 'approach', 'strategy']
        has_actionable = any(indicator in section_lower for indicator in actionable_indicators)

        # Sprawdź czy zawiera keywords związane z kategorią
        if category in self.knowledge_categories:
            keywords = self.knowledge_categories[category]['keywords']
            has_relevant_keywords = any(keyword in section_lower for keyword in keywords)
        else:
            has_relevant_keywords = True

        # Sprawdź czy nie jest zbyt ogólne
        too_general = any(phrase in section_lower for phrase in ['in general', 'generally speaking', 'it depends'])

        return (has_actionable or has_relevant_keywords) and not too_general and len(section) > 100

    def _extract_section_title(self, section: str) -> str:
        """Wyciąga tytuł sekcji"""
        lines = section.split('\n')
        first_line = lines[0].strip()

        # Jeśli pierwsza linia jest krótka, prawdopodobnie to tytuł
        if len(first_line) < 100:
            return first_line
        else:
            # Spróbuj wyciągnąć pierwsze zdanie
            sentences = first_line.split('.')
            if sentences and len(sentences[0]) < 100:
                return sentences[0].strip()
            else:
                return first_line[:75] + "..."

    def _assess_knowledge_confidence(self, section: str) -> float:
        """Ocenia confidence wiedzy w sekcji"""
        section_lower = section.lower()
        confidence = 0.5

        # Wskaźniki wysokiej pewności
        high_confidence = ['research shows', 'studies indicate', 'proven', 'demonstrated', 'evidence suggests']
        for indicator in high_confidence:
            if indicator in section_lower:
                confidence += 0.15

        # Wskaźniki niskiej pewności
        low_confidence = ['may', 'might', 'possibly', 'unclear', 'uncertain', 'limited evidence']
        for indicator in low_confidence:
            if indicator in section_lower:
                confidence -= 0.10

        # Konkretne dane/liczby zwiększają pewność
        import re
        if re.search(r'\d+%|\d+\.\d+', section):
            confidence += 0.05

        return max(0.0, min(1.0, confidence))

    def _is_zinzino_specific(self, section: str) -> bool:
        """Sprawdza czy sekcja jest specyficzna dla Zinzino"""
        section_lower = section.lower()

        zinzino_indicators = [
            'zinzino', 'balance oil', 'balance test', 'polyphenol omega',
            'omega balance', 'zinzino partner', 'network marketing zinzino'
        ]

        return any(indicator in section_lower for indicator in zinzino_indicators)

    def _calculate_content_hash(self, content: str) -> str:
        """Oblicza hash zawartości"""
        return hashlib.md5(content.encode()).hexdigest()

    async def _save_processed_material(self, material: ProcessedMaterial):
        """Zapisuje przetworzone materiały do bazy danych"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Zapisz główne informacje o materiale
        cursor.execute('''
            INSERT INTO processed_materials 
            (file_path, material_type, content_hash, scientific_credibility, 
             zinzino_relevance, wellness_category, corporate_bias_score,
             knowledge_points_count, processing_timestamp, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            material.file_path,
            material.material_type,
            material.content_hash,
            material.validation_result.overall_credibility,
            material.zinzino_relevance,
            material.wellness_category,
            material.validation_result.bias_score,
            len(material.knowledge_points),
            material.processing_timestamp,
            datetime.now().isoformat()
        ))

        material_id = cursor.lastrowid

        # Zapisz punkty wiedzy
        for kp in material.knowledge_points:
            cursor.execute('''
                INSERT INTO knowledge_points 
                (material_id, category, title, content, confidence_score, 
                 zinzino_specific, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                material_id,
                kp['category'],
                kp['title'],
                kp['content'],
                kp['confidence_score'],
                kp['zinzino_specific'],
                kp['extracted_at']
            ))

        conn.commit()
        conn.close()

    async def _update_learning_objectives(self):
        """Aktualizuje cele uczenia na podstawie przetworzonego materiału"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Policz obecne punkty wiedzy w każdej kategorii
        for category in self.knowledge_categories.keys():
            cursor.execute('''
                SELECT COUNT(*) FROM knowledge_points WHERE category = ?
            ''', (category,))
            current_count = cursor.fetchone()[0]

            # Aktualizuj cel uczenia
            cursor.execute('''
                UPDATE learning_objectives 
                SET current_knowledge_points = ?, last_updated = ?
                WHERE category = ?
            ''', (current_count, datetime.now().isoformat(), category))

        conn.commit()
        conn.close()

    async def get_knowledge_for_content_generation(self, category: str, limit: int = 10) -> List[Dict]:
        """Pobiera wiedzę do generowania contentu"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT kp.*, pm.performance_score 
            FROM knowledge_points kp
            JOIN processed_materials pm ON kp.material_id = pm.id
            WHERE kp.category = ? AND pm.is_active = 1
            ORDER BY kp.confidence_score DESC, pm.performance_score DESC
            LIMIT ?
        ''', (category, limit))

        results = cursor.fetchall()
        conn.close()

        knowledge_points = []
        for row in results:
            knowledge_points.append({
                'id': row[0],
                'title': row[3],
                'content': row[4],
                'confidence_score': row[5],
                'zinzino_specific': row[6],
                'performance_score': row[-1]
            })

        return knowledge_points

    async def get_learning_status(self) -> Dict:
        """Zwraca status uczenia się systemu"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Pobierz cele uczenia
        cursor.execute('SELECT * FROM learning_objectives ORDER BY priority DESC')
        objectives = cursor.fetchall()

        # Pobierz statystyki materiałów
        cursor.execute('''
            SELECT 
                COUNT(*) as total_materials,
                AVG(scientific_credibility) as avg_credibility,
                AVG(zinzino_relevance) as avg_zinzino_relevance,
                SUM(knowledge_points_count) as total_knowledge_points
            FROM processed_materials WHERE is_active = 1
        ''')
        stats = cursor.fetchone()

        conn.close()

        return {
            'learning_objectives': objectives,
            'total_materials': stats[0],
            'average_credibility': stats[1] or 0,
            'average_zinzino_relevance': stats[2] or 0,
            'total_knowledge_points': stats[3] or 0,
            'last_updated': datetime.now().isoformat()
        }


# Test funkcji
async def test_material_processor():
    """Test procesora materiałów"""

    processor = MaterialProcessor()

    print("🧠 Testing Material Processor...")
    print("=" * 50)

    # Test 1: Sprawdź status uczenia
    status = await processor.get_learning_status()
    print(f"Learning Status:")
    print(f"- Total materials: {status['total_materials']}")
    print(f"- Average credibility: {status['average_credibility']:.2f}")
    print(f"- Total knowledge points: {status['total_knowledge_points']}")

    # Test 2: Symuluj przetwarzanie materiału tekstowego
    test_content = """
    # Omega-3 Fatty Acids and Mental Health

    Recent systematic reviews have demonstrated that omega-3 fatty acids, 
    particularly EPA and DHA, play crucial roles in brain function and mental health. 
    Studies show that proper omega-3 balance can significantly reduce symptoms of 
    depression and anxiety.

    ## Key Benefits:
    - Improved mood regulation
    - Enhanced cognitive function  
    - Reduced inflammation in the brain
    - Better stress management

    The ideal omega-6 to omega-3 ratio should be between 3:1 and 5:1 for optimal health.
    Many people have ratios as high as 20:1 due to modern diets, leading to chronic inflammation.

    ## Zinzino BalanceOil Approach:
    Zinzino's BalanceOil combines high-quality omega-3 from fish oil with polyphenols 
    from cold-pressed olive oil. This synergistic formula helps achieve optimal omega balance
    as verified by the BalanceTest.

    Research indicates that this approach can improve omega balance within 120 days,
    supporting both physical and mental wellness.
    """

    # Zapisz jako tymczasowy plik
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file = Path(f.name)

    print(f"\n📄 Processing test material...")

    try:
        # Przetwórz materiał
        processed = await processor.process_single_material(temp_file)

        if processed:
            print(f"✅ Successfully processed!")
            print(f"- Material type: {processed.material_type}")
            print(f"- Scientific credibility: {processed.validation_result.overall_credibility:.2f}")
            print(f"- Zinzino relevance: {processed.zinzino_relevance:.2f}")
            print(f"- Wellness category: {processed.wellness_category}")
            print(f"- Knowledge points: {len(processed.knowledge_points)}")
            print(f"- Corporate bias: {processed.validation_result.bias_score:.2f}")

            # Pokaż przykładowe punkty wiedzy
            if processed.knowledge_points:
                print(f"\n📚 Sample Knowledge Points:")
                for i, kp in enumerate(processed.knowledge_points[:2]):
                    print(f"{i + 1}. {kp['title']}")
                    print(f"   Confidence: {kp['confidence_score']:.2f}")
                    print(f"   Zinzino specific: {kp['zinzino_specific']}")
                    print(f"   Content preview: {kp['content'][:100]}...")
        else:
            print("❌ Material was not processed (failed quality criteria)")

    finally:
        # Cleanup
        temp_file.unlink()

    # Test 3: Pobierz wiedzę do generowania contentu
    print(f"\n🎯 Testing knowledge retrieval...")
    knowledge = await processor.get_knowledge_for_content_generation('mental_health', limit=3)
    print(f"Retrieved {len(knowledge)} knowledge points for mental_health category")

    for kp in knowledge:
        print(f"- {kp['title']} (confidence: {kp['confidence_score']:.2f})")


if __name__ == "__main__":
    asyncio.run(test_material_processor())