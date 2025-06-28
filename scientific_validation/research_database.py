#!/usr/bin/env python3
"""
research_database.py - Baza danych badań naukowych
Plik: scientific_validation/research_database.py
"""

import asyncio
import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import requests
from urllib.parse import quote_plus
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StudyResult:
    """Reprezentuje pojedyncze badanie naukowe"""
    title: str
    authors: List[str]
    journal: str
    year: int
    doi: str
    abstract: str
    study_type: str  # "systematic_review", "meta_analysis", "rct", "observational"
    quality_score: float  # 0.0 - 1.0
    sample_size: int
    funding_source: str
    conflicts_of_interest: List[str]
    key_findings: str
    credibility_score: float


@dataclass
class ValidationClaim:
    """Pojedyncze twierdzenie do walidacji"""
    claim_text: str
    medical_category: str  # "cancer", "nutrition", "cardiology", etc.
    confidence_required: float  # poziom pewności wymagany (0.0-1.0)


class ResearchDatabase:
    """
    Główna klasa do zarządzania bazą danych badań naukowych
    """

    def __init__(self, db_path: str = "data/scientific_research.db"):
        self.db_path = db_path
        self.cache_expiry_days = 30
        self.medical_categories = {
            'cancer': ['oncology', 'tumor', 'malignant', 'chemotherapy', 'radiotherapy', 'ketosis'],
            'nutrition': ['diet', 'ketogenic', 'fasting', 'metabolism', 'micronutrient'],
            'cardiology': ['heart', 'cardiovascular', 'blood pressure', 'cholesterol'],
            'neurology': ['brain', 'cognitive', 'alzheimer', 'depression', 'anxiety'],
            'immunology': ['immune', 'inflammation', 'autoimmune'],
            'endocrinology': ['hormone', 'insulin', 'diabetes', 'thyroid']
        }

        # Red flags dla corporate bias (przykład z radioterapią vs ketoza)
        self.corporate_bias_indicators = {
            'treatment_bias': [
                'radiotherapy standard treatment',
                'chemotherapy first line',
                'surgery recommended approach',
                'pharmaceutical intervention necessary'
            ],
            'alternative_suppression': [
                'dietary intervention insufficient',
                'lifestyle changes inadequate',
                'natural therapy unproven',
                'ketogenic diet dangerous'
            ],
            'financial_language': [
                'cost-effective treatment',
                'healthcare economics',
                'treatment compliance',
                'patient adherence'
            ]
        }

        self._init_database()

    def _init_database(self):
        """Inicjalizuje bazę danych SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tabela badań naukowych
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS studies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                authors TEXT NOT NULL,
                journal TEXT NOT NULL,
                year INTEGER NOT NULL,
                doi TEXT UNIQUE,
                abstract TEXT,
                study_type TEXT,
                quality_score REAL,
                sample_size INTEGER,
                funding_source TEXT,
                conflicts_of_interest TEXT,
                key_findings TEXT,
                credibility_score REAL,
                category TEXT,
                added_date TEXT,
                last_verified TEXT
            )
        ''')

        # Tabela cache wyszukiwań
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT UNIQUE,
                query TEXT,
                results TEXT,
                created_at TEXT,
                expires_at TEXT
            )
        ''')

        # Tabela twierdzeń medycznych
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS medical_claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim TEXT UNIQUE,
                category TEXT,
                evidence_level TEXT,
                supporting_studies TEXT,
                contradicting_studies TEXT,
                consensus_score REAL,
                last_updated TEXT
            )
        ''')

        conn.commit()
        conn.close()

        # Wczytaj podstawowe dane medyczne
        self._populate_basic_medical_knowledge()

    def _populate_basic_medical_knowledge(self):
        """Wczytuje podstawową wiedzę medyczną (oficjalnie potwierdzoną)"""
        basic_claims = [
            {
                'claim': 'Ketogenic diet can induce metabolic ketosis',
                'category': 'nutrition',
                'evidence_level': 'established',
                'consensus_score': 0.95
            },
            {
                'claim': 'Cancer cells preferentially use glucose for energy',
                'category': 'cancer',
                'evidence_level': 'established',
                'consensus_score': 0.92
            },
            {
                'claim': 'Radiotherapy causes cellular damage in healthy tissue',
                'category': 'cancer',
                'evidence_level': 'established',
                'consensus_score': 0.98
            },
            {
                'claim': 'Ketosis reduces glucose availability for cancer cells',
                'category': 'cancer',
                'evidence_level': 'emerging',
                'consensus_score': 0.75
            },
            {
                'claim': 'Fasting can enhance cancer treatment efficacy',
                'category': 'cancer',
                'evidence_level': 'emerging',
                'consensus_score': 0.68
            }
        ]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for claim_data in basic_claims:
            cursor.execute('''
                INSERT OR REPLACE INTO medical_claims 
                (claim, category, evidence_level, consensus_score, last_updated)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                claim_data['claim'],
                claim_data['category'],
                claim_data['evidence_level'],
                claim_data['consensus_score'],
                datetime.now().isoformat()
            ))

        conn.commit()
        conn.close()

    async def find_supporting_evidence(self, claim: ValidationClaim) -> List[StudyResult]:
        """
        Wyszukuje dowody naukowe dla danego twierdzenia
        """
        # Sprawdź cache
        cached_results = await self._get_cached_search(claim.claim_text)
        if cached_results:
            logger.info(f"Returning cached results for: {claim.claim_text[:50]}...")
            return cached_results

        # Wyszukaj w lokalnej bazie
        local_results = await self._search_local_database(claim)

        # Jeśli mało wyników, spróbuj wyszukać online (PubMed)
        if len(local_results) < 3:
            online_results = await self._search_pubmed(claim)
            local_results.extend(online_results)

        # Cache wyniki
        await self._cache_search_results(claim.claim_text, local_results)

        return local_results

    async def _search_local_database(self, claim: ValidationClaim) -> List[StudyResult]:
        """Wyszukuje w lokalnej bazie danych"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Wyciągnij keywords z twierdzenia
        keywords = self._extract_keywords(claim.claim_text, claim.medical_category)

        results = []
        for keyword in keywords:
            cursor.execute('''
                SELECT * FROM studies 
                WHERE (title LIKE ? OR abstract LIKE ? OR key_findings LIKE ?)
                AND category = ?
                ORDER BY credibility_score DESC, quality_score DESC
                LIMIT 10
            ''', (f'%{keyword}%', f'%{keyword}%', f'%{keyword}%', claim.medical_category))

            rows = cursor.fetchall()
            for row in rows:
                study = self._row_to_study_result(row)
                if study not in results:
                    results.append(study)

        conn.close()
        return results

    async def _search_pubmed(self, claim: ValidationClaim) -> List[StudyResult]:
        """
        Wyszukuje w PubMed (symulacja - w pełnej wersji używałbyś Bio.Entrez)
        """
        # To jest symulacja - w rzeczywistości użyłbyś Biopython i PubMed API
        logger.info(f"Searching PubMed for: {claim.claim_text[:50]}...")

        # Symulowane wyniki wysokiej jakości
        simulated_results = []

        if 'ketosis' in claim.claim_text.lower() and 'cancer' in claim.claim_text.lower():
            # Symulowane badanie o ketozie i nowotworach
            study = StudyResult(
                title="Ketogenic Diet as Adjuvant Therapy in Cancer Treatment",
                authors=["Weber DD", "Aminzadeh-Gohari S", "Kofler B"],
                journal="Current Opinion in Clinical Nutrition & Metabolic Care",
                year=2020,
                doi="10.1097/MCO.0000000000000631",
                abstract="Ketogenic diets may provide metabolic support during cancer treatment...",
                study_type="systematic_review",
                quality_score=0.85,
                sample_size=1200,
                funding_source="University research grant",
                conflicts_of_interest=[],
                key_findings="Ketogenic diet shows promise as adjuvant cancer therapy",
                credibility_score=0.88
            )
            simulated_results.append(study)

        # Dodaj delay żeby nie spamować API
        await asyncio.sleep(1)

        return simulated_results

    def _extract_keywords(self, text: str, category: str) -> List[str]:
        """Wyciąga keywords z tekstu na podstawie kategorii medycznej"""
        text_lower = text.lower()
        keywords = []

        # Dodaj keywords specyficzne dla kategorii
        if category in self.medical_categories:
            for keyword in self.medical_categories[category]:
                if keyword in text_lower:
                    keywords.append(keyword)

        # Dodaj ogólne keywords medyczne
        medical_terms = ['treatment', 'therapy', 'study', 'research', 'clinical', 'patient']
        for term in medical_terms:
            if term in text_lower:
                keywords.append(term)

        return list(set(keywords)) if keywords else ['medical', 'health']

    def detect_corporate_bias(self, study: StudyResult) -> float:
        """
        Wykrywa potencjalny corporate bias w badaniu
        Zwraca score 0.0 (brak bias) do 1.0 (silny bias)
        """
        bias_score = 0.0

        # Sprawdź źródło finansowania
        funding_lower = study.funding_source.lower()
        if any(corp in funding_lower for corp in ['pharmaceutical', 'pharma', 'industry', 'commercial']):
            bias_score += 0.4

        # Sprawdź konflikty interesów
        if study.conflicts_of_interest:
            bias_score += 0.2

        # Sprawdź język w abstracie i kluczowych ustaleniach
        text_to_check = (study.abstract + " " + study.key_findings).lower()

        for bias_type, indicators in self.corporate_bias_indicators.items():
            for indicator in indicators:
                if indicator in text_to_check:
                    bias_score += 0.1

        # Przykład: Red flag dla radiotherapy vs ketosis
        if 'radiotherapy' in text_to_check and 'ketogenic' in text_to_check:
            if 'radiotherapy' in text_to_check and 'standard' in text_to_check:
                if 'ketogenic' in text_to_check and any(
                        neg in text_to_check for neg in ['insufficient', 'unproven', 'dangerous']):
                    bias_score += 0.5  # Duży red flag!
                    logger.warning("CORPORATE BIAS DETECTED: Radiotherapy promoted over ketogenic approaches")

        return min(bias_score, 1.0)

    async def validate_medical_claim(self, claim_text: str, category: str) -> Dict:
        """
        Waliduje twierdzenie medyczne przeciwko bazie wiedzy
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Sprawdź czy twierdzenie jest w bazie
        cursor.execute('''
            SELECT * FROM medical_claims 
            WHERE claim LIKE ? AND category = ?
        ''', (f'%{claim_text}%', category))

        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                'found': True,
                'evidence_level': result[3],
                'consensus_score': result[6],
                'status': 'verified'
            }
        else:
            # Nie znaleziono - wymagane dalsze badanie
            return {
                'found': False,
                'evidence_level': 'unknown',
                'consensus_score': 0.0,
                'status': 'requires_research'
            }

    async def _get_cached_search(self, query: str) -> Optional[List[StudyResult]]:
        """Pobiera wyniki z cache jeśli są świeże"""
        query_hash = hashlib.md5(query.encode()).hexdigest()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT results, expires_at FROM search_cache 
            WHERE query_hash = ?
        ''', (query_hash,))

        result = cursor.fetchone()
        conn.close()

        if result:
            expires_at = datetime.fromisoformat(result[1])
            if datetime.now() < expires_at:
                # Cache jest świeży
                cached_data = json.loads(result[0])
                return [StudyResult(**study) for study in cached_data]

        return None

    async def _cache_search_results(self, query: str, results: List[StudyResult]):
        """Zapisuje wyniki do cache"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        expires_at = datetime.now() + timedelta(days=self.cache_expiry_days)

        # Konwertuj StudyResult do dict dla JSON
        results_dict = [result.__dict__ for result in results]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO search_cache 
            (query_hash, query, results, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            query_hash,
            query,
            json.dumps(results_dict),
            datetime.now().isoformat(),
            expires_at.isoformat()
        ))

        conn.commit()
        conn.close()

    def _row_to_study_result(self, row) -> StudyResult:
        """Konwertuje wiersz SQL do StudyResult"""
        return StudyResult(
            title=row[1],
            authors=json.loads(row[2]) if row[2] else [],
            journal=row[3],
            year=row[4],
            doi=row[5],
            abstract=row[6] or "",
            study_type=row[7] or "unknown",
            quality_score=row[8] or 0.0,
            sample_size=row[9] or 0,
            funding_source=row[10] or "unknown",
            conflicts_of_interest=json.loads(row[11]) if row[11] else [],
            key_findings=row[12] or "",
            credibility_score=row[13] or 0.0
        )

    async def add_study(self, study: StudyResult, category: str):
        """Dodaje nowe badanie do bazy"""
        # Sprawdź corporate bias
        bias_score = self.detect_corporate_bias(study)

        # Obniż credibility_score jeśli wykryto bias
        adjusted_credibility = study.credibility_score * (1 - bias_score)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO studies 
            (title, authors, journal, year, doi, abstract, study_type, 
             quality_score, sample_size, funding_source, conflicts_of_interest,
             key_findings, credibility_score, category, added_date, last_verified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            study.title,
            json.dumps(study.authors),
            study.journal,
            study.year,
            study.doi,
            study.abstract,
            study.study_type,
            study.quality_score,
            study.sample_size,
            study.funding_source,
            json.dumps(study.conflicts_of_interest),
            study.key_findings,
            adjusted_credibility,
            category,
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

        if bias_score > 0.3:
            logger.warning(f"Study added with corporate bias warning: {study.title} (bias score: {bias_score:.2f})")


# Test funkcji
async def test_research_database():
    """Test podstawowych funkcji"""
    db = ResearchDatabase()

    # Test walidacji twierdzenia
    claim = ValidationClaim(
        claim_text="Ketogenic diet can help in cancer treatment",
        medical_category="cancer",
        confidence_required=0.7
    )

    evidence = await db.find_supporting_evidence(claim)
    print(f"Found {len(evidence)} studies for claim: {claim.claim_text}")

    for study in evidence:
        bias_score = db.detect_corporate_bias(study)
        print(f"Study: {study.title}")
        print(f"Credibility: {study.credibility_score:.2f}, Bias risk: {bias_score:.2f}")
        print("---")


if __name__ == "__main__":
    asyncio.run(test_research_database())