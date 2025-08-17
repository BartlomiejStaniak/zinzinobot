#!/usr/bin/.env python3
"""
pubmed_scraper.py - Scraper dla PubMed/badań naukowych
Plik: scientific_validation/pubmed_scraper.py
"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PubMedArticle:
    """Artykuł z PubMed"""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    publication_date: str
    doi: Optional[str]
    keywords: List[str]
    mesh_terms: List[str]
    publication_types: List[str]
    funding_sources: List[str]
    conflicts_of_interest: Optional[str]

    def to_study_result(self, quality_score: float = 0.7):
        """Konwertuje na StudyResult dla research_database"""
        from scientific_validation.research_database import StudyResult

        # Określ typ badania
        study_type = 'unknown'
        if 'Systematic Review' in self.publication_types:
            study_type = 'systematic_review'
        elif 'Meta-Analysis' in self.publication_types:
            study_type = 'meta_analysis'
        elif 'Randomized Controlled Trial' in self.publication_types:
            study_type = 'rct'
        elif 'Clinical Trial' in self.publication_types:
            study_type = 'clinical_trial'

        # Ekstraktuj sample size z abstraktu (uproszczone)
        import re
        sample_match = re.search(r'n\s*=\s*(\d+)', self.abstract.lower())
        sample_size = int(sample_match.group(1)) if sample_match else 0

        return StudyResult(
            title=self.title,
            authors=self.authors[:5],  # Max 5 autorów
            journal=self.journal,
            year=int(self.publication_date[:4]) if self.publication_date else datetime.now().year,
            doi=self.doi or '',
            abstract=self.abstract,
            study_type=study_type,
            quality_score=quality_score,
            sample_size=sample_size,
            funding_source='; '.join(self.funding_sources) if self.funding_sources else 'Not disclosed',
            conflicts_of_interest=[self.conflicts_of_interest] if self.conflicts_of_interest else [],
            key_findings=self._extract_key_findings(),
            credibility_score=quality_score
        )

    def _extract_key_findings(self) -> str:
        """Ekstraktuje kluczowe ustalenia z abstraktu"""
        # Szukaj sekcji Conclusions/Results
        import re

        conclusions_match = re.search(
            r'(conclusions?|results?|findings?):\s*(.+?)(?:methods?:|background:|$)',
            self.abstract.lower(),
            re.IGNORECASE | re.DOTALL
        )

        if conclusions_match:
            return conclusions_match.group(2).strip()[:500]
        else:
            # Weź ostatnie 200 znaków abstraktu
            return self.abstract[-200:] if len(self.abstract) > 200 else self.abstract


class PubMedScraper:
    """
    Scraper dla PubMed z obsługą multi-platform keywords
    """

    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.api_key = None  # Opcjonalnie można dodać API key

        # Rate limiting
        self.requests_per_second = 3 if not self.api_key else 10
        self.last_request_time = 0

        # Platform-specific search modifications
        self.platform_keywords = {
            'facebook': {
                'audience': 'general population',
                'focus': ['practical applications', 'lifestyle interventions']
            },
            'instagram': {
                'audience': 'young adults',
                'focus': ['visual health outcomes', 'wellness trends']
            },
            'tiktok': {
                'audience': 'adolescents young adults',
                'focus': ['rapid interventions', 'trending health topics']
            }
        }

        # Zinzino-specific keywords
        self.zinzino_keywords = [
            'omega-3 fatty acids',
            'omega-6 omega-3 ratio',
            'fish oil supplementation',
            'polyphenols olive oil',
            'metabolic health biomarkers',
            'inflammation markers nutrition'
        ]

    async def search_articles(self, query: str, max_results: int = 20,
                              platform: Optional[str] = None,
                              include_zinzino: bool = True) -> List[PubMedArticle]:
        """
        Wyszukuje artykuły w PubMed z opcjonalnym dostosowaniem do platformy
        """
        # Modyfikuj query dla platformy
        if platform and platform in self.platform_keywords:
            platform_data = self.platform_keywords[platform]
            query = f"{query} AND ({' OR '.join(platform_data['focus'])})"

        # Dodaj Zinzino keywords jeśli potrzeba
        if include_zinzino:
            zinzino_query = ' OR '.join(f'("{kw}")' for kw in self.zinzino_keywords[:3])
            query = f"({query}) AND ({zinzino_query})"

        # Wyszukaj PMIDs
        pmids = await self._search_pmids(query, max_results)

        if not pmids:
            logger.info(f"No articles found for query: {query}")
            return []

        # Pobierz szczegóły artykułów
        articles = await self._fetch_articles(pmids)

        # Filtruj według jakości
        if platform:
            articles = self._filter_for_platform(articles, platform)

        return articles

    async def _search_pmids(self, query: str, max_results: int) -> List[str]:
        """Wyszukuje PMIDs dla zapytania"""
        await self._rate_limit()

        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance',
            'datetype': 'pdat',
            'mindate': '2018',  # Tylko ostatnie 5 lat
            'maxdate': '2024'
        }

        if self.api_key:
            params['api_key'] = self.api_key

        async with aiohttp.ClientSession() as session:
            async with session.get(
                    f"{self.base_url}/esearch.fcgi",
                    params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('esearchresult', {}).get('idlist', [])
                else:
                    logger.error(f"PubMed search error: {response.status}")
                    return []

    async def _fetch_articles(self, pmids: List[str]) -> List[PubMedArticle]:
        """Pobiera szczegóły artykułów po PMID"""
        articles = []

        # Podziel na batche (max 200 per request)
        batch_size = 200
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            await self._rate_limit()

            params = {
                'db': 'pubmed',
                'id': ','.join(batch),
                'retmode': 'xml'
            }

            if self.api_key:
                params['api_key'] = self.api_key

            async with aiohttp.ClientSession() as session:
                async with session.get(
                        f"{self.base_url}/efetch.fcgi",
                        params=params
                ) as response:
                    if response.status == 200:
                        xml_data = await response.text()
                        batch_articles = self._parse_xml_articles(xml_data)
                        articles.extend(batch_articles)
                    else:
                        logger.error(f"PubMed fetch error: {response.status}")

        return articles

    def _parse_xml_articles(self, xml_data: str) -> List[PubMedArticle]:
        """Parsuje XML z artykułami"""
        articles = []

        try:
            root = ET.fromstring(xml_data)

            for article_elem in root.findall('.//PubmedArticle'):
                article = self._parse_single_article(article_elem)
                if article:
                    articles.append(article)

        except ET.ParseError as e:
            logger.error(f"XML parse error: {str(e)}")

        return articles

    def _parse_single_article(self, article_elem) -> Optional[PubMedArticle]:
        """Parsuje pojedynczy artykuł z XML"""
        try:
            # PMID
            pmid = article_elem.find('.//PMID').text

            # Title
            title_elem = article_elem.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else 'No title'

            # Abstract
            abstract_parts = []
            for abstract_text in article_elem.findall('.//AbstractText'):
                label = abstract_text.get('Label', '')
                text = abstract_text.text or ''
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = ' '.join(abstract_parts)

            # Authors
            authors = []
            for author in article_elem.findall('.//Author'):
                last_name = author.find('LastName')
                fore_name = author.find('ForeName')
                if last_name is not None and fore_name is not None:
                    authors.append(f"{fore_name.text} {last_name.text}")

            # Journal
            journal_elem = article_elem.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else 'Unknown Journal'

            # Publication date
            pub_date_elem = article_elem.find('.//PubDate')
            if pub_date_elem is not None:
                year = pub_date_elem.find('Year')
                month = pub_date_elem.find('Month')
                day = pub_date_elem.find('Day')

                date_parts = []
                if year is not None:
                    date_parts.append(year.text)
                if month is not None:
                    date_parts.append(month.text)
                if day is not None:
                    date_parts.append(day.text)

                publication_date = '-'.join(date_parts)
            else:
                publication_date = 'Unknown'

            # DOI
            doi = None
            for id_elem in article_elem.findall('.//ArticleId'):
                if id_elem.get('IdType') == 'doi':
                    doi = id_elem.text
                    break

            # Keywords
            keywords = [kw.text for kw in article_elem.findall('.//Keyword') if kw.text]

            # MeSH terms
            mesh_terms = []
            for mesh in article_elem.findall('.//MeshHeading/DescriptorName'):
                if mesh.text:
                    mesh_terms.append(mesh.text)

            # Publication types
            pub_types = [pt.text for pt in article_elem.findall('.//PublicationType') if pt.text]

            # Funding (jeśli dostępne)
            funding_sources = []
            for grant in article_elem.findall('.//Grant'):
                agency = grant.find('Agency')
                if agency is not None and agency.text:
                    funding_sources.append(agency.text)

            # Conflicts of interest (szukaj w abstrakcie)
            coi = None
            if 'conflict' in abstract.lower() and 'interest' in abstract.lower():
                # Ekstraktuj sekcję COI
                import re
                coi_match = re.search(
                    r'conflict.{0,10}interest[s]?:?\s*(.+?)(?:\.|$)',
                    abstract,
                    re.IGNORECASE
                )
                if coi_match:
                    coi = coi_match.group(1).strip()

            return PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                publication_date=publication_date,
                doi=doi,
                keywords=keywords,
                mesh_terms=mesh_terms,
                publication_types=pub_types,
                funding_sources=funding_sources,
                conflicts_of_interest=coi
            )

        except Exception as e:
            logger.error(f"Error parsing article: {str(e)}")
            return None

    def _filter_for_platform(self, articles: List[PubMedArticle],
                             platform: str) -> List[PubMedArticle]:
        """Filtruje artykuły odpowiednie dla platformy"""
        if platform not in self.platform_keywords:
            return articles

        platform_data = self.platform_keywords[platform]
        filtered = []

        for article in articles:
            score = 0

            # Sprawdź focus keywords w abstrakcie
            abstract_lower = article.abstract.lower()
            for focus_kw in platform_data['focus']:
                if focus_kw.lower() in abstract_lower:
                    score += 1

            # Preferuj nowsze badania dla platform social media
            if platform in ['instagram', 'tiktok']:
                try:
                    year = int(article.publication_date[:4])
                    if year >= 2020:
                        score += 1
                except:
                    pass

            # Dodaj jeśli ma wystarczający score
            if score > 0:
                filtered.append(article)

        # Sortuj według relevance
        return sorted(filtered, key=lambda a: len(a.keywords), reverse=True)

    async def _rate_limit(self):
        """Rate limiting dla API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.requests_per_second

        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)

        self.last_request_time = time.time()

    async def search_zinzino_specific(self, max_results: int = 50) -> List[PubMedArticle]:
        """Wyszukuje badania specyficzne dla Zinzino"""
        queries = [
            'omega-3 fatty acids AND (balance OR ratio) AND supplementation',
            'fish oil AND polyphenols AND olive oil',
            'omega-6 omega-3 ratio AND inflammation',
            'metabolic health AND omega-3 supplementation',
            'DHA EPA supplementation AND cognitive function'
        ]

        all_articles = []
        for query in queries:
            articles = await self.search_articles(
                query,
                max_results=max_results // len(queries),
                include_zinzino=False  # Już mamy specific queries
            )
            all_articles.extend(articles)

        # Usuń duplikaty
        seen_pmids = set()
        unique_articles = []
        for article in all_articles:
            if article.pmid not in seen_pmids:
                seen_pmids.add(article.pmid)
                unique_articles.append(article)

        return unique_articles

    async def get_trending_health_topics(self, platform: str) -> List[Dict[str, Any]]:
        """Pobiera trending topics dla platformy"""
        trending_queries = {
            'facebook': [
                'mental health workplace',
                'nutrition immune system',
                'stress management techniques'
            ],
            'instagram': [
                'gut health microbiome',
                'intermittent fasting benefits',
                'adaptogens stress'
            ],
            'tiktok': [
                'biohacking performance',
                'cold therapy benefits',
                'breathwork anxiety'
            ]
        }

        topics = []
        queries = trending_queries.get(platform, trending_queries['facebook'])

        for query in queries:
            articles = await self.search_articles(
                query,
                max_results=5,
                platform=platform
            )

            if articles:
                topics.append({
                    'topic': query,
                    'platform': platform,
                    'article_count': len(articles),
                    'top_findings': articles[0].title if articles else '',
                    'relevance_score': len(articles) / 5.0
                })

        return sorted(topics, key=lambda x: x['relevance_score'], reverse=True)


# Test funkcji
async def test_pubmed_scraper():
    """Test PubMed scraper"""
    scraper = PubMedScraper()

    print("🔬 Testing PubMed Scraper...")
    print("=" * 50)

    # Test 1: Wyszukiwanie ogólne
    print("\n📚 Searching for omega-3 studies...")
    articles = await scraper.search_articles(
        "omega-3 fatty acids mental health",
        max_results=5
    )

    print(f"Found {len(articles)} articles")
    for article in articles[:3]:
        print(f"\nTitle: {article.title}")
        print(f"Authors: {', '.join(article.authors[:3])}")
        print(f"Journal: {article.journal} ({article.publication_date})")
        print(f"DOI: {article.doi}")
        print(f"Study type: {article.publication_types}")

    # Test 2: Platform-specific search
    print("\n📱 Testing platform-specific search (Instagram)...")
    insta_articles = await scraper.search_articles(
        "wellness nutrition",
        max_results=3,
        platform='instagram'
    )

    print(f"Found {len(insta_articles)} Instagram-appropriate articles")

    # Test 3: Trending topics
    print("\n🔥 Getting trending topics for TikTok...")
    trends = await scraper.get_trending_health_topics('tiktok')

    for trend in trends:
        print(f"Topic: {trend['topic']} (relevance: {trend['relevance_score']:.2f})")


if __name__ == "__main__":
    asyncio.run(test_pubmed_scraper())