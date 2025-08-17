#!/usr/bin/.env python3
"""
credibility_scorer.py - Ocena wiarygodności źródeł medycznych
Plik: scientific_validation/credibility_scorer.py
"""

import re
import json
import asyncio
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Typy źródeł informacji"""
    PEER_REVIEWED_JOURNAL = "peer_reviewed"
    PREPRINT = "preprint"
    GOVERNMENT_AGENCY = "government"
    MEDICAL_INSTITUTION = "medical_institution"
    UNIVERSITY = "university"
    PHARMACEUTICAL_COMPANY = "pharmaceutical"
    HEALTH_WEBSITE = "health_website"
    BLOG = "blog"
    SOCIAL_MEDIA = "social_media"
    NEWS_MEDIA = "news_media"
    UNKNOWN = "unknown"


class FundingType(Enum):
    """Typy finansowania badań"""
    GOVERNMENT = "government"
    UNIVERSITY = "university"
    NON_PROFIT = "non_profit"
    PHARMACEUTICAL = "pharmaceutical"
    MEDICAL_DEVICE = "medical_device"
    FOOD_INDUSTRY = "food_industry"
    SUPPLEMENT_INDUSTRY = "supplement_industry"
    INDEPENDENT = "independent"
    UNKNOWN = "unknown"


@dataclass
class CredibilityScore:
    """Wynik oceny wiarygodności"""
    overall_score: float  # 0.0 - 1.0
    source_type_score: float
    funding_score: float
    journal_impact_score: float
    methodology_score: float
    bias_penalty: float
    transparency_score: float
    independence_score: float
    explanation: str
    red_flags: List[str]
    green_flags: List[str]


@dataclass
class SourceInfo:
    """Informacje o źródle"""
    title: str
    authors: List[str]
    journal: str
    year: int
    doi: str
    funding_sources: List[str]
    conflicts_of_interest: List[str]
    methodology: str
    sample_size: int
    study_duration: Optional[str]
    url: Optional[str]


class CredibilityScorer:
    """
    Główna klasa do oceny wiarygodności źródeł medycznych
    """

    def __init__(self):
        # Ranking czasopism medycznych (uproszczony)
        self.journal_rankings = {
            # Top tier journals
            'nature': 0.95,
            'science': 0.95,
            'new england journal of medicine': 0.95,
            'lancet': 0.94,
            'jama': 0.93,
            'british medical journal': 0.92,
            'plos medicine': 0.88,

            # Second tier
            'journal of clinical oncology': 0.85,
            'clinical cancer research': 0.82,
            'nutrition & metabolism': 0.75,
            'american journal of clinical nutrition': 0.80,

            # Lower tier but still credible
            'nutrients': 0.70,
            'frontiers in oncology': 0.65,
            'alternative medicine review': 0.60,

            # Predatory/questionable journals
            'international journal of cancer research': 0.30,
            'global journal of health science': 0.25
        }

        # Wiarygodne instytucje medyczne
        self.credible_institutions = {
            # Government agencies
            'nih': 0.95,
            'national institutes of health': 0.95,
            'cdc': 0.93,
            'fda': 0.90,
            'who': 0.88,
            'world health organization': 0.88,

            # Medical institutions
            'mayo clinic': 0.92,
            'cleveland clinic': 0.90,
            'johns hopkins': 0.95,
            'harvard medical school': 0.93,
            'stanford medicine': 0.92,

            # Research institutions
            'dana-farber cancer institute': 0.90,
            'memorial sloan kettering': 0.89,

            # International
            'cochrane collaboration': 0.96,
            'european medicines agency': 0.85
        }

        # Red flags dla corporate bias (rozszerzone o przykład radioterapia vs ketoza)
        self.bias_red_flags = {
            'pharmaceutical_funding': [
                'funded by pfizer', 'sponsored by novartis', 'merck research grant',
                'pharmaceutical company funding', 'industry sponsored study',
                'commercial funding source'
            ],
            'conflict_of_interest': [
                'consultant for pharmaceutical company',
                'board member of', 'stock options in',
                'patent holder', 'licensing agreement',
                'speaker fees from'
            ],
            'methodology_issues': [
                'small sample size', 'short duration study',
                'no control group', 'retrospective analysis only',
                'cherry-picked data', 'selective reporting'
            ],
            'treatment_bias_language': [
                'gold standard treatment while alternatives unproven',
                'established therapy versus experimental approaches',
                'standard of care superior to dietary interventions',
                'proven medical treatment unlike lifestyle changes'
            ],
            # Specyficzne dla przykładu radioterapia vs ketoza
            'radiotherapy_ketosis_bias': [
                'radiotherapy proven effective while ketogenic diet unvalidated',
                'radiation therapy standard care unlike metabolic approaches',
                'established oncology protocols superior to dietary modifications',
                'medical intervention necessary over nutritional therapy'
            ]
        }

        # Green flags (pozytywne wskaźniki)
        self.credibility_green_flags = {
            'independent_funding': [
                'government funded', 'university grant', 'non-profit foundation',
                'independent research', 'no conflicts of interest declared'
            ],
            'robust_methodology': [
                'randomized controlled trial', 'double-blind study',
                'large sample size', 'multi-center study',
                'systematic review', 'meta-analysis'
            ],
            'transparency': [
                'data publicly available', 'protocol pre-registered',
                'full methodology disclosed', 'raw data accessible',
                'replication study', 'peer reviewed'
            ],
            'balanced_approach': [
                'compared multiple treatment options',
                'discussed limitations', 'acknowledged uncertainties',
                'considered alternative approaches',
                'mentioned both benefits and risks'
            ]
        }

        # Scoring weights
        self.scoring_weights = {
            'source_type': 0.25,
            'funding': 0.20,
            'journal_impact': 0.20,
            'methodology': 0.15,
            'transparency': 0.10,
            'independence': 0.10
        }

    async def score_source_credibility(self, source_info: SourceInfo) -> CredibilityScore:
        """
        Główna metoda oceny wiarygodności źródła
        """
        logger.info(f"Scoring credibility for: {source_info.title[:50]}...")

        # 1. Oceń typ źródła
        source_type_score = self._score_source_type(source_info)

        # 2. Oceń finansowanie
        funding_score = self._score_funding(source_info.funding_sources)

        # 3. Oceń prestiż czasopisma
        journal_score = self._score_journal_impact(source_info.journal)

        # 4. Oceń metodologię
        methodology_score = self._score_methodology(source_info)

        # 5. Oceń transparentność
        transparency_score = self._score_transparency(source_info)

        # 6. Oceń niezależność
        independence_score = self._score_independence(source_info)

        # 7. Wykryj bias i oblicz karę
        bias_penalty = await self._detect_bias_penalty(source_info)

        # 8. Oblicz ogólny wynik
        weighted_score = (
                source_type_score * self.scoring_weights['source_type'] +
                funding_score * self.scoring_weights['funding'] +
                journal_score * self.scoring_weights['journal_impact'] +
                methodology_score * self.scoring_weights['methodology'] +
                transparency_score * self.scoring_weights['transparency'] +
                independence_score * self.scoring_weights['independence']
        )

        overall_score = max(0.0, weighted_score - bias_penalty)

        # 9. Generuj wyjaśnienie i flagi
        explanation = self._generate_explanation(source_info, overall_score)
        red_flags = await self._identify_red_flags(source_info)
        green_flags = self._identify_green_flags(source_info)

        return CredibilityScore(
            overall_score=overall_score,
            source_type_score=source_type_score,
            funding_score=funding_score,
            journal_impact_score=journal_score,
            methodology_score=methodology_score,
            bias_penalty=bias_penalty,
            transparency_score=transparency_score,
            independence_score=independence_score,
            explanation=explanation,
            red_flags=red_flags,
            green_flags=green_flags
        )

    def _score_source_type(self, source_info: SourceInfo) -> float:
        """Ocenia typ źródła"""
        # Określ typ na podstawie czasopisma/URL
        source_type = self._determine_source_type(source_info)

        type_scores = {
            SourceType.PEER_REVIEWED_JOURNAL: 0.95,
            SourceType.GOVERNMENT_AGENCY: 0.90,
            SourceType.MEDICAL_INSTITUTION: 0.85,
            SourceType.UNIVERSITY: 0.80,
            SourceType.PREPRINT: 0.60,
            SourceType.HEALTH_WEBSITE: 0.50,
            SourceType.NEWS_MEDIA: 0.30,
            SourceType.BLOG: 0.20,
            SourceType.SOCIAL_MEDIA: 0.10,
            SourceType.PHARMACEUTICAL_COMPANY: 0.25,  # Niski ze względu na bias
            SourceType.UNKNOWN: 0.40
        }

        return type_scores.get(source_type, 0.40)

    def _score_funding(self, funding_sources: List[str]) -> float:
        """Ocenia źródła finansowania"""
        if not funding_sources:
            return 0.50  # Nieznane finansowanie

        total_score = 0.0

        for funding in funding_sources:
            funding_lower = funding.lower()

            if any(term in funding_lower for term in ['government', 'nih', 'nsf', 'public']):
                total_score += 0.90
            elif any(term in funding_lower for term in ['university', 'academic', 'educational']):
                total_score += 0.85
            elif any(term in funding_lower for term in ['non-profit', 'foundation', 'charity']):
                total_score += 0.80
            elif any(term in funding_lower for term in ['pharmaceutical', 'pharma', 'drug company']):
                total_score += 0.20  # Duża kara za pharma funding
            elif any(term in funding_lower for term in ['medical device', 'equipment manufacturer']):
                total_score += 0.30
            elif any(term in funding_lower for term in ['food industry', 'supplement company']):
                total_score += 0.40
            else:
                total_score += 0.60  # Neutralne/nieznane

        return min(1.0, total_score / len(funding_sources))

    def _score_journal_impact(self, journal: str) -> float:
        """Ocenia prestiż czasopisma"""
        journal_lower = journal.lower()

        # Sprawdź bezpośrednie dopasowanie
        if journal_lower in self.journal_rankings:
            return self.journal_rankings[journal_lower]

        # Sprawdź częściowe dopasowania
        for journal_name, score in self.journal_rankings.items():
            if journal_name in journal_lower or journal_lower in journal_name:
                return score

        # Heurystyki dla nieznanych czasopism
        if 'international journal' in journal_lower:
            return 0.40  # Często predatory
        elif 'nature' in journal_lower or 'science' in journal_lower:
            return 0.85  # Prawdopodobnie prestiżowe
        elif any(term in journal_lower for term in ['medicine', 'medical', 'clinical']):
            return 0.60  # Medyczne, ale nieznane

        return 0.50  # Default dla nieznanych

    def _score_methodology(self, source_info: SourceInfo) -> float:
        """Ocenia jakość metodologii"""
        methodology_lower = source_info.methodology.lower()
        score = 0.50  # Bazowy wynik

        # Bonusy za dobrą metodologię
        if 'randomized controlled trial' in methodology_lower or 'rct' in methodology_lower:
            score += 0.30
        elif 'systematic review' in methodology_lower:
            score += 0.35
        elif 'meta-analysis' in methodology_lower:
            score += 0.40
        elif 'double-blind' in methodology_lower:
            score += 0.25
        elif 'prospective' in methodology_lower:
            score += 0.15

        # Kary za słabą metodologię
        if 'retrospective' in methodology_lower and 'prospective' not in methodology_lower:
            score -= 0.15
        if 'case report' in methodology_lower or 'case study' in methodology_lower:
            score -= 0.25
        if source_info.sample_size < 50:
            score -= 0.20
        elif source_info.sample_size < 100:
            score -= 0.10

        # Bonus za duże próby
        if source_info.sample_size > 1000:
            score += 0.15
        elif source_info.sample_size > 500:
            score += 0.10

        return max(0.0, min(1.0, score))

    def _score_transparency(self, source_info: SourceInfo) -> float:
        """Ocenia transparentność badania"""
        score = 0.50

        # Bonus za transparentność
        if source_info.doi:
            score += 0.20  # DOI oznacza publikację w indeksowanym czasopiśmie

        if source_info.conflicts_of_interest:
            if any('no conflict' in coi.lower() for coi in source_info.conflicts_of_interest):
                score += 0.15
            else:
                score += 0.10  # Przynajmniej ujawnili konflikty

        if len(source_info.funding_sources) > 0:
            score += 0.10  # Ujawnili źródła finansowania

        if len(source_info.authors) > 1:
            score += 0.05  # Współpraca zwiększa wiarygodność

        return max(0.0, min(1.0, score))

    def _score_independence(self, source_info: SourceInfo) -> float:
        """Ocenia niezależność badania"""
        score = 0.50

        # Sprawdź czy autorzy są niezależni
        author_affiliations = ' '.join(source_info.authors).lower()

        # Kary za powiązania z przemysłem
        if any(term in author_affiliations for term in ['pharmaceutical', 'pharma', 'drug company']):
            score -= 0.30

        if any(term in author_affiliations for term in ['medical device', 'biotech']):
            score -= 0.20

        # Bonusy za niezależne afiliacje
        if any(term in author_affiliations for term in ['university', 'hospital', 'medical center']):
            score += 0.20

        if any(term in author_affiliations for term in ['government', 'public health']):
            score += 0.25

        # Sprawdź konflikty interesów
        if source_info.conflicts_of_interest:
            coi_text = ' '.join(source_info.conflicts_of_interest).lower()
            if 'no conflict' in coi_text or 'no competing interest' in coi_text:
                score += 0.15
            else:
                # Są konflikty - oceń ich wagę
                if any(term in coi_text for term in ['consultant', 'advisory board', 'speaker fees']):
                    score -= 0.25
                if any(term in coi_text for term in ['stock', 'equity', 'patent']):
                    score -= 0.35

        return max(0.0, min(1.0, score))

    async def _detect_bias_penalty(self, source_info: SourceInfo) -> float:
        """Wykrywa bias i oblicza karę"""
        penalty = 0.0
        full_text = (
                source_info.title + " " +
                ' '.join(source_info.authors) + " " +
                source_info.methodology + " " +
                ' '.join(source_info.funding_sources) + " " +
                ' '.join(source_info.conflicts_of_interest)
        ).lower()

        # Sprawdź każdą kategorię red flags
        for bias_type, indicators in self.bias_red_flags.items():
            for indicator in indicators:
                if indicator.lower() in full_text:
                    if bias_type == 'pharmaceutical_funding':
                        penalty += 0.25
                        logger.warning(f"Pharmaceutical funding detected: {indicator}")
                    elif bias_type == 'conflict_of_interest':
                        penalty += 0.20
                        logger.warning(f"Conflict of interest detected: {indicator}")
                    elif bias_type == 'methodology_issues':
                        penalty += 0.15
                        logger.warning(f"Methodology issue detected: {indicator}")
                    elif bias_type == 'treatment_bias_language':
                        penalty += 0.30
                        logger.warning(f"Treatment bias language detected: {indicator}")
                    elif bias_type == 'radiotherapy_ketosis_bias':
                        penalty += 0.40  # Duża kara za ten specyficzny bias
                        logger.warning(f"RADIOTHERAPY vs KETOSIS BIAS DETECTED: {indicator}")

        # Specjalna detekcja dla przykładu radioterapia vs ketoza
        radiotherapy_terms = ['radiotherapy', 'radiation therapy', 'radioactive treatment']
        ketosis_terms = ['ketogenic', 'ketosis', 'metabolic therapy', 'fasting therapy']

        has_radiotherapy = any(term in full_text for term in radiotherapy_terms)
        has_ketosis = any(term in full_text for term in ketosis_terms)

        if has_radiotherapy and has_ketosis:
            # Sprawdź czy radiotherapy jest promowana kosztem ketosis
            positive_radio = any(
                term in full_text for term in ['effective radiotherapy', 'proven radiation', 'standard radiotherapy'])
            negative_keto = any(
                term in full_text for term in ['unproven ketogenic', 'dangerous ketosis', 'insufficient metabolic'])

            if positive_radio and negative_keto:
                penalty += 0.50
                logger.warning("MAJOR BIAS: Radiotherapy promoted while ketogenic approaches dismissed")

        return min(penalty, 0.8)  # Maksymalna kara 80%

    async def _identify_red_flags(self, source_info: SourceInfo) -> List[str]:
        """Identyfikuje red flags w źródle"""
        red_flags = []

        # Sprawdź finansowanie
        for funding in source_info.funding_sources:
            if any(term in funding.lower() for term in ['pharmaceutical', 'drug company', 'pharma']):
                red_flags.append(f"Finansowanie farmaceutyczne: {funding}")

        # Sprawdź konflikty interesów
        for coi in source_info.conflicts_of_interest:
            if 'no conflict' not in coi.lower():
                red_flags.append(f"Konflikt interesów: {coi}")

        # Sprawdź metodologię
        if source_info.sample_size < 50:
            red_flags.append(f"Mała próba badawcza: {source_info.sample_size} uczestników")

        # Sprawdź journal
        journal_score = self._score_journal_impact(source_info.journal)
        if journal_score < 0.4:
            red_flags.append(f"Czasopismo niskiej jakości: {source_info.journal}")

        # Sprawdź bias językowy w tytule
        title_lower = source_info.title.lower()
        if any(term in title_lower for term in self.bias_red_flags['treatment_bias_language']):
            red_flags.append("Język promujący konwencjonalne leczenie kosztem alternatyw")

        # Specjalny red flag dla radioterapia vs ketoza
        if ('radiotherapy' in title_lower or 'radiation' in title_lower) and \
                ('ketogenic' in title_lower or 'ketosis' in title_lower):
            methodology_lower = source_info.methodology.lower()
            if 'standard' in methodology_lower and 'unproven' in methodology_lower:
                red_flags.append("UWAGA: Możliwy bias - promowanie radioterapii kosztem ketogennej terapii")

        return red_flags

    def _identify_green_flags(self, source_info: SourceInfo) -> List[str]:
        """Identyfikuje pozytywne wskaźniki"""
        green_flags = []

        # Sprawdź niezależne finansowanie
        for funding in source_info.funding_sources:
            if any(term in funding.lower() for term in ['government', 'university', 'non-profit']):
                green_flags.append(f"Niezależne finansowanie: {funding}")

        # Sprawdź brak konfliktów
        if any('no conflict' in coi.lower() for coi in source_info.conflicts_of_interest):
            green_flags.append("Brak konfliktów interesów")

        # Sprawdź wysoką jakość metodologii
        methodology_lower = source_info.methodology.lower()
        if 'randomized controlled trial' in methodology_lower:
            green_flags.append("Randomized Controlled Trial - najwyższa jakość dowodów")
        elif 'systematic review' in methodology_lower:
            green_flags.append("Przegląd systematyczny - silne dowody")
        elif 'meta-analysis' in methodology_lower:
            green_flags.append("Meta-analiza - bardzo silne dowody")

        # Sprawdź dużą próbę
        if source_info.sample_size > 1000:
            green_flags.append(f"Duża próba badawcza: {source_info.sample_size} uczestników")

        # Sprawdź prestiżowy journal
        journal_score = self._score_journal_impact(source_info.journal)
        if journal_score > 0.85:
            green_flags.append(f"Prestiżowy journal: {source_info.journal}")

        # Sprawdź DOI
        if source_info.doi:
            green_flags.append("Publikacja z DOI - łatwa weryfikacja")

        return green_flags

    def _generate_explanation(self, source_info: SourceInfo, overall_score: float) -> str:
        """Generuje wyjaśnienie oceny wiarygodności"""
        explanation = f"OCENA WIARYGODNOŚCI: {overall_score:.2f}/1.00\n\n"

        explanation += f"Źródło: {source_info.title}\n"
        explanation += f"Journal: {source_info.journal} ({source_info.year})\n"
        explanation += f"Autorzy: {len(source_info.authors)} autorów\n"
        explanation += f"Próba: {source_info.sample_size} uczestników\n\n"

        # Interpretacja wyniku
        if overall_score >= 0.85:
            explanation += "⭐ BARDZO WYSOKA WIARYGODNOŚĆ - można w pełni ufać tym informacjom\n"
        elif overall_score >= 0.70:
            explanation += "✅ WYSOKA WIARYGODNOŚĆ - wiarygodne źródło informacji\n"
        elif overall_score >= 0.55:
            explanation += "⚠️ ŚREDNIA WIARYGODNOŚĆ - wymagana ostrożność w interpretacji\n"
        elif overall_score >= 0.40:
            explanation += "❌ NISKA WIARYGODNOŚĆ - wysokie ryzyko błędnych informacji\n"
        else:
            explanation += "🚫 BARDZO NISKA WIARYGODNOŚĆ - nie polecane jako źródło\n"

        # Główne czynniki wpływające na ocenę
        explanation += "\nGłówne czynniki:\n"

        journal_score = self._score_journal_impact(source_info.journal)
        funding_score = self._score_funding(source_info.funding_sources)
        methodology_score = self._score_methodology(source_info)

        explanation += f"• Prestiż czasopisma: {journal_score:.2f}\n"
        explanation += f"• Jakość finansowania: {funding_score:.2f}\n"
        explanation += f"• Metodologia: {methodology_score:.2f}\n"

        if funding_score < 0.5:
            explanation += "⚠️ UWAGA: Problematyczne źródła finansowania\n"

        if methodology_score < 0.5:
            explanation += "⚠️ UWAGA: Słaba metodologia badania\n"

        return explanation

    def _determine_source_type(self, source_info: SourceInfo) -> SourceType:
        """Określa typ źródła na podstawie dostępnych informacji"""
        journal_lower = source_info.journal.lower()

        # Sprawdź URL jeśli dostępny
        if source_info.url:
            domain = urlparse(source_info.url).netloc.lower()

            if any(gov in domain for gov in ['.gov', 'nih.gov', 'cdc.gov', 'fda.gov']):
                return SourceType.GOVERNMENT_AGENCY
            elif any(edu in domain for edu in ['.edu', 'university', 'college']):
                return SourceType.UNIVERSITY
            elif any(med in domain for med in ['mayo', 'cleveland', 'hopkins']):
                return SourceType.MEDICAL_INSTITUTION
            elif 'pharmaceutical' in domain or 'pharma' in domain:
                return SourceType.PHARMACEUTICAL_COMPANY

        # Sprawdź journal
        if any(term in journal_lower for term in ['journal', 'medicine', 'medical', 'clinical']):
            return SourceType.PEER_REVIEWED_JOURNAL
        elif 'preprint' in journal_lower or 'arxiv' in journal_lower:
            return SourceType.PREPRINT
        elif any(term in journal_lower for term in ['blog', 'website', 'news']):
            return SourceType.BLOG

        return SourceType.UNKNOWN

    async def batch_score_sources(self, sources: List[SourceInfo]) -> List[CredibilityScore]:
        """Ocenia wiarygodność wielu źródeł jednocześnie"""
        results = []

        for source in sources:
            score = await self.score_source_credibility(source)
            results.append(score)

        return results

    def rank_sources_by_credibility(self, scored_sources: List[Tuple[SourceInfo, CredibilityScore]]) -> List[
        Tuple[SourceInfo, CredibilityScore]]:
        """Sortuje źródła według wiarygodności"""
        return sorted(scored_sources, key=lambda x: x[1].overall_score, reverse=True)

    def filter_high_credibility_sources(self, scored_sources: List[Tuple[SourceInfo, CredibilityScore]],
                                        min_score: float = 0.7) -> List[Tuple[SourceInfo, CredibilityScore]]:
        """Filtruje tylko wysokiej jakości źródła"""
        return [(source, score) for source, score in scored_sources if score.overall_score >= min_score]


# Test funkcji
async def test_credibility_scorer():
    """Test oceny wiarygodności"""

    scorer = CredibilityScorer()

    # Test 1: Wysokiej jakości źródło
    high_quality_source = SourceInfo(
        title="Ketogenic Diet in Cancer Treatment: Systematic Review and Meta-Analysis",
        authors=["Dr. Smith A", "Dr. Johnson B", "Prof. Williams C"],
        journal="New England Journal of Medicine",
        year=2023,
        doi="10.1056/NEJMoa2023001",
        funding_sources=["National Institutes of Health", "University Research Grant"],
        conflicts_of_interest=["No conflicts of interest declared"],
        methodology="Systematic review and meta-analysis of 15 randomized controlled trials",
        sample_size=3500,
        study_duration="24 months",
        url="https://www.nejm.org/doi/full/10.1056/NEJMoa2023001"
    )

    # Test 2: Źródło z corporate bias (przykład radioterapia vs ketoza)
    biased_source = SourceInfo(
        title="Radiotherapy remains gold standard while ketogenic approaches unproven in cancer treatment",
        authors=["Dr. Pharma Inc", "Medical Device Corp"],
        journal="International Journal of Oncology Research",
        year=2023,
        doi="",
        funding_sources=["Radiotherapy Equipment Manufacturer Inc", "Pharmaceutical Research Ltd"],
        conflicts_of_interest=["Authors receive consulting fees from radiotherapy companies"],
        methodology="Retrospective analysis comparing standard radiotherapy to experimental dietary interventions",
        sample_size=45,
        study_duration="6 months",
        url="https://ijor-journal.com/article123"
    )

    print("Testing credibility scorer...")

    # Oceń oba źródła
    high_quality_score = await scorer.score_source_credibility(high_quality_source)
    biased_score = await scorer.score_source_credibility(biased_source)

    print(f"\n=== HIGH QUALITY SOURCE ===")
    print(f"Overall Score: {high_quality_score.overall_score:.2f}")
    print(f"Red Flags: {high_quality_score.red_flags}")
    print(f"Green Flags: {high_quality_score.green_flags}")
    print(f"Explanation:\n{high_quality_score.explanation}")

    print(f"\n=== BIASED SOURCE ===")
    print(f"Overall Score: {biased_score.overall_score:.2f}")
    print(f"Bias Penalty: {biased_score.bias_penalty:.2f}")
    print(f"Red Flags: {biased_score.red_flags}")
    print(f"Green Flags: {biased_score.green_flags}")
    print(f"Explanation:\n{biased_score.explanation}")


if __name__ == "__main__":
    asyncio.run(test_credibility_scorer())