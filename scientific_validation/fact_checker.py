#!/usr/bin/.env python3
"""
fact_checker.py - Sprawdzanie faktów medycznych
Plik: scientific_validation/fact_checker.py
"""

import re
import asyncio
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import spacy
from .research_database import ResearchDatabase, ValidationClaim

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClaimType(Enum):
    """Typy twierdzeń medycznych"""
    TREATMENT_EFFICACY = "treatment_efficacy"
    SIDE_EFFECTS = "side_effects"
    MECHANISM = "mechanism"
    PREVENTION = "prevention"
    DIAGNOSIS = "diagnosis"
    NUTRITION = "nutrition"
    LIFESTYLE = "lifestyle"


class EvidenceLevel(Enum):
    """Poziomy dowodów naukowych"""
    SYSTEMATIC_REVIEW = 1  # Najwyższy
    META_ANALYSIS = 2
    RANDOMIZED_TRIAL = 3
    COHORT_STUDY = 4
    CASE_CONTROL = 5
    CASE_SERIES = 6
    EXPERT_OPINION = 7  # Najniższy


@dataclass
class FactCheckResult:
    """Wynik sprawdzenia faktu"""
    claim: str
    is_supported: bool
    confidence_score: float  # 0.0 - 1.0
    evidence_level: EvidenceLevel
    supporting_studies: int
    contradicting_studies: int
    bias_detected: bool
    corporate_influence: float
    explanation: str
    warnings: List[str]
    recommendations: List[str]


@dataclass
class ExtractedClaim:
    """Wyciągnięte twierdzenie z tekstu"""
    text: str
    claim_type: ClaimType
    confidence: float
    medical_terms: List[str]
    context: str


class MedicalFactChecker:
    """
    Główna klasa do sprawdzania faktów medycznych
    """

    def __init__(self, research_db: ResearchDatabase):
        self.research_db = research_db

        # Wzorce dla różnych typów twierdzeń
        self.claim_patterns = {
            ClaimType.TREATMENT_EFFICACY: [
                r'(\w+)\s+(?:cures?|treats?|heals?|eliminates?)\s+(\w+)',
                r'(\w+)\s+(?:is effective|works|helps)\s+(?:for|against|with)\s+(\w+)',
                r'(\w+)\s+(?:therapy|treatment)\s+(?:improves?|reduces?)\s+(\w+)'
            ],
            ClaimType.SIDE_EFFECTS: [
                r'(\w+)\s+(?:causes?|leads? to|results? in)\s+(\w+)',
                r'(\w+)\s+(?:side effects?|adverse effects?)\s+(?:include|are)\s+(\w+)',
                r'(\w+)\s+(?:may cause|can cause)\s+(\w+)'
            ],
            ClaimType.MECHANISM: [
                r'(\w+)\s+(?:works by|functions by|acts by)\s+(\w+)',
                r'(\w+)\s+(?:targets?|affects?|influences?)\s+(\w+)',
                r'(\w+)\s+(?:mechanism|pathway|process)\s+(?:involves?|includes?)\s+(\w+)'
            ],
            ClaimType.NUTRITION: [
                r'(\w+)\s+(?:diet|nutrition|food)\s+(?:helps|improves|prevents)\s+(\w+)',
                r'(\w+)\s+(?:deficiency|supplementation)\s+(?:causes|leads to|prevents)\s+(\w+)',
                r'(?:eating|consuming)\s+(\w+)\s+(?:reduces|increases|affects)\s+(\w+)'
            ]
        }

        # Red flags dla corporate bias (rozszerzając przykład radioterapia vs ketoza)
        self.bias_indicators = {
            'pharmaceutical_bias': [
                'standard of care',
                'first-line treatment',
                'FDA approved',
                'clinically proven',
                'gold standard'
            ],
            'alternative_suppression': [
                'unproven therapy',
                'not scientifically validated',
                'insufficient evidence',
                'dangerous alternative',
                'no clinical trials'
            ],
            'profit_motivation': [
                'expensive treatment necessary',
                'long-term medication required',
                'regular monitoring needed',
                'specialist consultation required'
            ],
            'ketosis_suppression': [
                'ketogenic diet dangerous',
                'fasting not recommended',
                'dietary changes insufficient',
                'metabolic therapy unproven'
            ]
        }

        # Konflikt radioterapia vs ketoza (główny przykład)
        self.treatment_conflicts = {
            'radiotherapy_vs_ketosis': {
                'conventional': ['radiotherapy', 'radiation therapy', 'radioactive treatment'],
                'alternative': ['ketogenic diet', 'ketosis', 'metabolic therapy', 'fasting'],
                'bias_indicators': [
                    'radiotherapy is standard treatment while ketogenic approaches are unproven',
                    'radiation therapy effectiveness established unlike dietary interventions',
                    'metabolic therapies require more research before clinical implementation'
                ]
            }
        }

        # Inicjalizuj NLP (opcjonalnie - jeśli masz spaCy)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

    async def check_facts_in_text(self, text: str) -> List[FactCheckResult]:
        """
        Główna metoda - sprawdza wszystkie fakty w podanym tekście
        """
        logger.info(f"Fact-checking text: {text[:100]}...")

        # 1. Wyciągnij twierdzenia z tekstu
        extracted_claims = await self.extract_claims(text)

        # 2. Sprawdź każde twierdzenie
        fact_check_results = []
        for claim in extracted_claims:
            result = await self.verify_single_claim(claim)
            fact_check_results.append(result)

        # 3. Sprawdź konflikty interesów na poziomie całego tekstu
        corporate_bias = await self.detect_text_bias(text, fact_check_results)

        # 4. Dodaj ostrzeżenia o bias do wszystkich wyników
        for result in fact_check_results:
            if corporate_bias > 0.5:
                result.warnings.append("WYSOKIE RYZYKO CORPORATE BIAS W CAŁYM TEKŚCIE")
            result.corporate_influence = max(result.corporate_influence, corporate_bias)

        return fact_check_results

    async def extract_claims(self, text: str) -> List[ExtractedClaim]:
        """
        Wyciąga twierdzenia medyczne z tekstu
        """
        claims = []
        text_lower = text.lower()

        # Przeszukaj wzorce dla każdego typu twierdzenia
        for claim_type, patterns in self.claim_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    claim_text = match.group(0)

                    # Wyciągnij terminy medyczne
                    medical_terms = self._extract_medical_terms(claim_text)

                    # Oceń pewność na podstawie języka
                    confidence = self._assess_claim_confidence(claim_text)

                    claims.append(ExtractedClaim(
                        text=claim_text,
                        claim_type=claim_type,
                        confidence=confidence,
                        medical_terms=medical_terms,
                        context=self._get_surrounding_context(text, match.start(), match.end())
                    ))

        # Usuń duplikaty
        unique_claims = []
        seen_texts = set()
        for claim in claims:
            if claim.text not in seen_texts:
                unique_claims.append(claim)
                seen_texts.add(claim.text)

        return unique_claims

    async def verify_single_claim(self, claim: ExtractedClaim) -> FactCheckResult:
        """
        Weryfikuje pojedyncze twierdzenie
        """
        logger.info(f"Verifying claim: {claim.text}")

        # Określ kategorię medyczną
        category = self._determine_medical_category(claim.medical_terms)

        # Stwórz ValidationClaim dla research_database
        validation_claim = ValidationClaim(
            claim_text=claim.text,
            medical_category=category,
            confidence_required=0.7
        )

        # Znajdź dowody
        evidence = await self.research_db.find_supporting_evidence(validation_claim)

        # Oceń dowody
        supporting_studies = []
        contradicting_studies = []
        total_bias = 0.0

        for study in evidence:
            bias_score = self.research_db.detect_corporate_bias(study)
            total_bias += bias_score

            # Sprawdź czy badanie wspiera czy przeczy twierdzeniu
            if self._study_supports_claim(study, claim):
                supporting_studies.append(study)
            else:
                contradicting_studies.append(study)

        # Sprawdź czy to konflikt typu "radioterapia vs ketoza"
        conflict_bias = self._detect_treatment_conflict_bias(claim.text, claim.context)

        # Oblicz ogólne wskaźniki
        evidence_level = self._determine_evidence_level(evidence)
        avg_bias = total_bias / len(evidence) if evidence else 0.0
        avg_bias = max(avg_bias, conflict_bias)  # Użyj wyższego z bias

        # Oblicz confidence score
        confidence_score = self._calculate_confidence_score(
            supporting_studies, contradicting_studies, evidence_level, avg_bias
        )

        # Określ czy twierdzenie jest wspierane
        is_supported = len(supporting_studies) > len(contradicting_studies) and confidence_score > 0.6

        # Generuj wyjaśnienie i ostrzeżenia
        explanation = self._generate_explanation(
            claim, supporting_studies, contradicting_studies, evidence_level
        )

        warnings = self._generate_warnings(avg_bias, conflict_bias, evidence_level)
        recommendations = self._generate_recommendations(claim, evidence_level, avg_bias)

        return FactCheckResult(
            claim=claim.text,
            is_supported=is_supported,
            confidence_score=confidence_score,
            evidence_level=evidence_level,
            supporting_studies=len(supporting_studies),
            contradicting_studies=len(contradicting_studies),
            bias_detected=avg_bias > 0.3,
            corporate_influence=avg_bias,
            explanation=explanation,
            warnings=warnings,
            recommendations=recommendations
        )

    def _extract_medical_terms(self, text: str) -> List[str]:
        """Wyciąga terminy medyczne z tekstu"""
        medical_terms = []
        text_lower = text.lower()

        # Podstawowe terminy medyczne
        common_medical_terms = [
            'cancer', 'tumor', 'malignant', 'benign', 'chemotherapy', 'radiotherapy',
            'ketosis', 'ketogenic', 'glucose', 'insulin', 'metabolism', 'fasting',
            'inflammation', 'immune', 'hormone', 'enzyme', 'protein', 'vitamin',
            'mineral', 'supplement', 'diet', 'nutrition', 'therapy', 'treatment',
            'diagnosis', 'symptom', 'disease', 'disorder', 'syndrome', 'condition'
        ]

        for term in common_medical_terms:
            if term in text_lower:
                medical_terms.append(term)

        # Użyj NLP jeśli dostępne
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['DISEASE', 'CHEMICAL', 'DRUG']:
                    medical_terms.append(ent.text.lower())

        return list(set(medical_terms))

    def _assess_claim_confidence(self, claim_text: str) -> float:
        """Ocenia pewność twierdzenia na podstawie języka"""
        confidence = 0.5  # bazowa pewność

        # Silne stwierdzenia
        strong_indicators = ['always', 'never', 'all', 'every', 'completely', 'totally']
        weak_indicators = ['may', 'might', 'could', 'possibly', 'sometimes', 'often']
        moderate_indicators = ['usually', 'generally', 'typically', 'commonly']

        text_lower = claim_text.lower()

        for indicator in strong_indicators:
            if indicator in text_lower:
                confidence += 0.3

        for indicator in weak_indicators:
            if indicator in text_lower:
                confidence -= 0.2

        for indicator in moderate_indicators:
            if indicator in text_lower:
                confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def _get_surrounding_context(self, text: str, start: int, end: int, window: int = 100) -> str:
        """Pobiera kontekst wokół znalezionego twierdzenia"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]

    def _determine_medical_category(self, medical_terms: List[str]) -> str:
        """Określa kategorię medyczną na podstawie terminów"""
        categories = self.research_db.medical_categories

        category_scores = {}
        for category, keywords in categories.items():
            score = 0
            for term in medical_terms:
                if term in keywords:
                    score += 1
            if score > 0:
                category_scores[category] = score

        if category_scores:
            return max(category_scores, key=category_scores.get)
        else:
            return 'general'

    def _study_supports_claim(self, study, claim: ExtractedClaim) -> bool:
        """Sprawdza czy badanie wspiera dane twierdzenie"""
        study_text = (study.title + " " + study.abstract + " " + study.key_findings).lower()
        claim_terms = set(claim.medical_terms)

        # Sprawdź czy badanie zawiera terminy z twierdzenia
        study_terms = self._extract_medical_terms(study_text)
        overlap = len(claim_terms.intersection(set(study_terms)))

        # Jeśli jest duże pokrycie terminów, prawdopodobnie wspiera
        return overlap >= len(claim_terms) * 0.5

    def _detect_treatment_conflict_bias(self, claim_text: str, context: str) -> float:
        """
        Wykrywa bias w konfliktach leczenia (np. radioterapia vs ketoza)
        """
        bias_score = 0.0
        full_text = (claim_text + " " + context).lower()

        for conflict_name, conflict_data in self.treatment_conflicts.items():
            conventional_found = any(term in full_text for term in conflict_data['conventional'])
            alternative_found = any(term in full_text for term in conflict_data['alternative'])

            if conventional_found and alternative_found:
                # Sprawdź czy jest bias w kierunku konwencjonalnego leczenia
                for bias_indicator in conflict_data['bias_indicators']:
                    if bias_indicator.lower() in full_text:
                        bias_score += 0.4
                        logger.warning(f"Treatment conflict bias detected: {conflict_name}")

        return min(bias_score, 1.0)

    def _determine_evidence_level(self, evidence: List) -> EvidenceLevel:
        """Określa poziom dowodów na podstawie dostępnych badań"""
        if not evidence:
            return EvidenceLevel.EXPERT_OPINION

        # Znajdź najwyższy poziom dowodów
        best_level = EvidenceLevel.EXPERT_OPINION

        for study in evidence:
            if study.study_type == 'systematic_review':
                best_level = min(best_level, EvidenceLevel.SYSTEMATIC_REVIEW)
            elif study.study_type == 'meta_analysis':
                best_level = min(best_level, EvidenceLevel.META_ANALYSIS)
            elif study.study_type == 'rct':
                best_level = min(best_level, EvidenceLevel.RANDOMIZED_TRIAL)
            elif study.study_type == 'cohort':
                best_level = min(best_level, EvidenceLevel.COHORT_STUDY)

        return best_level

    def _calculate_confidence_score(self, supporting_studies: List, contradicting_studies: List,
                                    evidence_level: EvidenceLevel, bias_score: float) -> float:
        """Oblicza confidence score dla twierdzenia"""
        # Bazowy score na podstawie badań
        support_ratio = len(supporting_studies) / max(1, len(supporting_studies) + len(contradicting_studies))

        # Bonus za wysoką jakość dowodów
        evidence_bonus = {
            EvidenceLevel.SYSTEMATIC_REVIEW: 0.3,
            EvidenceLevel.META_ANALYSIS: 0.25,
            EvidenceLevel.RANDOMIZED_TRIAL: 0.2,
            EvidenceLevel.COHORT_STUDY: 0.1,
            EvidenceLevel.CASE_CONTROL: 0.05,
            EvidenceLevel.CASE_SERIES: 0.0,
            EvidenceLevel.EXPERT_OPINION: -0.1
        }.get(evidence_level, 0.0)

        # Kara za bias
        bias_penalty = bias_score * 0.5

        confidence = support_ratio + evidence_bonus - bias_penalty
        return max(0.0, min(1.0, confidence))

    def _generate_explanation(self, claim: ExtractedClaim, supporting_studies: List,
                              contradicting_studies: List, evidence_level: EvidenceLevel) -> str:
        """Generuje wyjaśnienie wyniku fact-check"""
        explanation = f"Twierdzenie: '{claim.text}'\n\n"

        if supporting_studies:
            explanation += f"WSPIERAJĄCE DOWODY ({len(supporting_studies)} badań):\n"
            for study in supporting_studies[:3]:  # Pokaż top 3
                explanation += f"- {study.title} ({study.journal}, {study.year})\n"
                explanation += f"  Jakość: {study.quality_score:.2f}, Wiarygodność: {study.credibility_score:.2f}\n"

        if contradicting_studies:
            explanation += f"\nPRZECIWNE DOWODY ({len(contradicting_studies)} badań):\n"
            for study in contradicting_studies[:2]:  # Pokaż top 2
                explanation += f"- {study.title} ({study.journal}, {study.year})\n"

        explanation += f"\nPoziom dowodów: {evidence_level.name.replace('_', ' ').title()}"

        return explanation

    def _generate_warnings(self, bias_score: float, conflict_bias: float,
                           evidence_level: EvidenceLevel) -> List[str]:
        """Generuje ostrzeżenia"""
        warnings = []

        if bias_score > 0.5:
            warnings.append("WYSOKIE RYZYKO CORPORATE BIAS - sprawdź źródła finansowania badań")

        if conflict_bias > 0.3:
            warnings.append("WYKRYTO KONFLIKT LECZENIA - możliwe promowanie droższych opcji kosztem skuteczniejszych")

        if evidence_level in [EvidenceLevel.CASE_SERIES, EvidenceLevel.EXPERT_OPINION]:
            warnings.append("NISKI POZIOM DOWODÓW - wymagane więcej badań wysokiej jakości")

        # Specjalne ostrzeżenie dla przykładu radioterapia vs ketoza
        if any(term in warnings[0].lower() if warnings else "" for term in ['radiotherapy', 'ketogenic']):
            warnings.append("UWAGA: Sprawdź czy promowane nie są drogie zabiegi kosztem naturalnych metod")

        return warnings

    def _generate_recommendations(self, claim: ExtractedClaim, evidence_level: EvidenceLevel,
                                  bias_score: float) -> List[str]:
        """Generuje rekomendacje"""
        recommendations = []

        if evidence_level in [EvidenceLevel.SYSTEMATIC_REVIEW, EvidenceLevel.META_ANALYSIS]:
            recommendations.append("Silne dowody naukowe - można ufać temu twierdzeniu")
        elif evidence_level == EvidenceLevel.RANDOMIZED_TRIAL:
            recommendations.append("Dobre dowody - twierdzenie prawdopodobnie prawdziwe")
        else:
            recommendations.append("Słabe dowody - wymagana ostrożność w interpretacji")

        if bias_score > 0.3:
            recommendations.append("Sprawdź niezależne źródła i badania nie finansowane przez przemysł")
            recommendations.append("Poszukaj badań porównujących z alternatywnymi metodami")

        if claim.claim_type == ClaimType.TREATMENT_EFFICACY:
            recommendations.append("Skonsultuj się z niezależnym specjalistą przed podjęciem decyzji o leczeniu")

        return recommendations

    async def detect_text_bias(self, text: str, fact_results: List[FactCheckResult]) -> float:
        """Wykrywa bias na poziomie całego tekstu"""
        text_lower = text.lower()
        total_bias = 0.0
        bias_count = 0

        # Sprawdź ogólne wskaźniki bias
        for bias_type, indicators in self.bias_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    total_bias += 0.2
                    bias_count += 1

        # Sprawdź pattern: konwencjonalne leczenie promowane, alternatywy krytykowane
        conventional_positive = 0
        alternative_negative = 0

        conventional_terms = ['surgery', 'chemotherapy', 'radiotherapy', 'pharmaceutical', 'drug']
        alternative_terms = ['ketogenic', 'fasting', 'natural', 'dietary', 'lifestyle']

        for conv_term in conventional_terms:
            if conv_term in text_lower:
                # Sprawdź czy jest pozytywny kontekst
                if any(pos in text_lower for pos in ['effective', 'proven', 'recommended', 'standard']):
                    conventional_positive += 1

        for alt_term in alternative_terms:
            if alt_term in text_lower:
                # Sprawdź czy jest negatywny kontekst
                if any(neg in text_lower for neg in ['unproven', 'dangerous', 'insufficient', 'not recommended']):
                    alternative_negative += 1

        # Jeśli konwencjonalne są promowane, a alternatywy krytykowane = red flag
        if conventional_positive > 0 and alternative_negative > 0:
            total_bias += 0.5
            logger.warning("Pattern detected: conventional treatments promoted while alternatives criticized")

        # Uśrednij bias z fact-check results
        if fact_results:
            avg_fact_bias = sum(r.corporate_influence for r in fact_results) / len(fact_results)
            total_bias = max(total_bias, avg_fact_bias)

        return min(total_bias, 1.0)


# Test funkcji
async def test_fact_checker():
    """Test fact checkera"""
    from .research_database import ResearchDatabase

    # Inicjalizuj komponenty
    research_db = ResearchDatabase()
    fact_checker = MedicalFactChecker(research_db)

    # Test text z potencjalnym bias (przykład radioterapia vs ketoza)
    test_text = """
    Radiotherapy remains the gold standard treatment for cancer patients. 
    While some patients inquire about ketogenic diets, these approaches 
    are not scientifically validated and may be dangerous. 
    Established radiation therapy protocols have proven effectiveness, 
    unlike unproven dietary interventions that require extensive research.
    """

    print("Testing fact checker with potentially biased text...")
    results = await fact_checker.check_facts_in_text(test_text)

    for result in results:
        print(f"\nClaim: {result.claim}")
        print(f"Supported: {result.is_supported}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Corporate bias: {result.corporate_influence:.2f}")
        print(f"Warnings: {result.warnings}")
        print("---")


if __name__ == "__main__":
    asyncio.run(test_fact_checker())