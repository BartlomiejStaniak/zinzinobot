#!/usr/bin/.env python3
"""
scientific_validator.py - Główny walidator naukowy
Plik: core/scientific_validator.py
"""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

# Import lokalnych modułów
from scientific_validation.research_database import ResearchDatabase, ValidationClaim
from scientific_validation.fact_checker import MedicalFactChecker, FactCheckResult
from scientific_validation.credibility_scorer import CredibilityScorer, SourceInfo, CredibilityScore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Kompletny wynik walidacji naukowej"""
    content: str
    overall_credibility: float  # 0.0 - 1.0
    is_scientifically_valid: bool
    confidence_level: str  # "high", "medium", "low", "very_low"

    # Szczegółowe wyniki
    fact_check_results: List[FactCheckResult]
    source_credibility_scores: List[CredibilityScore]

    # Red flags i ostrzeżenia
    corporate_bias_detected: bool
    bias_score: float
    major_red_flags: List[str]
    warnings: List[str]

    # Rekomendacje
    recommendations: List[str]
    alternative_sources_needed: bool

    # Metadata
    validation_timestamp: str
    processing_time_seconds: float

    # Summary dla agentów AI
    agent_summary: str
    should_use_content: bool


class ScientificValidator:
    """
    Główny walidator naukowy - łączy wszystkie komponenty
    """

    def __init__(self):
        self.research_db = ResearchDatabase()
        self.fact_checker = MedicalFactChecker(self.research_db)
        self.credibility_scorer = CredibilityScorer()

        # Progi akceptacji
        self.validation_thresholds = {
            'minimum_credibility': 0.60,  # Minimalny próg wiarygodności
            'high_confidence': 0.80,  # Wysoka pewność
            'bias_warning_threshold': 0.30,  # Próg ostrzeżenia o bias
            'bias_rejection_threshold': 0.60,  # Próg odrzucenia z powodu bias
            'minimum_supporting_studies': 2  # Min. liczba wspierających badań
        }

        # Kategorie treści wymagające specjalnej uwagi
        self.sensitive_categories = {
            'cancer_treatment': {
                'keywords': ['cancer', 'tumor', 'chemotherapy', 'radiotherapy', 'oncology'],
                'extra_scrutiny': True,
                'bias_multiplier': 1.5
            },
            'alternative_medicine': {
                'keywords': ['ketogenic', 'fasting', 'natural', 'holistic', 'alternative'],
                'extra_scrutiny': True,
                'bias_multiplier': 1.2
            },
            'nutrition_health': {
                'keywords': ['diet', 'nutrition', 'supplement', 'vitamin', 'mineral'],
                'extra_scrutiny': False,
                'bias_multiplier': 1.0
            }
        }

    async def validate_content(self, content: str, context: str = "") -> ValidationResult:
        """
        Główna metoda walidacji treści
        """
        start_time = datetime.now()
        logger.info(f"Starting scientific validation of content: {content[:100]}...")

        try:
            # 1. Sprawdź fakty w treści
            fact_check_results = await self.fact_checker.check_facts_in_text(content)

            # 2. Oceń wiarygodność źródeł (jeśli są dostępne w treści)
            source_credibility_scores = await self._evaluate_sources_in_content(content)

            # 3. Wykryj corporate bias
            bias_analysis = await self._comprehensive_bias_analysis(content, fact_check_results)

            # 4. Sprawdź kategorie wrażliwe
            sensitivity_analysis = self._analyze_content_sensitivity(content)

            # 5. Oblicz ogólną wiarygodność
            overall_credibility = self._calculate_overall_credibility(
                fact_check_results, source_credibility_scores, bias_analysis, sensitivity_analysis
            )

            # 6. Określ czy treść jest naukowo poprawna
            is_valid, confidence_level = self._determine_validity(overall_credibility, bias_analysis)

            # 7. Generuj ostrzeżenia i rekomendacje
            warnings = self._generate_warnings(bias_analysis, fact_check_results, sensitivity_analysis)
            recommendations = self._generate_recommendations(overall_credibility, bias_analysis, fact_check_results)

            # 8. Określ czy agenci AI powinni używać tej treści
            should_use = self._should_agents_use_content(overall_credibility, bias_analysis)

            # 9. Stwórz podsumowanie dla agentów
            agent_summary = self._create_agent_summary(overall_credibility, bias_analysis, is_valid)

            processing_time = (datetime.now() - start_time).total_seconds()

            result = ValidationResult(
                content=content,
                overall_credibility=overall_credibility,
                is_scientifically_valid=is_valid,
                confidence_level=confidence_level,
                fact_check_results=fact_check_results,
                source_credibility_scores=source_credibility_scores,
                corporate_bias_detected=bias_analysis['bias_detected'],
                bias_score=bias_analysis['bias_score'],
                major_red_flags=bias_analysis['major_red_flags'],
                warnings=warnings,
                recommendations=recommendations,
                alternative_sources_needed=overall_credibility < self.validation_thresholds['minimum_credibility'],
                validation_timestamp=datetime.now().isoformat(),
                processing_time_seconds=processing_time,
                agent_summary=agent_summary,
                should_use_content=should_use
            )

            logger.info(
                f"Validation completed in {processing_time:.2f}s. Overall credibility: {overall_credibility:.2f}")
            return result

        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            # Zwróć bezpieczny wynik w przypadku błędu
            return self._create_error_result(content, str(e))

    async def _evaluate_sources_in_content(self, content: str) -> List[CredibilityScore]:
        """Ocenia wiarygodność źródeł znalezionych w treści"""
        # Prosta implementacja - w przyszłości można rozszerzyć o wykrywanie URL, DOI, etc.
        sources = []

        # Znajdź potencjalne źródła (DOI, URL, nazwy czasopism)
        import re

        # Szukaj DOI
        doi_pattern = r'10\.\d{4,}\/[-._;()\/:a-zA-Z0-9]+'
        dois = re.findall(doi_pattern, content)

        # Szukaj nazw czasopism
        journal_pattern = r'(Journal of|New England Journal|Nature|Science|Lancet|JAMA|BMJ)[\w\s]*'
        journals = re.findall(journal_pattern, content, re.IGNORECASE)

        # Dla każdego znalezionego źródła stwórz SourceInfo i oceń
        for doi in dois:
            # Symulowane dane - w rzeczywistości pobierałbyś z CrossRef API
            source_info = SourceInfo(
                title="Found source with DOI",
                authors=["Unknown"],
                journal="Unknown Journal",
                year=2023,
                doi=doi,
                funding_sources=[],
                conflicts_of_interest=[],
                methodology="Unknown",
                sample_size=0,
                study_duration=None,
                url=None
            )
            score = await self.credibility_scorer.score_source_credibility(source_info)
            sources.append(score)

        return sources

    async def _comprehensive_bias_analysis(self, content: str, fact_results: List[FactCheckResult]) -> Dict:
        """Kompleksowa analiza bias"""
        content_lower = content.lower()

        # Zbierz bias z fact-check results
        fact_bias_scores = [r.corporate_influence for r in fact_results]
        avg_fact_bias = sum(fact_bias_scores) / len(fact_bias_scores) if fact_bias_scores else 0.0

        # Wykryj specyficzne wzorce bias
        treatment_bias = self._detect_treatment_bias(content)
        language_bias = self._detect_biased_language(content)
        financial_bias = self._detect_financial_bias(content)

        # Specjalna detekcja dla przykładu radioterapia vs ketoza
        radiotherapy_ketosis_bias = self._detect_radiotherapy_ketosis_bias(content)

        # Oblicz ogólny bias score
        total_bias = max(avg_fact_bias, treatment_bias, language_bias, financial_bias, radiotherapy_ketosis_bias)

        # Zbierz major red flags
        major_red_flags = []
        if radiotherapy_ketosis_bias > 0.4:
            major_red_flags.append("WYKRYTO BIAS: Promowanie radioterapii kosztem ketogennej terapii")
        if financial_bias > 0.5:
            major_red_flags.append("WYKRYTO BIAS: Wpływ finansowy korporacji farmaceutycznych")
        if treatment_bias > 0.5:
            major_red_flags.append("WYKRYTO BIAS: Systematyczne promowanie drogich opcji leczenia")

        return {
            'bias_detected': total_bias > self.validation_thresholds['bias_warning_threshold'],
            'bias_score': total_bias,
            'treatment_bias': treatment_bias,
            'language_bias': language_bias,
            'financial_bias': financial_bias,
            'radiotherapy_ketosis_bias': radiotherapy_ketosis_bias,
            'major_red_flags': major_red_flags
        }

    def _detect_treatment_bias(self, content: str) -> float:
        """Wykrywa bias w przedstawianiu opcji leczenia"""
        content_lower = content.lower()
        bias_score = 0.0

        # Wzorce promowania drogich opcji
        expensive_promotion = [
            'gold standard treatment', 'first-line therapy', 'established medical intervention',
            'proven pharmaceutical approach', 'standard oncology protocol'
        ]

        # Wzorce krytykowania tanich/naturalnych opcji
        cheap_criticism = [
            'unproven dietary intervention', 'insufficient lifestyle changes',
            'dangerous natural therapy', 'unvalidated nutritional approach'
        ]

        expensive_count = sum(1 for phrase in expensive_promotion if phrase in content_lower)
        cheap_critic_count = sum(1 for phrase in cheap_criticism if phrase in content_lower)

        if expensive_count > 0 and cheap_critic_count > 0:
            bias_score = 0.6  # Silny wzorzec bias
        elif expensive_count > cheap_critic_count:
            bias_score = 0.3

        return bias_score

    def _detect_biased_language(self, content: str) -> float:
        """Wykrywa stronniczy język"""
        content_lower = content.lower()
        bias_indicators = 0

        # Marketing speak
        marketing_phrases = [
            'breakthrough treatment', 'revolutionary therapy', 'cutting-edge medicine',
            'proven results', 'clinically tested', 'doctor recommended'
        ]

        # Fear-mongering language
        fear_phrases = [
            'dangerous if untreated', 'life-threatening condition', 'urgent intervention required',
            'delay may be fatal', 'immediate treatment necessary'
        ]

        for phrase in marketing_phrases + fear_phrases:
            if phrase in content_lower:
                bias_indicators += 1

        return min(bias_indicators * 0.1, 0.5)  # Max 0.5 bias score

    def _detect_financial_bias(self, content: str) -> float:
        """Wykrywa wpływ finansowy"""
        content_lower = content.lower()

        financial_indicators = [
            'sponsored by', 'funded by pharmaceutical', 'partnership with',
            'clinical trial sponsored', 'industry collaboration'
        ]

        pharma_companies = [
            'pfizer', 'novartis', 'roche', 'merck', 'bristol myers',
            'johnson & johnson', 'abbvie', 'amgen'
        ]

        bias_score = 0.0

        for indicator in financial_indicators:
            if indicator in content_lower:
                bias_score += 0.2

        for company in pharma_companies:
            if company in content_lower:
                bias_score += 0.15

        return min(bias_score, 0.7)

    def _detect_radiotherapy_ketosis_bias(self, content: str) -> float:
        """Wykrywa specyficzny bias radioterapia vs ketoza (główny przykład)"""
        content_lower = content.lower()

        # Sprawdź czy oba tematy są obecne
        radiotherapy_terms = ['radiotherapy', 'radiation therapy', 'radioactive treatment', 'radiation oncology']
        ketosis_terms = ['ketogenic diet', 'ketosis', 'metabolic therapy', 'fasting therapy', 'nutritional ketosis']

        has_radiotherapy = any(term in content_lower for term in radiotherapy_terms)
        has_ketosis = any(term in content_lower for term in ketosis_terms)

        if not (has_radiotherapy and has_ketosis):
            return 0.0  # Brak konfliktu do sprawdzenia

        bias_score = 0.0

        # Wzorce promowania radioterapii
        radio_positive = [
            'radiotherapy proven effective', 'radiation therapy gold standard',
            'established radiotherapy protocols', 'radiotherapy first-line treatment',
            'radiation therapy recommended approach'
        ]

        # Wzorce krytykowania ketosis
        ketosis_negative = [
            'ketogenic diet unproven', 'ketosis dangerous for cancer',
            'metabolic therapy insufficient evidence', 'fasting therapy not recommended',
            'nutritional approaches inadequate'
        ]

        radio_positive_count = sum(1 for phrase in radio_positive if phrase in content_lower)
        ketosis_negative_count = sum(1 for phrase in ketosis_negative if phrase in content_lower)

        # Jeśli radiotherapy jest promowana, a ketosis krytykowana = duży red flag
        if radio_positive_count > 0 and ketosis_negative_count > 0:
            bias_score = 0.8  # Bardzo wysoki bias
            logger.warning("MAJOR BIAS DETECTED: Radiotherapy promoted while ketogenic therapy criticized")
        elif radio_positive_count > 0:
            bias_score = 0.3
        elif ketosis_negative_count > 0:
            bias_score = 0.4

        # Dodatkowy wzorzec: koszt vs skuteczność
        cost_benefit_bias = [
            'expensive treatment necessary despite', 'cost-effective radiotherapy over',
            'insurance covers radiotherapy not', 'affordable radiation therapy versus'
        ]

        for phrase in cost_benefit_bias:
            if phrase in content_lower:
                bias_score += 0.2
                logger.warning(f"Cost-benefit bias detected: {phrase}")

        return min(bias_score, 1.0)

    def _analyze_content_sensitivity(self, content: str) -> Dict:
        """Analizuje wrażliwość treści (czy wymaga dodatkowej uwagi)"""
        content_lower = content.lower()

        detected_categories = []
        max_bias_multiplier = 1.0

        for category, details in self.sensitive_categories.items():
            keyword_matches = sum(1 for keyword in details['keywords'] if keyword in content_lower)

            if keyword_matches > 0:
                detected_categories.append({
                    'category': category,
                    'keyword_matches': keyword_matches,
                    'extra_scrutiny': details['extra_scrutiny'],
                    'bias_multiplier': details['bias_multiplier']
                })
                max_bias_multiplier = max(max_bias_multiplier, details['bias_multiplier'])

        return {
            'detected_categories': detected_categories,
            'requires_extra_scrutiny': any(cat['extra_scrutiny'] for cat in detected_categories),
            'bias_multiplier': max_bias_multiplier,
            'is_sensitive_content': len(detected_categories) > 0
        }

    def _calculate_overall_credibility(self, fact_results: List[FactCheckResult],
                                       source_scores: List[CredibilityScore],
                                       bias_analysis: Dict, sensitivity_analysis: Dict) -> float:
        """Oblicza ogólną wiarygodność treści"""

        # 1. Średnia z fact-check results
        if fact_results:
            fact_credibility = sum(r.confidence_score for r in fact_results) / len(fact_results)
            supported_ratio = sum(1 for r in fact_results if r.is_supported) / len(fact_results)
        else:
            fact_credibility = 0.5  # Neutralne jeśli brak fact-check
            supported_ratio = 0.5

        # 2. Średnia z source credibility
        if source_scores:
            source_credibility = sum(s.overall_score for s in source_scores) / len(source_scores)
        else:
            source_credibility = 0.5  # Neutralne jeśli brak źródeł

        # 3. Kara za bias
        bias_penalty = bias_analysis['bias_score'] * sensitivity_analysis['bias_multiplier']

        # 4. Oblicz weighted average
        weights = {
            'facts': 0.4,
            'sources': 0.3,
            'support_ratio': 0.3
        }

        base_score = (
                fact_credibility * weights['facts'] +
                source_credibility * weights['sources'] +
                supported_ratio * weights['support_ratio']
        )

        # 5. Zastosuj karę za bias
        final_score = max(0.0, base_score - bias_penalty)

        return final_score

    def _determine_validity(self, credibility: float, bias_analysis: Dict) -> Tuple[bool, str]:
        """Określa czy treść jest naukowo poprawna"""

        # Sprawdź czy bias nie jest zbyt wysoki
        if bias_analysis['bias_score'] > self.validation_thresholds['bias_rejection_threshold']:
            return False, "very_low"  # Odrzuć z powodu bias

        # Określ na podstawie wiarygodności
        if credibility >= self.validation_thresholds['high_confidence']:
            return True, "high"
        elif credibility >= self.validation_thresholds['minimum_credibility']:
            return True, "medium"
        elif credibility >= 0.4:
            return False, "low"
        else:
            return False, "very_low"

    def _generate_warnings(self, bias_analysis: Dict, fact_results: List[FactCheckResult],
                           sensitivity_analysis: Dict) -> List[str]:
        """Generuje ostrzeżenia"""
        warnings = []

        # Ostrzeżenia o bias
        if bias_analysis['bias_detected']:
            warnings.append(f"WYKRYTO CORPORATE BIAS (score: {bias_analysis['bias_score']:.2f})")

        if bias_analysis['radiotherapy_ketosis_bias'] > 0.3:
            warnings.append("UWAGA: Możliwe promowanie radioterapii kosztem ketogennej terapii")

        if bias_analysis['financial_bias'] > 0.4:
            warnings.append("WYKRYTO wpływ finansowy korporacji farmaceutycznych")

        # Ostrzeżenia z fact-check
        for result in fact_results:
            if not result.is_supported and result.confidence_score > 0.6:
                warnings.append(f"NIEPOTWIERDZONE TWIERDZENIE: {result.claim[:50]}...")

        # Ostrzeżenia o wrażliwości treści
        if sensitivity_analysis['requires_extra_scrutiny']:
            categories = [cat['category'] for cat in sensitivity_analysis['detected_categories']]
            warnings.append(f"WRAŻLIWA TREŚĆ - wymagana dodatkowa ostrożność: {', '.join(categories)}")

        return warnings

    def _generate_recommendations(self, credibility: float, bias_analysis: Dict,
                                  fact_results: List[FactCheckResult]) -> List[str]:
        """Generuje rekomendacje"""
        recommendations = []

        if credibility < self.validation_thresholds['minimum_credibility']:
            recommendations.append("ZALECENIE: Poszukaj dodatkowych, niezależnych źródeł")

        if bias_analysis['bias_score'] > 0.3:
            recommendations.append("ZALECENIE: Sprawdź źródła finansowania i konflikty interesów")
            recommendations.append("ZALECENIE: Poszukaj badań niezależnych od przemysłu")

        if bias_analysis['radiotherapy_ketosis_bias'] > 0.3:
            recommendations.append("ZALECENIE: Sprawdź niezależne badania nad terapią ketogenną w leczeniu nowotworów")
            recommendations.append("ZALECENIE: Porównaj koszty i skuteczność różnych opcji leczenia")

        # Sprawdź czy jest mało wspierających badań
        supporting_studies = sum(r.supporting_studies for r in fact_results)
        if supporting_studies < self.validation_thresholds['minimum_supporting_studies']:
            recommendations.append("ZALECENIE: Potrzebne więcej badań wspierających te twierdzenia")

        if credibility > 0.8:
            recommendations.append("POZYTYWNE: Treść oparta na solidnych dowodach naukowych")

        return recommendations

    def _should_agents_use_content(self, credibility: float, bias_analysis: Dict) -> bool:
        """Określa czy agenci AI powinni używać tej treści"""

        # Nie używaj jeśli bias zbyt wysoki
        if bias_analysis['bias_score'] > self.validation_thresholds['bias_rejection_threshold']:
            return False

        # Nie używaj jeśli wiarygodność zbyt niska
        if credibility < self.validation_thresholds['minimum_credibility']:
            return False

        # Specjalna uwaga dla radiotherapy vs ketosis bias
        if bias_analysis['radiotherapy_ketosis_bias'] > 0.5:
            return False

        return True

    def _create_agent_summary(self, credibility: float, bias_analysis: Dict, is_valid: bool) -> str:
        """Tworzy podsumowanie dla agentów AI"""

        if not is_valid:
            return f"⛔ NIE UŻYWAJ - Niska wiarygodność ({credibility:.2f}) lub wysoki bias ({bias_analysis['bias_score']:.2f})"

        if credibility > 0.8 and bias_analysis['bias_score'] < 0.2:
            return f"✅ BEZPIECZNE DO UŻYCIA - Wysoka wiarygodność ({credibility:.2f}), niski bias"

        if bias_analysis['radiotherapy_ketosis_bias'] > 0.3:
            return f"⚠️ OSTROŻNIE - Możliwy bias w leczeniu nowotworów (radiotherapy vs ketosis)"

        if bias_analysis['bias_score'] > 0.3:
            return f"⚠️ UŻYJ Z OSTROŻNOŚCIĄ - Wykryto corporate bias ({bias_analysis['bias_score']:.2f})"

        return f"✅ MOŻNA UŻYWAĆ - Wiarygodność: {credibility:.2f}, ale sprawdź kontekst"

    def _create_error_result(self, content: str, error_msg: str) -> ValidationResult:
        """Tworzy bezpieczny wynik w przypadku błędu"""
        return ValidationResult(
            content=content,
            overall_credibility=0.0,
            is_scientifically_valid=False,
            confidence_level="very_low",
            fact_check_results=[],
            source_credibility_scores=[],
            corporate_bias_detected=True,  # Bezpieczne założenie
            bias_score=1.0,
            major_red_flags=[f"ERROR DURING VALIDATION: {error_msg}"],
            warnings=[f"Błąd walidacji: {error_msg}", "Nie używaj tej treści do czasu rozwiązania problemu"],
            recommendations=["Sprawdź treść manualnie", "Poszukaj alternatywnych źródeł"],
            alternative_sources_needed=True,
            validation_timestamp=datetime.now().isoformat(),
            processing_time_seconds=0.0,
            agent_summary="⛔ BŁĄD WALIDACJI - NIE UŻYWAJ",
            should_use_content=False
        )

    async def validate_batch_content(self, content_list: List[str]) -> List[ValidationResult]:
        """Waliduje wiele treści jednocześnie"""
        results = []

        for content in content_list:
            result = await self.validate_content(content)
            results.append(result)

        return results

    def get_validation_summary(self, results: List[ValidationResult]) -> Dict:
        """Tworzy podsumowanie walidacji dla wielu treści"""
        total = len(results)
        valid_count = sum(1 for r in results if r.is_scientifically_valid)
        high_bias_count = sum(1 for r in results if r.bias_score > 0.5)
        usable_count = sum(1 for r in results if r.should_use_content)

        avg_credibility = sum(r.overall_credibility for r in results) / total if total > 0 else 0

        return {
            'total_validated': total,
            'scientifically_valid': valid_count,
            'high_bias_detected': high_bias_count,
            'usable_by_agents': usable_count,
            'average_credibility': avg_credibility,
            'validation_rate': valid_count / total if total > 0 else 0,
            'bias_rate': high_bias_count / total if total > 0 else 0,
            'usability_rate': usable_count / total if total > 0 else 0
        }


# Test kompletnego systemu
async def test_scientific_validator():
    """Test kompletnego systemu walidacji"""

    validator = ScientificValidator()

    # Test 1: Treść z potencjalnym bias (radioterapia vs ketoza)
    biased_content = """
    Radiotherapy remains the gold standard treatment for cancer patients. 
    Recent studies confirm that radiation therapy protocols are the most effective 
    first-line approach for oncology treatment. While some patients inquire about 
    ketogenic diets and metabolic therapies, these approaches are unproven and 
    may be dangerous. Established radiotherapy has decades of clinical evidence, 
    unlike experimental dietary interventions that lack sufficient research.
    Patients should focus on proven medical treatments rather than risky alternatives.
    """

    # Test 2: Treść wysoko-jakościowa
    good_content = """
    Recent systematic reviews and meta-analyses suggest that ketogenic diets 
    may have therapeutic potential in cancer treatment. Multiple randomized 
    controlled trials published in peer-reviewed journals indicate that 
    metabolic therapies can work synergistically with conventional treatments. 
    Both radiotherapy and ketogenic approaches have shown benefits in different 
    contexts, and personalized treatment plans should consider all evidence-based options.
    """

    print("🔬 Testing Scientific Validator...")
    print("=" * 50)

    # Test biased content
    print("\n📋 Test 1: Potentially Biased Content")
    biased_result = await validator.validate_content(biased_content)

    print(f"Overall Credibility: {biased_result.overall_credibility:.2f}")
    print(f"Scientifically Valid: {biased_result.is_scientifically_valid}")
    print(f"Corporate Bias: {biased_result.bias_score:.2f}")
    print(f"Should Use: {biased_result.should_use_content}")
    print(f"Agent Summary: {biased_result.agent_summary}")
    print(f"Major Red Flags: {biased_result.major_red_flags}")
    print(f"Warnings: {biased_result.warnings[:2]}")  # First 2 warnings

    # Test good content
    print(f"\n📋 Test 2: High-Quality Content")
    good_result = await validator.validate_content(good_content)

    print(f"Overall Credibility: {good_result.overall_credibility:.2f}")
    print(f"Scientifically Valid: {good_result.is_scientifically_valid}")
    print(f"Corporate Bias: {good_result.bias_score:.2f}")
    print(f"Should Use: {good_result.should_use_content}")
    print(f"Agent Summary: {good_result.agent_summary}")

    # Test batch validation
    print(f"\n📊 Batch Validation Summary")
    batch_results = await validator.validate_batch_content([biased_content, good_content])
    summary = validator.get_validation_summary(batch_results)

    print(f"Total validated: {summary['total_validated']}")
    print(f"Validation rate: {summary['validation_rate']:.1%}")
    print(f"Bias rate: {summary['bias_rate']:.1%}")
    print(f"Usability rate: {summary['usability_rate']:.1%}")
    print(f"Average credibility: {summary['average_credibility']:.2f}")


if __name__ == "__main__":
    asyncio.run(test_scientific_validator())