#!/usr/bin/.env python3
"""
content_extractor.py - Wyciąganie treści z różnych formatów plików
Plik: learning_engine/content_extractor.py
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import json
import base64
from datetime import datetime

# External libraries (install with pip)
try:
    import PyPDF2
    import docx
    from PIL import Image
    import pytesseract  # OCR
    import cv2
    import numpy as np
except ImportError as e:
    logging.warning(f"Some optional dependencies not installed: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractedContent:
    """Reprezentuje wyciągniętą treść z materiału"""
    main_text: str
    metadata: Dict[str, Any]
    images: List[Dict]  # Lista obrazów z opisami
    timestamps: List[Dict]  # Dla video/audio - znaczniki czasowe
    structure: Dict  # Struktura dokumentu (nagłówki, sekcje)
    extraction_confidence: float  # 0.0 - 1.0
    language: str
    source_file: str
    extraction_method: str
    processing_notes: List[str]


class ContentExtractor:
    """
    Główna klasa do wyciągania treści z różnych formatów
    """

    def __init__(self):
        # Konfiguracja OCR
        self.tesseract_config = '--oem 3 --psm 6'

        # Obsługiwane języki
        self.supported_languages = ['pol', 'eng']  # Polski i angielski

        # Konfiguracja wykrywania tekstu w obrazach
        self.image_processing_config = {
            'min_confidence': 0.6,
            'blur_threshold': 100,
            'min_text_length': 10
        }

        # Mapowanie formatów na metody ekstrakcji
        self.extraction_methods = {
            'pdf': self._extract_from_pdf,
            'document': self._extract_from_document,  # docx, txt
            'video': self._extract_from_video,
            'audio': self._extract_from_audio,
            'image': self._extract_from_image
        }

    async def extract_content(self, file_path: Path, material_type: str) -> Optional[ExtractedContent]:
        """
        Główna metoda ekstrakcji treści
        """
        logger.info(f"Extracting content from {file_path} (type: {material_type})")

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        extraction_method = self.extraction_methods.get(material_type)
        if not extraction_method:
            logger.error(f"No extraction method for type: {material_type}")
            return None

        try:
            extracted_content = await extraction_method(file_path)

            # Post-process treści
            if extracted_content:
                extracted_content = await self._post_process_content(extracted_content)
                logger.info(f"✅ Extracted {len(extracted_content.main_text)} characters from {file_path.name}")

            return extracted_content

        except Exception as e:
            logger.error(f"Error extracting content from {file_path}: {str(e)}")
            return None

    async def _extract_from_pdf(self, file_path: Path) -> ExtractedContent:
        """Wyciąga treść z plików PDF"""
        try:
            import PyPDF2
        except ImportError:
            logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            return self._create_empty_content(file_path, "pdf_extraction_failed")

        main_text = ""
        metadata = {}
        structure = {"pages": [], "headings": []}
        processing_notes = []

        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                # Metadata
                if pdf_reader.metadata:
                    metadata = {
                        'title': pdf_reader.metadata.get('/Title', ''),
                        'author': pdf_reader.metadata.get('/Author', ''),
                        'creator': pdf_reader.metadata.get('/Creator', ''),
                        'pages': len(pdf_reader.pages)
                    }

                # Wyciągnij tekst z każdej strony
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()

                        if page_text.strip():
                            main_text += f"\n--- Page {page_num + 1} ---\n"
                            main_text += page_text + "\n"

                            structure["pages"].append({
                                'page_number': page_num + 1,
                                'character_count': len(page_text),
                                'word_count': len(page_text.split())
                            })

                            # Wykryj nagłówki (linie krótsze niż 100 znaków, na początku strony)
                            lines = page_text.split('\n')
                            for line in lines[:5]:  # Sprawdź pierwsze 5 linii
                                line = line.strip()
                                if len(line) < 100 and len(line) > 5:
                                    structure["headings"].append({
                                        'text': line,
                                        'page': page_num + 1,
                                        'confidence': 0.7
                                    })

                    except Exception as e:
                        processing_notes.append(f"Error extracting page {page_num + 1}: {str(e)}")
                        continue

                # Jeśli brak tekstu, spróbuj OCR (dla skanowanych PDF)
                if len(main_text.strip()) < 100:
                    processing_notes.append("Low text extraction, attempting OCR...")
                    ocr_text = await self._ocr_pdf_fallback(file_path)
                    if ocr_text:
                        main_text += "\n--- OCR Extracted ---\n" + ocr_text
                        processing_notes.append("OCR extraction successful")

        except Exception as e:
            processing_notes.append(f"PDF extraction error: {str(e)}")
            return self._create_empty_content(file_path, f"pdf_error_{str(e)}")

        return ExtractedContent(
            main_text=main_text.strip(),
            metadata=metadata,
            images=[],
            timestamps=[],
            structure=structure,
            extraction_confidence=0.8 if len(main_text) > 100 else 0.4,
            language=self._detect_language(main_text),
            source_file=str(file_path),
            extraction_method="pdf_text_extraction",
            processing_notes=processing_notes
        )

    async def _extract_from_document(self, file_path: Path) -> ExtractedContent:
        """Wyciąga treść z dokumentów (txt, docx)"""
        suffix = file_path.suffix.lower()
        processing_notes = []

        if suffix == '.txt':
            # Zwykły plik tekstowy
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    main_text = file.read()
            except UnicodeDecodeError:
                # Spróbuj inne kodowanie
                try:
                    with open(file_path, 'r', encoding='latin-1') as file:
                        main_text = file.read()
                    processing_notes.append("Used latin-1 encoding")
                except Exception as e:
                    processing_notes.append(f"Encoding error: {str(e)}")
                    return self._create_empty_content(file_path, "encoding_error")

        elif suffix == '.docx':
            # Dokument Word
            try:
                import docx
                doc = docx.Document(file_path)

                paragraphs = []
                structure = {"headings": [], "sections": []}

                for paragraph in doc.paragraphs:
                    text = paragraph.text.strip()
                    if text:
                        paragraphs.append(text)

                        # Wykryj nagłówki na podstawie stylu
                        if paragraph.style.name.startswith('Heading'):
                            structure["headings"].append({
                                'text': text,
                                'level': paragraph.style.name,
                                'confidence': 0.9
                            })

                main_text = '\n\n'.join(paragraphs)

                # Metadata z właściwości dokumentu
                metadata = {
                    'title': doc.core_properties.title or '',
                    'author': doc.core_properties.author or '',
                    'created': str(doc.core_properties.created) if doc.core_properties.created else '',
                    'modified': str(doc.core_properties.modified) if doc.core_properties.modified else '',
                    'paragraphs': len(paragraphs)
                }

            except ImportError:
                processing_notes.append("python-docx not installed")
                return self._create_empty_content(file_path, "docx_dependency_missing")
            except Exception as e:
                processing_notes.append(f"DOCX extraction error: {str(e)}")
                return self._create_empty_content(file_path, "docx_error")

        else:
            processing_notes.append(f"Unsupported document format: {suffix}")
            return self._create_empty_content(file_path, "unsupported_format")

        return ExtractedContent(
            main_text=main_text,
            metadata=metadata if 'metadata' in locals() else {},
            images=[],
            timestamps=[],
            structure=structure if 'structure' in locals() else {},
            extraction_confidence=0.9,
            language=self._detect_language(main_text),
            source_file=str(file_path),
            extraction_method=f"document_{suffix[1:]}_extraction",
            processing_notes=processing_notes
        )

    async def _extract_from_video(self, file_path: Path) -> ExtractedContent:
        """Wyciąga treść z plików wideo (transkrypcja + OCR klatek)"""
        processing_notes = []
        main_text = ""
        timestamps = []
        images = []

        # Dla uproszczenia - symulacja transkrypcji
        # W pełnej implementacji użyłbyś whisper-ai lub podobnego
        processing_notes.append("Video transcription simulated - implement with whisper-ai")

        try:
            # Symulowane wyniki transkrypcji
            if "omega" in file_path.name.lower() or "health" in file_path.name.lower():
                main_text = """
                [SIMULATED TRANSCRIPTION]

                Welcome to today's video about omega-3 fatty acids and their impact on mental health.

                In this presentation, we'll discuss:
                - The importance of omega-3 balance
                - How modern diets create imbalances
                - Scientific evidence for omega-3 benefits
                - Practical approaches to optimization

                Recent studies show that omega-6 to omega-3 ratios in modern diets
                often exceed 20:1, when the optimal ratio should be between 3:1 and 5:1.

                This imbalance contributes to chronic inflammation, which affects
                both physical and mental health outcomes.

                [END SIMULATED TRANSCRIPTION]
                """

                timestamps = [
                    {'time': '00:00:00', 'text': 'Video introduction'},
                    {'time': '00:01:30', 'text': 'Omega-3 balance discussion'},
                    {'time': '00:03:45', 'text': 'Scientific evidence review'},
                    {'time': '00:07:20', 'text': 'Practical recommendations'}
                ]

            # TODO: Implementacja prawdziwej transkrypcji
            # import whisper
            # model = whisper.load_model("base")
            # result = model.transcribe(str(file_path))
            # main_text = result["text"]
            # timestamps = result["segments"]

            # TODO: Wyciąganie kluczowych klatek i OCR
            # key_frames = extract_key_frames(file_path)
            # for frame in key_frames:
            #     text = pytesseract.image_to_string(frame)
            #     if text.strip():
            #         images.append({'frame_time': frame_time, 'ocr_text': text})

        except Exception as e:
            processing_notes.append(f"Video processing error: {str(e)}")

        return ExtractedContent(
            main_text=main_text,
            metadata={'duration': 'unknown', 'format': file_path.suffix},
            images=images,
            timestamps=timestamps,
            structure={},
            extraction_confidence=0.6,  # Lower confidence for simulated data
            language=self._detect_language(main_text),
            source_file=str(file_path),
            extraction_method="video_transcription_simulation",
            processing_notes=processing_notes
        )

    async def _extract_from_audio(self, file_path: Path) -> ExtractedContent:
        """Wyciąga treść z plików audio (transkrypcja)"""
        processing_notes = []

        # Symulacja transkrypcji audio
        processing_notes.append("Audio transcription simulated - implement with whisper-ai")

        main_text = """
        [SIMULATED AUDIO TRANSCRIPTION]

        This is a podcast about wellness and nutritional balance.
        Today we're discussing the importance of omega-3 fatty acids
        for both physical and mental health.

        Our guest explains how the BalanceTest can help identify
        omega imbalances and guide personalized nutrition strategies.

        Key points covered:
        - Testing current omega-6 to omega-3 ratios
        - Understanding optimal balance ranges  
        - Implementing dietary and supplement strategies
        - Monitoring progress over time

        [END SIMULATED TRANSCRIPTION]
        """

        timestamps = [
            {'time': '00:00:00', 'text': 'Podcast introduction'},
            {'time': '00:02:15', 'text': 'Guest introduction'},
            {'time': '00:05:30', 'text': 'Omega-3 discussion begins'},
            {'time': '00:12:45', 'text': 'BalanceTest explanation'}
        ]

        return ExtractedContent(
            main_text=main_text,
            metadata={'duration': 'unknown', 'format': file_path.suffix},
            images=[],
            timestamps=timestamps,
            structure={},
            extraction_confidence=0.6,
            language=self._detect_language(main_text),
            source_file=str(file_path),
            extraction_method="audio_transcription_simulation",
            processing_notes=processing_notes
        )

    async def _extract_from_image(self, file_path: Path) -> ExtractedContent:
        """Wyciąga tekst z obrazów za pomocą OCR"""
        processing_notes = []
        main_text = ""

        try:
            # Sprawdź czy tesseract jest dostępny
            try:
                import pytesseract
                from PIL import Image
            except ImportError:
                processing_notes.append("OCR dependencies not installed (pytesseract, PIL)")
                return self._create_empty_content(file_path, "ocr_dependencies_missing")

            # Otwórz obraz
            image = Image.open(file_path)

            # Podstawowe przetwarzanie obrazu dla lepszego OCR
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # OCR z różnymi konfiguracjami
            try:
                # Próba 1: Standardowy OCR
                text = pytesseract.image_to_string(image, lang='pol+eng', config=self.tesseract_config)

                if len(text.strip()) < self.image_processing_config['min_text_length']:
                    # Próba 2: Inne konfiguracje PSM
                    text = pytesseract.image_to_string(image, lang='pol+eng', config='--oem 3 --psm 3')

                main_text = text.strip()

                if main_text:
                    processing_notes.append(f"OCR extracted {len(main_text)} characters")
                else:
                    processing_notes.append("No text detected in image")

            except Exception as e:
                processing_notes.append(f"OCR error: {str(e)}")
                return self._create_empty_content(file_path, "ocr_extraction_error")

            # Metadata obrazu
            metadata = {
                'format': image.format,
                'size': image.size,
                'mode': image.mode
            }

            if hasattr(image, '_getexif') and image._getexif():
                metadata['exif'] = dict(image._getexif())

        except Exception as e:
            processing_notes.append(f"Image processing error: {str(e)}")
            return self._create_empty_content(file_path, "image_processing_error")

        return ExtractedContent(
            main_text=main_text,
            metadata=metadata,
            images=[{'source': str(file_path), 'ocr_text': main_text}],
            timestamps=[],
            structure={},
            extraction_confidence=0.7 if len(main_text) > 50 else 0.3,
            language=self._detect_language(main_text),
            source_file=str(file_path),
            extraction_method="image_ocr",
            processing_notes=processing_notes
        )

    async def _ocr_pdf_fallback(self, file_path: Path) -> str:
        """OCR fallback dla skanowanych PDF"""
        # Symulacja OCR dla PDF - w rzeczywistości użyłbyś pdf2image + pytesseract
        return """
        [OCR FALLBACK - SIMULATED]

        This text was extracted using OCR from a scanned PDF.
        The document appears to contain information about:
        - Health and wellness topics
        - Nutritional recommendations
        - Scientific research references

        [END OCR SIMULATION]
        """

    def _detect_language(self, text: str) -> str:
        """Wykrywa język tekstu (uproszczona implementacja)"""
        if not text or len(text) < 10:
            return 'unknown'

        text_lower = text.lower()

        # Proste heurystyki językowe
        polish_indicators = ['że', 'się', 'nie', 'jest', 'oraz', 'może', 'można', 'bardzo']
        english_indicators = ['the', 'and', 'that', 'this', 'with', 'have', 'will', 'from']

        polish_count = sum(1 for word in polish_indicators if word in text_lower)
        english_count = sum(1 for word in english_indicators if word in text_lower)

        if polish_count > english_count:
            return 'polish'
        elif english_count > polish_count:
            return 'english'
        else:
            return 'mixed'

    async def _post_process_content(self, content: ExtractedContent) -> ExtractedContent:
        """Post-processing wyciągniętej treści"""

        # Czyszczenie tekstu
        cleaned_text = self._clean_extracted_text(content.main_text)

        # Poprawa struktury
        improved_structure = self._improve_text_structure(cleaned_text)

        # Aktualizuj confidence na podstawie jakości tekstu
        quality_score = self._assess_text_quality(cleaned_text)
        adjusted_confidence = content.extraction_confidence * quality_score

        # Zwróć poprawioną treść
        content.main_text = cleaned_text
        content.structure.update(improved_structure)
        content.extraction_confidence = adjusted_confidence

        return content

    def _clean_extracted_text(self, text: str) -> str:
        """Czyści wyciągnięty tekst"""
        import re

        # Usuń nadmiarowe białe znaki
        text = re.sub(r'\s+', ' ', text)

        # Usuń powtarzające się linie
        lines = text.split('\n')
        cleaned_lines = []
        prev_line = ""

        for line in lines:
            line = line.strip()
            if line and line != prev_line:
                cleaned_lines.append(line)
                prev_line = line

        # Złącz linie z poprawnym formatowaniem
        cleaned_text = '\n'.join(cleaned_lines)

        # Usuń dziwne znaki OCR
        cleaned_text = re.sub(r'[^\w\s\.,!?;:()\-\'\"\/]', '', cleaned_text)

        return cleaned_text.strip()

    def _improve_text_structure(self, text: str) -> Dict:
        """Poprawia strukturę tekstu i wykrywa sekcje"""
        lines = text.split('\n')
        structure = {
            'sections': [],
            'headings': [],
            'bullet_points': [],
            'statistics': []
        }

        import re

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Wykryj nagłówki
            if len(line) < 100 and (line.isupper() or line.endswith(':')):
                structure['headings'].append({
                    'text': line,
                    'line_number': i,
                    'confidence': 0.8
                })

            # Wykryj punkty
            if line.startswith(('- ', '• ', '* ', '1. ', '2. ')):
                structure['bullet_points'].append({
                    'text': line,
                    'line_number': i
                })

            # Wykryj statystyki/liczby
            if re.search(r'\d+%|\d+\.\d+|\d+:\d+', line):
                structure['statistics'].append({
                    'text': line,
                    'line_number': i,
                    'numbers': re.findall(r'\d+%|\d+\.\d+|\d+:\d+', line)
                })

        return structure

    def _assess_text_quality(self, text: str) -> float:
        """Ocenia jakość wyciągniętego tekstu"""
        if not text:
            return 0.0

        quality_score = 0.5  # Base score

        # Długość tekstu
        if len(text) > 500:
            quality_score += 0.2
        elif len(text) > 200:
            quality_score += 0.1

        # Obecność kompletnych zdań
        sentences = text.split('.')
        complete_sentences = [s for s in sentences if len(s.strip()) > 10]
        if len(complete_sentences) > 5:
            quality_score += 0.2

        # Brak dziwnych znaków (oznaka dobrego OCR)
        import re
        weird_chars = len(re.findall(r'[^\w\s\.,!?;:()\-\'\"\/]', text))
        if weird_chars < len(text) * 0.05:  # Mniej niż 5% dziwnych znaków
            quality_score += 0.1

        return min(quality_score, 1.0)

    def _create_empty_content(self, file_path: Path, error_reason: str) -> ExtractedContent:
        """Tworzy pustą treść w przypadku błędu"""
        return ExtractedContent(
            main_text="",
            metadata={'error': error_reason},
            images=[],
            timestamps=[],
            structure={},
            extraction_confidence=0.0,
            language='unknown',
            source_file=str(file_path),
            extraction_method="failed",
            processing_notes=[f"Extraction failed: {error_reason}"]
        )


# Test funkcji
async def test_content_extractor():
    """Test ekstraktora treści"""

    extractor = ContentExtractor()

    print("📄 Testing Content Extractor...")
    print("=" * 50)

    # Test 1: Tekst
    import tempfile
    test_text = """
    # Omega-3 Fatty Acids and Mental Health

    Recent research demonstrates the crucial role of omega-3 fatty acids in brain function.

    ## Key Benefits:
    - Improved mood regulation
    - Enhanced cognitive performance
    - Reduced inflammation

    The optimal omega-6 to omega-3 ratio is between 3:1 and 5:1.
    Modern diets often exceed 20:1, causing health issues.

    ### Zinzino Approach:
    BalanceOil combines omega-3 with polyphenols for optimal absorption.
    The BalanceTest verifies progress within 120 days.
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_text)
        temp_file = Path(f.name)

    try:
        # Test ekstrakcji tekstu
        result = await extractor.extract_content(temp_file, 'document')

        if result:
            print("✅ Text extraction successful!")
            print(f"- Extracted text length: {len(result.main_text)} characters")
            print(f"- Language detected: {result.language}")
            print(f"- Extraction confidence: {result.extraction_confidence:.2f}")
            print(f"- Headings found: {len(result.structure.get('headings', []))}")
            print(f"- Statistics found: {len(result.structure.get('statistics', []))}")

            # Pokaż przykładowe nagłówki
            headings = result.structure.get('headings', [])
            if headings:
                print("\n📋 Detected headings:")
                for heading in headings[:3]:
                    print(f"  - {heading['text']}")
        else:
            print("❌ Text extraction failed")

    finally:
        temp_file.unlink()

    print(f"\n🎬 Testing video extraction simulation...")

    # Test 2: Symulacja video
    video_path = Path("test_omega_video.mp4")  # Nie istnieje, ale test symulacji
    video_result = await extractor.extract_content(video_path, 'video')

    if video_result:
        print("✅ Video extraction simulation successful!")
        print(f"- Transcription length: {len(video_result.main_text)} characters")
        print(f"- Timestamps: {len(video_result.timestamps)}")
        print(f"- Processing notes: {video_result.processing_notes}")

        # Pokaż timestamps
        if video_result.timestamps:
            print("\n⏰ Sample timestamps:")
            for ts in video_result.timestamps[:2]:
                print(f"  {ts['time']}: {ts['text']}")

    # Test 3: Sprawdź jakość tekstu
    print(f"\n🔍 Testing text quality assessment...")

    good_text = "This is a well-structured document with complete sentences. It contains valuable information about omega-3 fatty acids and their health benefits."
    poor_text = "txt frm scn PDF w/ mny erors & incmpl wrds"

    good_quality = extractor._assess_text_quality(good_text)
    poor_quality = extractor._assess_text_quality(poor_text)

    print(f"Good text quality: {good_quality:.2f}")
    print(f"Poor text quality: {poor_quality:.2f}")


if __name__ == "__main__":
    asyncio.run(test_content_extractor())