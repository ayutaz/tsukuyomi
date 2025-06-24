"""
Japanese text normalization and phonemization

This module handles Japanese text preprocessing including normalization,
tokenization, and conversion to phoneme sequences for TTS.
"""

import re
import unicodedata
import warnings
from typing import Dict, List, Optional, Tuple, Union

# Japanese text processing libraries
try:
    import MeCab

    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False
    warnings.warn("MeCab not available. Japanese tokenization will be limited.")

try:
    import jaconv

    JACONV_AVAILABLE = True
except ImportError:
    JACONV_AVAILABLE = False
    warnings.warn(
        "jaconv not available. Some Japanese text conversions will be limited."
    )

try:
    import pykakasi

    KAKASI_AVAILABLE = True
except ImportError:
    KAKASI_AVAILABLE = False
    warnings.warn("pykakasi not available. Kanji reading conversion will be limited.")

try:
    from .text2phonemesequence import Text2PhonemeSequence

    TEXT2PHONEME_AVAILABLE = True
except ImportError:
    TEXT2PHONEME_AVAILABLE = False
    warnings.warn(
        "text2phonemesequence not available. Phoneme conversion will be limited."
    )


class JapaneseTextNormalizer:
    """
    Japanese text normalizer for TTS preprocessing.

    Handles text normalization, number/symbol conversion, and phonemization
    for Japanese text input.
    """

    def __init__(
        self,
        use_gpu: bool = False,
        mecab_dict: str = "unidic",
        normalize_numbers: bool = True,
        normalize_symbols: bool = True,
    ):
        """
        Initialize Japanese text normalizer.

        Args:
            use_gpu: Whether to use GPU for phonemization (if available)
            mecab_dict: MeCab dictionary to use ("unidic" or "ipadic")
            normalize_numbers: Whether to convert numbers to readings
            normalize_symbols: Whether to normalize symbols
        """
        self.use_gpu = use_gpu
        self.normalize_numbers = normalize_numbers
        self.normalize_symbols = normalize_symbols

        # Initialize MeCab
        if MECAB_AVAILABLE:
            try:
                if mecab_dict == "unidic":
                    self.mecab = MeCab.Tagger("-d /usr/local/lib/mecab/dic/unidic")
                else:
                    self.mecab = MeCab.Tagger()
            except:
                self.mecab = MeCab.Tagger()
        else:
            self.mecab = None

        # Initialize Kakasi for reading generation
        if KAKASI_AVAILABLE:
            self.kakasi = pykakasi.kakasi()
            self.kakasi.setMode("J", "H")  # Japanese to Hiragana
            self.kakasi.setMode("K", "H")  # Katakana to Hiragana
            self.kakasi_converter = self.kakasi.getConverter()
        else:
            self.kakasi_converter = None

        # Initialize phonemizer
        if TEXT2PHONEME_AVAILABLE:
            self.phonemizer = Text2PhonemeSequence(language="jpn", is_cuda=use_gpu)
        else:
            self.phonemizer = None

        # Number reading dictionary
        self.number_readings = {
            "0": "ゼロ",
            "1": "イチ",
            "2": "ニ",
            "3": "サン",
            "4": "ヨン",
            "5": "ゴ",
            "6": "ロク",
            "7": "ナナ",
            "8": "ハチ",
            "9": "キュウ",
            "10": "ジュウ",
            "100": "ヒャク",
            "1000": "セン",
            "10000": "マン",
        }

        # Symbol mapping
        self.symbol_map = {
            "。": "。",
            "、": "、",
            "？": "？",
            "！": "！",
            ".": "。",
            ",": "、",
            "?": "？",
            "!": "！",
            "・": "・",
            "…": "…",
            "～": "～",
            "ー": "ー",
            "（": "（",
            "）": "）",
            "(": "（",
            ")": "）",
            "「": "「",
            "」": "」",
            "『": "『",
            "』": "』",
        }

    def normalize(self, text: str) -> str:
        """
        Normalize Japanese text for TTS.

        Args:
            text: Input Japanese text

        Returns:
            Normalized text
        """
        # Basic text cleaning
        text = unicodedata.normalize("NFKC", text)
        text = text.strip()

        # Normalize symbols
        if self.normalize_symbols:
            text = self._normalize_symbols(text)

        # Normalize numbers
        if self.normalize_numbers:
            text = self._normalize_numbers(text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Handle English words (convert to katakana if possible)
        text = self._handle_english(text)

        return text

    def _normalize_symbols(self, text: str) -> str:
        """Normalize symbols in text."""
        for old, new in self.symbol_map.items():
            text = text.replace(old, new)

        # Remove or replace other problematic symbols
        text = re.sub(
            r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\u3400-\u4DBF"
            r"a-zA-Z0-9\s。、？！・…～ー（）「」『』]",
            "",
            text,
        )

        return text

    def _normalize_numbers(self, text: str) -> str:
        """Convert numbers to Japanese readings."""
        # Handle year patterns (e.g., 2024年)
        text = re.sub(
            r"(\d{4})年", lambda m: self._number_to_japanese(m.group(1)) + "ネン", text
        )

        # Handle general numbers
        text = re.sub(r"\d+", lambda m: self._number_to_japanese(m.group(0)), text)

        return text

    def _number_to_japanese(self, num_str: str) -> str:
        """Convert number string to Japanese reading."""
        if len(num_str) == 1:
            return self.number_readings.get(num_str, num_str)

        # For longer numbers, convert digit by digit (simplified)
        result = []
        for digit in num_str:
            result.append(self.number_readings.get(digit, digit))

        return "".join(result)

    def _handle_english(self, text: str) -> str:
        """Convert English words to katakana where possible."""
        if JACONV_AVAILABLE:
            # Convert alphabet to full-width
            text = jaconv.h2z(text, ascii=True)

        # Simple mapping for common English words
        english_to_katakana = {
            "AI": "エーアイ",
            "TTS": "ティーティーエス",
            "OK": "オーケー",
            "NG": "エヌジー",
        }

        for eng, kana in english_to_katakana.items():
            text = text.replace(eng, kana)

        return text

    def text_to_phonemes(
        self, text: str, normalize: bool = True, return_tokens: bool = False
    ) -> Union[str, Tuple[str, List[str]]]:
        """
        Convert Japanese text to phoneme sequence.

        Args:
            text: Input Japanese text
            normalize: Whether to normalize text first
            return_tokens: Whether to return tokenized text as well

        Returns:
            If return_tokens=False: Phoneme sequence string
            If return_tokens=True: Tuple of (phoneme_sequence, tokens)
        """
        # Normalize text if requested
        if normalize:
            text = self.normalize(text)

        tokens = []

        # Tokenize with MeCab if available
        if self.mecab is not None:
            node = self.mecab.parseToNode(text)
            while node:
                if node.surface:
                    tokens.append(node.surface)
                node = node.next
        else:
            # Fallback: simple character-based tokenization
            tokens = list(text)

        # Convert to phonemes
        if self.phonemizer is not None:
            phoneme_seq = self.phonemizer.inference_text(text)
        else:
            # Fallback: convert to hiragana as pseudo-phonemes
            if self.kakasi_converter is not None:
                phoneme_seq = self.kakasi_converter.do(text)
            else:
                phoneme_seq = text  # Last resort: return original text

        if return_tokens:
            return phoneme_seq, tokens
        else:
            return phoneme_seq

    def process_for_tts(
        self, text: str, add_silence: bool = True
    ) -> Dict[str, Union[str, List[str]]]:
        """
        Full preprocessing pipeline for TTS.

        Args:
            text: Input Japanese text
            add_silence: Whether to add silence tokens

        Returns:
            Dictionary containing:
                - normalized_text: Normalized text
                - phoneme_sequence: Phoneme sequence
                - tokens: List of tokens
        """
        # Normalize
        normalized_text = self.normalize(text)

        # Get phonemes and tokens
        phoneme_sequence, tokens = self.text_to_phonemes(
            normalized_text, normalize=False, return_tokens=True
        )

        # Add silence tokens if requested
        if add_silence:
            if isinstance(phoneme_sequence, str):
                phoneme_sequence = f"<SIL> {phoneme_sequence} <SIL>"
            tokens = ["<SIL>"] + tokens + ["<SIL>"]

        return {
            "normalized_text": normalized_text,
            "phoneme_sequence": phoneme_sequence,
            "tokens": tokens,
        }

    def batch_process(
        self, texts: List[str], add_silence: bool = True
    ) -> List[Dict[str, Union[str, List[str]]]]:
        """
        Process multiple texts for TTS.

        Args:
            texts: List of input texts
            add_silence: Whether to add silence tokens

        Returns:
            List of processed results
        """
        results = []
        for text in texts:
            results.append(self.process_for_tts(text, add_silence))
        return results
