"""
Real phonemizer implementation using multiple backends.

This module provides actual phoneme conversion functionality
for multiple languages, replacing the mock text2phonemesequence.
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# Language-specific imports
try:
    from phonemizer import phonemize
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator

    PHONEMIZER_AVAILABLE = True
except ImportError:
    PHONEMIZER_AVAILABLE = False
    logging.warning("phonemizer not available, falling back to basic implementation")

try:
    import pyopenjtalk

    OPENJTALK_AVAILABLE = True
except ImportError:
    OPENJTALK_AVAILABLE = False
    logging.warning("pyopenjtalk not available for Japanese G2P")

try:
    import pypinyin

    PYPINYIN_AVAILABLE = True
except ImportError:
    PYPINYIN_AVAILABLE = False
    logging.warning("pypinyin not available for Chinese G2P")

try:
    from g2p_en import G2p

    G2P_EN_AVAILABLE = True
except ImportError:
    G2P_EN_AVAILABLE = False
    logging.warning("g2p_en not available for English G2P")


class Language(Enum):
    """Supported languages."""

    JAPANESE = "ja"
    ENGLISH = "en"
    CHINESE = "zh"
    KOREAN = "ko"
    MULTILINGUAL = "multi"


@dataclass
class PhonemeConfig:
    """Configuration for phonemizer."""

    language: Language = Language.JAPANESE
    use_ipa: bool = True
    preserve_punctuation: bool = True
    with_stress: bool = True
    tie_token: str = ""
    language_switch_token: str = " [lang] "


class MultilingualPhonemizer:
    """
    Multilingual phonemizer with language-specific backends.

    Supports Japanese, English, Chinese, and Korean with
    appropriate G2P models for each language.
    """

    def __init__(self, config: Optional[PhonemeConfig] = None):
        """Initialize phonemizer with configuration."""
        self.config = config or PhonemeConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize language-specific backends
        self._init_backends()

        # Phoneme mappings for consistency
        self._init_phoneme_mappings()

    def _init_backends(self) -> None:
        """Initialize language-specific G2P backends."""
        # Japanese backend
        if OPENJTALK_AVAILABLE:
            self.logger.info("Using OpenJTalk for Japanese G2P")
        else:
            self.logger.warning("OpenJTalk not available, Japanese G2P will be limited")

        # English backend
        if G2P_EN_AVAILABLE:
            self.g2p_en = G2p()
            self.logger.info("Using g2p_en for English G2P")
        else:
            self.g2p_en = None

        # General phonemizer backend
        if PHONEMIZER_AVAILABLE:
            self.separator = Separator(
                phone=self.config.tie_token, word=" ", syllable=""
            )
            self.logger.info("Using phonemizer for general G2P")

    def _init_phoneme_mappings(self) -> None:
        """Initialize phoneme mappings for consistency."""
        # IPA to X-SAMPA mappings for common phones
        self.ipa_to_xsampa = {
            "a": "a",
            "e": "e",
            "i": "i",
            "o": "o",
            "u": "u",
            "ə": "@",
            "ɪ": "I",
            "ʊ": "U",
            "ɛ": "E",
            "ɔ": "O",
            "æ": "{",
            "ɑ": "A",
            "ʌ": "V",
            "ɒ": "Q",
            # Consonants
            "p": "p",
            "b": "b",
            "t": "t",
            "d": "d",
            "k": "k",
            "g": "g",
            "f": "f",
            "v": "v",
            "θ": "T",
            "ð": "D",
            "s": "s",
            "z": "z",
            "ʃ": "S",
            "ʒ": "Z",
            "h": "h",
            "m": "m",
            "n": "n",
            "ŋ": "N",
            "l": "l",
            "ɹ": "r\\",
            "j": "j",
            "w": "w",
            # Special markers
            "ˈ": "'",
            "ˌ": "%",
            " ": " ",
            ".": ".",
        }

    def phonemize(
        self,
        text: str | Sequence[str],
        language: Optional[Language] = None,
        return_tokens: bool = False,
    ) -> str | list[str] | tuple[list[str], list[list[str]]]:
        """
        Convert text to phonemes.

        Args:
            text: Input text or list of texts
            language: Language override (uses config default if None)
            return_tokens: If True, also return token boundaries

        Returns:
            Phoneme string(s), optionally with token boundaries
        """
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = list(text)
            single_input = False

        language = language or self.config.language

        # Process based on language
        if language == Language.JAPANESE:
            results = [self._phonemize_japanese(t, return_tokens) for t in texts]
        elif language == Language.ENGLISH:
            results = [self._phonemize_english(t, return_tokens) for t in texts]
        elif language == Language.CHINESE:
            results = [self._phonemize_chinese(t, return_tokens) for t in texts]
        else:
            results = [
                self._phonemize_general(t, language.value, return_tokens) for t in texts
            ]

        if single_input:
            return results[0]
        return results

    def _phonemize_japanese(
        self, text: str, return_tokens: bool = False
    ) -> str | tuple[str, list[str]]:
        """Japanese-specific phonemization using OpenJTalk."""
        if not OPENJTALK_AVAILABLE:
            return self._phonemize_general(text, "ja", return_tokens)

        # Clean text
        text = self._clean_text(text)

        # Get pronunciation from OpenJTalk
        try:
            # Returns katakana pronunciation
            pronunciation = pyopenjtalk.g2p(text, kana=True)

            # Convert katakana to phonemes
            phonemes = self._katakana_to_phonemes(pronunciation)

            if return_tokens:
                # Simple word segmentation
                tokens = text.split()
                return phonemes, tokens

            return phonemes

        except Exception as e:
            self.logger.error(f"Japanese G2P failed: {e}")
            return self._fallback_phonemize(text, return_tokens)

    def _phonemize_english(
        self, text: str, return_tokens: bool = False
    ) -> str | tuple[str, list[str]]:
        """English-specific phonemization using g2p_en."""
        if not G2P_EN_AVAILABLE:
            return self._phonemize_general(text, "en-us", return_tokens)

        # Clean text
        text = self._clean_text(text)

        try:
            # Get ARPAbet phones
            phones = self.g2p_en(text)

            # Convert to IPA if requested
            if self.config.use_ipa:
                phones = [self._arpa_to_ipa(p) for p in phones]

            phoneme_str = " ".join(phones)

            if return_tokens:
                tokens = text.split()
                return phoneme_str, tokens

            return phoneme_str

        except Exception as e:
            self.logger.error(f"English G2P failed: {e}")
            return self._fallback_phonemize(text, return_tokens)

    def _phonemize_chinese(
        self, text: str, return_tokens: bool = False
    ) -> str | tuple[str, list[str]]:
        """Chinese-specific phonemization using pypinyin."""
        if not PYPINYIN_AVAILABLE:
            return self._phonemize_general(text, "cmn", return_tokens)

        # Clean text
        text = self._clean_text(text)

        try:
            # Get pinyin
            pinyin_list = pypinyin.pinyin(text, style=pypinyin.Style.TONE3)

            # Convert pinyin to phonemes
            phonemes = []
            tokens = []

            for pinyin in pinyin_list:
                if pinyin[0]:
                    phone = self._pinyin_to_phoneme(pinyin[0])
                    phonemes.append(phone)
                    tokens.append(pinyin[0])

            phoneme_str = " ".join(phonemes)

            if return_tokens:
                return phoneme_str, tokens

            return phoneme_str

        except Exception as e:
            self.logger.error(f"Chinese G2P failed: {e}")
            return self._fallback_phonemize(text, return_tokens)

    def _phonemize_general(
        self, text: str, language: str, return_tokens: bool = False
    ) -> str | tuple[str, list[str]]:
        """General phonemization using phonemizer library."""
        if not PHONEMIZER_AVAILABLE:
            return self._fallback_phonemize(text, return_tokens)

        try:
            # Use espeak backend for IPA
            phonemes = phonemize(
                text,
                language=language,
                backend="espeak",
                separator=self.separator,
                strip=True,
                preserve_punctuation=self.config.preserve_punctuation,
                with_stress=self.config.with_stress,
            )

            if return_tokens:
                tokens = text.split()
                return phonemes, tokens

            return phonemes

        except Exception as e:
            self.logger.error(f"General G2P failed: {e}")
            return self._fallback_phonemize(text, return_tokens)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize input text."""
        # Remove extra whitespace
        text = " ".join(text.split())

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(""", "'").replace(""", "'")

        return text

    def _katakana_to_phonemes(self, katakana: str) -> str:
        """Convert katakana to IPA phonemes."""
        # Simplified mapping - should be expanded
        katakana_to_ipa = {
            "ア": "a",
            "イ": "i",
            "ウ": "ɯ",
            "エ": "e",
            "オ": "o",
            "カ": "ka",
            "キ": "ki",
            "ク": "kɯ",
            "ケ": "ke",
            "コ": "ko",
            "ガ": "ɡa",
            "ギ": "ɡi",
            "グ": "ɡɯ",
            "ゲ": "ɡe",
            "ゴ": "ɡo",
            "サ": "sa",
            "シ": "ɕi",
            "ス": "sɯ",
            "セ": "se",
            "ソ": "so",
            "ザ": "za",
            "ジ": "ʑi",
            "ズ": "zɯ",
            "ゼ": "ze",
            "ゾ": "zo",
            "タ": "ta",
            "チ": "t͡ɕi",
            "ツ": "t͡sɯ",
            "テ": "te",
            "ト": "to",
            "ダ": "da",
            "ヂ": "d͡ʑi",
            "ヅ": "d͡zɯ",
            "デ": "de",
            "ド": "do",
            "ナ": "na",
            "ニ": "ɲi",
            "ヌ": "nɯ",
            "ネ": "ne",
            "ノ": "no",
            "ハ": "ha",
            "ヒ": "çi",
            "フ": "ɸɯ",
            "ヘ": "he",
            "ホ": "ho",
            "バ": "ba",
            "ビ": "bi",
            "ブ": "bɯ",
            "ベ": "be",
            "ボ": "bo",
            "パ": "pa",
            "ピ": "pi",
            "プ": "pɯ",
            "ペ": "pe",
            "ポ": "po",
            "マ": "ma",
            "ミ": "mi",
            "ム": "mɯ",
            "メ": "me",
            "モ": "mo",
            "ヤ": "ja",
            "ユ": "jɯ",
            "ヨ": "jo",
            "ラ": "ɾa",
            "リ": "ɾi",
            "ル": "ɾɯ",
            "レ": "ɾe",
            "ロ": "ɾo",
            "ワ": "wa",
            "ヲ": "o",
            "ン": "ɴ",
            "ー": "ː",
            "ッ": "ʔ",
            "、": ",",
            "。": ".",
        }

        phonemes = []
        i = 0
        while i < len(katakana):
            # Check for two-character combinations first
            if i + 1 < len(katakana):
                two_char = katakana[i : i + 2]
                if two_char in katakana_to_ipa:
                    phonemes.append(katakana_to_ipa[two_char])
                    i += 2
                    continue

            # Single character
            char = katakana[i]
            if char in katakana_to_ipa:
                phonemes.append(katakana_to_ipa[char])
            else:
                # Keep unknown characters as-is
                phonemes.append(char)
            i += 1

        return " ".join(phonemes)

    def _arpa_to_ipa(self, arpa: str) -> str:
        """Convert ARPAbet to IPA."""
        # Basic mapping - should be expanded
        arpa_to_ipa_map = {
            "AA": "ɑ",
            "AE": "æ",
            "AH": "ʌ",
            "AO": "ɔ",
            "AW": "aʊ",
            "AY": "aɪ",
            "EH": "ɛ",
            "ER": "ɝ",
            "EY": "eɪ",
            "IH": "ɪ",
            "IY": "i",
            "OW": "oʊ",
            "OY": "ɔɪ",
            "UH": "ʊ",
            "UW": "u",
            "B": "b",
            "CH": "t͡ʃ",
            "D": "d",
            "DH": "ð",
            "F": "f",
            "G": "ɡ",
            "HH": "h",
            "JH": "d͡ʒ",
            "K": "k",
            "L": "l",
            "M": "m",
            "N": "n",
            "NG": "ŋ",
            "P": "p",
            "R": "ɹ",
            "S": "s",
            "SH": "ʃ",
            "T": "t",
            "TH": "θ",
            "V": "v",
            "W": "w",
            "Y": "j",
            "Z": "z",
            "ZH": "ʒ",
        }

        # Remove stress markers
        arpa_clean = re.sub(r"[0-2]", "", arpa)

        return arpa_to_ipa_map.get(arpa_clean, arpa_clean.lower())

    def _pinyin_to_phoneme(self, pinyin: str) -> str:
        """Convert pinyin to IPA phonemes."""
        # This is a simplified version - should be expanded
        # Remove tone numbers
        pinyin_base = re.sub(r"[1-5]", "", pinyin)

        # Basic conversion (very simplified)
        return pinyin_base

    def _fallback_phonemize(
        self, text: str, return_tokens: bool = False
    ) -> str | tuple[str, list[str]]:
        """Fallback phonemization when no backend is available."""
        # Very basic character-based approach
        phonemes = list(text.lower())
        phoneme_str = " ".join(phonemes)

        if return_tokens:
            tokens = text.split()
            return phoneme_str, tokens

        return phoneme_str

    def get_phoneme_set(self, language: Optional[Language] = None) -> set[str]:
        """Get the set of possible phonemes for a language."""
        language = language or self.config.language

        if language == Language.JAPANESE:
            # Japanese phoneme set (simplified)
            return {
                "a",
                "i",
                "ɯ",
                "e",
                "o",
                "ka",
                "ki",
                "kɯ",
                "ke",
                "ko",
                "ɡa",
                "ɡi",
                "ɡɯ",
                "ɡe",
                "ɡo",
                "sa",
                "ɕi",
                "sɯ",
                "se",
                "so",
                "za",
                "ʑi",
                "zɯ",
                "ze",
                "zo",
                "ta",
                "t͡ɕi",
                "t͡sɯ",
                "te",
                "to",
                "da",
                "d͡ʑi",
                "d͡zɯ",
                "de",
                "do",
                "na",
                "ɲi",
                "nɯ",
                "ne",
                "no",
                "ha",
                "çi",
                "ɸɯ",
                "he",
                "ho",
                "ba",
                "bi",
                "bɯ",
                "be",
                "bo",
                "pa",
                "pi",
                "pɯ",
                "pe",
                "po",
                "ma",
                "mi",
                "mɯ",
                "me",
                "mo",
                "ja",
                "jɯ",
                "jo",
                "ɾa",
                "ɾi",
                "ɾɯ",
                "ɾe",
                "ɾo",
                "wa",
                "o",
                "ɴ",
                "ː",
                "ʔ",
                ",",
                ".",
                " ",
            }
        elif language == Language.ENGLISH:
            # English phoneme set (IPA)
            return {
                "i",
                "ɪ",
                "e",
                "ɛ",
                "æ",
                "a",
                "ɑ",
                "ɔ",
                "o",
                "ʊ",
                "u",
                "ʌ",
                "ə",
                "ɝ",
                "ɚ",
                "p",
                "b",
                "t",
                "d",
                "k",
                "ɡ",
                "f",
                "v",
                "θ",
                "ð",
                "s",
                "z",
                "ʃ",
                "ʒ",
                "h",
                "t͡ʃ",
                "d͡ʒ",
                "m",
                "n",
                "ŋ",
                "l",
                "ɹ",
                "j",
                "w",
                "'",
                "ˌ",
                ",",
                ".",
                " ",
            }
        else:
            # Return a basic set for other languages
            return set("abcdefghijklmnopqrstuvwxyz .,!?'-")


# For backward compatibility
def create_phonemizer(language: str = "ja") -> MultilingualPhonemizer:
    """Create a phonemizer instance for the specified language."""
    lang_map = {
        "ja": Language.JAPANESE,
        "en": Language.ENGLISH,
        "zh": Language.CHINESE,
        "ko": Language.KOREAN,
    }

    config = PhonemeConfig(language=lang_map.get(language, Language.JAPANESE))
    return MultilingualPhonemizer(config)
