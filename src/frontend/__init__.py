"""
Frontend package for Tsukuyomi TTS

Contains text processing, normalization, and phonemization components.
"""

from .text_normalizer import JapaneseTextNormalizer

__all__ = ["JapaneseTextNormalizer"]