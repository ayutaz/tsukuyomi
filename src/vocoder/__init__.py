"""
Vocoder package for Tsukuyomi TTS

Contains neural vocoders for converting acoustic features to waveforms.
"""

from .bigvgan import BigVGANVocoder

__all__ = ["BigVGANVocoder"]
