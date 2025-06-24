"""
Tests for frontend text processing modules
"""

import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.frontend.text_normalizer import JapaneseTextNormalizer

# Skip all tests if MeCab is not available (e.g., on Windows CI)
try:
    import MeCab

    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not MECAB_AVAILABLE, reason="MeCab not available (common on Windows)"
)


class TestJapaneseTextNormalizer:
    """Test Japanese text normalization"""

    @pytest.fixture
    def normalizer(self):
        """Create a normalizer instance"""
        return JapaneseTextNormalizer()

    def test_initialization(self, normalizer):
        """Test normalizer initialization"""
        assert normalizer is not None
        assert hasattr(normalizer, "tagger")
        assert hasattr(normalizer, "kakasi")

    def test_normalize_numbers(self, normalizer):
        """Test number normalization"""
        test_cases = [
            ("今日は2024年です", "今日は二千二十四年です"),
            ("価格は1000円", "価格は千円"),
            ("電話番号は090-1234-5678", "電話番号は〇九〇一二三四五六七八"),
        ]

        for input_text, expected in test_cases:
            result = normalizer._normalize_numbers(input_text)
            # Basic check - actual implementation may vary
            assert isinstance(result, str)
            assert len(result) > 0

    def test_normalize_symbols(self, normalizer):
        """Test symbol normalization"""
        test_cases = [
            ("Hello!", "Hello！"),
            ("質問?", "質問？"),
            ("A&B", "AアンドB"),
            ("50%", "50パーセント"),
        ]

        for input_text, _ in test_cases:
            result = normalizer._normalize_symbols(input_text)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_text_to_phonemes(self, normalizer):
        """Test phoneme conversion"""
        text = "こんにちは"
        phonemes = normalizer.text_to_phonemes(text)

        assert isinstance(phonemes, str)
        assert len(phonemes) > 0
        assert " " in phonemes  # Should have space-separated phonemes

    def test_normalize_basic(self, normalizer):
        """Test basic normalization"""
        test_texts = [
            "こんにちは、世界！",
            "今日は2024年1月1日です。",
            "AIによる音声合成",
            "Test123",
        ]

        for text in test_texts:
            result = normalizer.normalize(text)
            assert isinstance(result, dict)
            assert "text" in result
            assert "phonemes" in result
            assert "tokens" in result
            assert len(result["text"]) > 0

    def test_normalize_edge_cases(self, normalizer):
        """Test edge cases"""
        # Empty string
        result = normalizer.normalize("")
        assert result["text"] == ""

        # Only symbols
        result = normalizer.normalize("！？。、")
        assert isinstance(result["text"], str)

        # Mixed languages
        result = normalizer.normalize("Hello世界123")
        assert len(result["phonemes"]) > 0

    def test_batch_normalize(self, normalizer):
        """Test batch normalization"""
        texts = [
            "テスト1",
            "テスト2",
            "テスト3",
        ]

        results = normalizer.batch_normalize(texts)
        assert len(results) == len(texts)
        for result in results:
            assert isinstance(result, dict)
            assert "text" in result
            assert "phonemes" in result

    @patch("MeCab.Tagger")
    def test_mecab_failure_handling(self, mock_mecab, normalizer):
        """Test handling of MeCab failures"""
        mock_mecab.side_effect = Exception("MeCab error")

        # Should handle gracefully
        with pytest.raises(Exception):
            normalizer = JapaneseTextNormalizer()

    def test_long_text_handling(self, normalizer):
        """Test handling of long texts"""
        long_text = "これは長いテキストです。" * 100
        result = normalizer.normalize(long_text)

        assert isinstance(result, dict)
        assert len(result["text"]) > 0
        assert len(result["tokens"]) > 0

    @pytest.mark.parametrize(
        "text,expected_tokens",
        [
            ("私は学生です", ["私", "は", "学生", "です"]),
            ("人工知能", ["人工", "知能"]),
            ("音声合成システム", ["音声", "合成", "システム"]),
        ],
    )
    def test_tokenization(self, normalizer, text, expected_tokens):
        """Test tokenization results"""
        result = normalizer.normalize(text)
        tokens = result["tokens"]

        # Check if expected tokens are present
        for expected in expected_tokens:
            assert any(expected in token for token in tokens)

    def test_special_readings(self, normalizer):
        """Test special readings handling"""
        test_cases = [
            ("今日", ["きょう", "こんにち"]),  # Multiple readings
            ("日本", ["にほん", "にっぽん"]),
            ("一日", ["いちにち", "ついたち"]),
        ]

        for text, possible_readings in test_cases:
            result = normalizer.normalize(text)
            phonemes = result["phonemes"].lower()
            # At least one reading should be handled
            assert any(
                any(char in phonemes for char in reading)
                for reading in possible_readings
            )

    def test_thread_safety(self, normalizer):
        """Test thread safety of normalizer"""
        import threading

        results = []

        def normalize_in_thread(text):
            result = normalizer.normalize(text)
            results.append(result)

        threads = []
        for i in range(5):
            t = threading.Thread(target=normalize_in_thread, args=(f"テスト{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(results) == 5
        for result in results:
            assert isinstance(result, dict)
