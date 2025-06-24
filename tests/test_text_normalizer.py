"""
Unit tests for Japanese text normalization.

Tests the text normalization pipeline including number conversion,
symbol handling, and various Japanese-specific text processing.
"""

from typing import List, Tuple
from unittest.mock import MagicMock, patch

import pytest

from src.frontend.text_normalizer import JapaneseTextNormalizer


class TestJapaneseTextNormalizer:
    """Test suite for Japanese text normalization."""

    @pytest.fixture
    def normalizer(self) -> JapaneseTextNormalizer:
        """Create JapaneseTextNormalizer instance."""
        # Mock MeCab if not available
        with patch("src.frontend.text_normalizer.MeCab") as mock_mecab:
            mock_tagger = MagicMock()
            mock_mecab.Tagger.return_value = mock_tagger
            normalizer = JapaneseTextNormalizer()
        return normalizer

    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("123", "いちにさん"),
            ("2024年", "にせんにじゅうよねん"),
            ("3.14", "さんてんいちよん"),
            ("1,000", "せん"),
            ("¥100", "ひゃくえん"),
        ],
    )
    def test_number_normalization(
        self, normalizer: JapaneseTextNormalizer, input_text: str, expected: str
    ) -> None:
        """Test number to Japanese conversion."""
        result = normalizer.normalize_numbers(input_text)
        assert expected in result

    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("AI", "エーアイ"),
            ("USB", "ユーエスビー"),
            ("Wi-Fi", "ワイファイ"),
            ("OK", "オーケー"),
        ],
    )
    def test_alphabet_conversion(
        self, normalizer: JapaneseTextNormalizer, input_text: str, expected: str
    ) -> None:
        """Test alphabet to katakana conversion."""
        result = normalizer.convert_alphabet_to_kana(input_text)
        assert result == expected

    @pytest.mark.parametrize(
        "input_text,expected_contains",
        [
            ("おはよう！", "おはよう"),
            ("そうですか？", "そうですか"),
            ("すごい〜", "すごい"),
            ("本当に…", "本当に"),
            ("「こんにちは」", "こんにちは"),
        ],
    )
    def test_symbol_normalization(
        self,
        normalizer: JapaneseTextNormalizer,
        input_text: str,
        expected_contains: str,
    ) -> None:
        """Test symbol normalization."""
        result = normalizer.normalize_symbols(input_text)
        assert expected_contains in result

    def test_full_text_normalization(self, normalizer: JapaneseTextNormalizer) -> None:
        """Test complete text normalization pipeline."""
        input_text = "今日は2024年12月25日です。気温は-5度でした。"
        result = normalizer.normalize(input_text)

        # Check that numbers are converted
        assert "2024" not in result
        assert "12" not in result
        assert "25" not in result
        assert "-5" not in result

        # Check that Japanese text is preserved
        assert "今日は" in result
        assert "です" in result

    def test_mixed_language_text(self, normalizer: JapaneseTextNormalizer) -> None:
        """Test normalization of mixed Japanese-English text."""
        input_text = "私はAIを使って日本語のTTSを作っています。"
        result = normalizer.normalize(input_text)

        assert "AI" not in result  # Should be converted to katakana
        assert "TTS" not in result  # Should be converted to katakana
        assert "私は" in result
        assert "を使って" in result

    def test_special_readings(self, normalizer: JapaneseTextNormalizer) -> None:
        """Test special reading patterns."""
        test_cases = [
            ("今日", ["きょう", "こんにち"]),  # Multiple possible readings
            ("明日", ["あした", "あす", "みょうにち"]),
            ("一人", ["ひとり", "いちにん"]),
        ]

        for text, possible_readings in test_cases:
            result = normalizer.get_reading(text)
            assert any(reading in result for reading in possible_readings)

    def test_empty_input(self, normalizer: JapaneseTextNormalizer) -> None:
        """Test handling of empty input."""
        assert normalizer.normalize("") == ""
        assert normalizer.normalize_numbers("") == ""
        assert normalizer.convert_alphabet_to_kana("") == ""

    def test_whitespace_handling(self, normalizer: JapaneseTextNormalizer) -> None:
        """Test whitespace normalization."""
        input_text = "こんにちは　　　世界"  # Full-width spaces
        result = normalizer.normalize(input_text)
        # Should normalize to single spaces
        assert "　　　" not in result

    @pytest.mark.parametrize(
        "input_char,expected_type",
        [
            ("あ", "hiragana"),
            ("ア", "katakana"),
            ("漢", "kanji"),
            ("A", "alphabet"),
            ("1", "number"),
            ("！", "symbol"),
        ],
    )
    def test_character_type_detection(
        self, normalizer: JapaneseTextNormalizer, input_char: str, expected_type: str
    ) -> None:
        """Test character type detection."""
        char_type = normalizer.get_char_type(input_char)
        assert char_type == expected_type


class TestTextNormalizerEdgeCases:
    """Test edge cases for text normalizer."""

    @pytest.fixture
    def normalizer(self) -> JapaneseTextNormalizer:
        """Create normalizer with mocked MeCab."""
        with patch("src.frontend.text_normalizer.MeCab") as mock_mecab:
            mock_tagger = MagicMock()
            mock_mecab.Tagger.return_value = mock_tagger
            normalizer = JapaneseTextNormalizer()
        return normalizer

    def test_very_long_text(self, normalizer: JapaneseTextNormalizer) -> None:
        """Test handling of very long text."""
        long_text = "これは長いテキストです。" * 1000
        result = normalizer.normalize(long_text)
        assert len(result) > 0
        assert "これは長いテキストです" in result

    def test_unicode_normalization(self, normalizer: JapaneseTextNormalizer) -> None:
        """Test Unicode normalization (NFKC)."""
        input_text = "ｶﾀｶﾅ"  # Half-width katakana
        result = normalizer.normalize(input_text)
        assert "カタカナ" in result  # Should be full-width

    def test_emoji_handling(self, normalizer: JapaneseTextNormalizer) -> None:
        """Test emoji handling."""
        input_text = "こんにちは😊楽しいです🎉"
        result = normalizer.normalize(input_text)
        # Emojis should be removed or converted to descriptive text
        assert "😊" not in result
        assert "🎉" not in result

    def test_url_handling(self, normalizer: JapaneseTextNormalizer) -> None:
        """Test URL handling."""
        input_text = "詳細はhttps://example.comをご覧ください。"
        result = normalizer.normalize(input_text)
        # URL should be handled appropriately
        assert "https://" not in result or "URL" in result

    def test_date_formats(self, normalizer: JapaneseTextNormalizer) -> None:
        """Test various date format normalizations."""
        date_formats = [
            "2024/12/25",
            "2024-12-25",
            "令和6年12月25日",
            "R6.12.25",
        ]

        for date in date_formats:
            result = normalizer.normalize(date)
            # Should contain Japanese reading of the date
            assert any(char in "年月日" for char in result)


class TestTextNormalizerPerformance:
    """Performance tests for text normalizer."""

    @pytest.fixture
    def normalizer(self) -> JapaneseTextNormalizer:
        """Create normalizer for performance testing."""
        with patch("src.frontend.text_normalizer.MeCab") as mock_mecab:
            mock_tagger = MagicMock()
            # Mock parse to return quickly
            mock_tagger.parse.return_value = (
                "テスト\t名詞,一般,*,*,*,*,テスト,テスト,テスト\nEOS\n"
            )
            mock_mecab.Tagger.return_value = mock_tagger
            normalizer = JapaneseTextNormalizer()
        return normalizer

    @pytest.mark.benchmark
    def test_normalization_speed(
        self, normalizer: JapaneseTextNormalizer, benchmark
    ) -> None:
        """Benchmark normalization speed."""
        test_text = "本日は2024年12月25日、気温は15度です。AIの発展により、日本語TTSも進化しています。"

        result = benchmark(normalizer.normalize, test_text)
        assert len(result) > 0

    def test_batch_normalization(self, normalizer: JapaneseTextNormalizer) -> None:
        """Test batch text normalization."""
        texts = [
            "こんにちは",
            "今日はいい天気です",
            "明日は雨が降るでしょう",
            "来週の予定を確認してください",
        ] * 10

        results = [normalizer.normalize(text) for text in texts]
        assert len(results) == len(texts)
        assert all(len(result) > 0 for result in results)
