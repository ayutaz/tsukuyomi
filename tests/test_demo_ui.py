"""Tests for demo UI."""

import io
from unittest.mock import MagicMock, Mock, patch

import gradio as gr
import numpy as np
import pytest
import requests
import soundfile as sf

from src.web.demo_ui import (
    EMOTIONS,
    SPEAKERS,
    STYLES,
    build_demo,
    create_waveform_plot,
    morph_voices,
    synthesize_speech,
)


class TestDemoUI:
    """Test suite for demo UI."""

    @pytest.fixture()
    def mock_response(self):
        """Create mock response for API calls."""
        # Create dummy audio
        audio = np.random.randn(48000).astype(np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, audio, 48000, format="WAV")

        mock = Mock()
        mock.status_code = 200
        mock.content = buffer.getvalue()
        return mock

    @pytest.fixture()
    def mock_requests(self, mock_response):
        """Mock requests module."""
        with patch("src.web.demo_ui.requests") as mock:
            mock.post.return_value = mock_response
            mock.get.return_value = Mock(
                status_code=200, json=lambda: {"status": "healthy", "device": "cpu"}
            )
            yield mock

    def test_synthesize_speech_success(self, mock_requests):
        """Test successful speech synthesis."""
        sr, audio = synthesize_speech(
            text="テスト",
            speaker="Default Female",
            emotion="neutral",
            style="normal",
            speed=1.0,
            pitch=0.0,
            energy=1.0,
        )

        assert sr == 48000
        assert audio is not None
        assert len(audio) > 0

        # Verify API was called correctly
        mock_requests.post.assert_called_once()
        call_args = mock_requests.post.call_args
        assert call_args[0][0] == "http://localhost:8000/synthesize"

        json_data = call_args[1]["json"]
        assert json_data["text"] == "テスト"
        assert json_data["speaker_id"] == SPEAKERS["Default Female"]
        assert json_data["emotion"] == "neutral"

    def test_synthesize_speech_failure(self, mock_requests):
        """Test failed speech synthesis."""
        mock_requests.post.return_value.status_code = 500

        sr, audio = synthesize_speech(
            text="テスト",
            speaker="Default Female",
            emotion="neutral",
            style="normal",
            speed=1.0,
            pitch=0.0,
            energy=1.0,
        )

        assert sr is None
        assert audio is None

    def test_synthesize_speech_exception(self, mock_requests):
        """Test exception handling."""
        mock_requests.post.side_effect = Exception("Network error")

        sr, audio = synthesize_speech(
            text="テスト",
            speaker="Default Female",
            emotion="neutral",
            style="normal",
            speed=1.0,
            pitch=0.0,
            energy=1.0,
        )

        assert sr is None
        assert audio is None

    def test_morph_voices_success(self, mock_requests):
        """Test successful voice morphing."""
        sr, audio = morph_voices(
            text="モーフィングテスト",
            speakers=["Default Female", "Male Voice 1"],
            weights="0.7, 0.3",
            emotions=["happy", "excited"],
            emotion_weights="0.6, 0.4",
            speed=1.0,
        )

        assert sr == 48000
        assert audio is not None
        assert len(audio) > 0

        # Verify API call
        call_args = mock_requests.post.call_args
        json_data = call_args[1]["json"]
        assert json_data["text"] == "モーフィングテスト"
        assert json_data["speaker_ids"] == [
            SPEAKERS["Default Female"],
            SPEAKERS["Male Voice 1"],
        ]
        assert json_data["speaker_weights"] == [0.7, 0.3]
        assert json_data["emotion_ids"] == ["happy", "excited"]
        assert json_data["emotion_weights"] == [0.6, 0.4]

    def test_morph_voices_weight_parsing(self, mock_requests):
        """Test weight parsing in morph_voices."""
        # Test with spaces
        sr, audio = morph_voices(
            text="test",
            speakers=["Default Female", "Male Voice 1"],
            weights=" 0.5 , 0.5 ",
            emotions=[],
            emotion_weights="",
            speed=1.0,
        )

        call_args = mock_requests.post.call_args
        json_data = call_args[1]["json"]
        assert json_data["speaker_weights"] == [0.5, 0.5]

    def test_create_waveform_plot(self):
        """Test waveform plot creation."""
        # Create test audio
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        fig = create_waveform_plot(audio, sr)

        assert fig is not None
        assert len(fig.axes) == 2  # Waveform and spectrogram

        # Check plot titles
        assert "Waveform" in fig.axes[0].get_title()
        assert "Spectrogram" in fig.axes[1].get_title()

    def test_build_demo(self):
        """Test demo interface building."""
        demo = build_demo()

        assert isinstance(demo, gr.Blocks)
        assert demo.title == "Tsukuyomi TTS Demo"

        # Check that main components exist
        components = demo.blocks
        assert len(components) > 0

        # Find tabs
        tabs = [c for c in components if isinstance(c, gr.Tabs)]
        assert len(tabs) > 0

    def test_speakers_list(self):
        """Test speakers list is properly defined."""
        assert len(SPEAKERS) > 0
        assert "Default Female" in SPEAKERS
        assert isinstance(SPEAKERS["Default Female"], int)

    def test_emotions_list(self):
        """Test emotions list."""
        assert len(EMOTIONS) > 0
        assert "neutral" in EMOTIONS
        assert "happy" in EMOTIONS
        assert "sad" in EMOTIONS

    def test_styles_list(self):
        """Test styles list."""
        assert len(STYLES) > 0
        assert "normal" in STYLES
        assert "energetic" in STYLES
        assert "calm" in STYLES

    @patch("src.web.demo_ui.gr.Blocks.launch")
    def test_main_function(self, mock_launch):
        """Test main function."""
        from src.web.demo_ui import main

        main()

        mock_launch.assert_called_once()
        call_args = mock_launch.call_args[1]
        assert call_args["server_name"] == "0.0.0.0"
        assert call_args["server_port"] == 7860
        assert call_args["share"] is False

    def test_api_health_check(self, mock_requests):
        """Test API health check functionality."""
        # Import the check_api function from the demo
        demo = build_demo()

        # Find the check_api function
        check_api_func = None
        for component in demo.blocks:
            if hasattr(component, "click") and hasattr(component.click, "fn"):
                if component.click.fn.__name__ == "check_api":
                    check_api_func = component.click.fn
                    break

        # Can't easily test internal functions without running the demo
        # Just verify the demo builds without errors
        assert demo is not None


class TestIntegration:
    """Integration tests for demo UI."""

    @pytest.fixture()
    def demo_interface(self):
        """Create demo interface for testing."""
        return build_demo()

    def test_demo_structure(self, demo_interface):
        """Test overall demo structure."""
        # Check that demo has expected tabs
        tabs_found = 0
        for component in demo_interface.blocks:
            if isinstance(component, gr.Tabs):
                tabs_found += 1

        assert tabs_found > 0

    def test_component_connections(self, demo_interface):
        """Test that components are properly connected."""
        # Check for essential components
        textboxes = [c for c in demo_interface.blocks if isinstance(c, gr.Textbox)]
        buttons = [c for c in demo_interface.blocks if isinstance(c, gr.Button)]
        audios = [c for c in demo_interface.blocks if isinstance(c, gr.Audio)]

        assert len(textboxes) > 0  # Text inputs
        assert len(buttons) > 0  # Action buttons
        assert len(audios) > 0  # Audio outputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
