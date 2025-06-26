"""
Tests for Web UI components
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

# Import web UI modules
from src.tools.web_ui.app import main as app_main
from src.tools.web_ui.components.audio_player import AudioPlayer
from src.tools.web_ui.components.emotion_controller import EmotionController
from src.tools.web_ui.components.speaker_selector import SpeakerSelector
from src.tools.web_ui.components.style_selector import StyleSelector
from src.tools.web_ui.pages.batch_processing import render_batch_processing_page
from src.tools.web_ui.pages.model_management import render_model_management_page
from src.tools.web_ui.pages.settings import render_settings_page
from src.tools.web_ui.pages.synthesis import render_synthesis_page
from src.tools.web_ui.pages.voice_cloning import render_voice_cloning_page
from src.tools.web_ui.pages.voice_morphing import render_voice_morphing_page
from src.tools.web_ui.utils import (
    format_duration,
    get_available_models,
    load_audio,
    load_model_config,
    save_audio,
)


class TestWebUIUtils:
    """Test Web UI utility functions"""

    def test_save_audio(self):
        """Test audio saving"""
        # Create dummy audio
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            save_audio(audio, f.name, sample_rate=22050)

            # Check file exists
            assert Path(f.name).exists()

            # Load and verify
            loaded_audio, sr = load_audio(f.name)
            assert sr == 22050
            assert len(loaded_audio) == len(audio)

            # Cleanup
            Path(f.name).unlink()

    def test_load_audio(self):
        """Test audio loading"""
        # Create temporary audio file
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            save_audio(audio, f.name, sample_rate=22050)

            # Load audio
            loaded_audio, sr = load_audio(f.name)

            assert isinstance(loaded_audio, np.ndarray)
            assert sr == 22050
            assert loaded_audio.dtype == np.float32

            # Cleanup
            Path(f.name).unlink()

    def test_format_duration(self):
        """Test duration formatting"""
        assert format_duration(0) == "0:00"
        assert format_duration(59) == "0:59"
        assert format_duration(60) == "1:00"
        assert format_duration(3661) == "1:01:01"

    def test_get_available_models(self):
        """Test getting available models"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock model files
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()

            # Create model directories
            (models_dir / "model1").mkdir()
            (models_dir / "model1" / "config.json").write_text("{}")
            (models_dir / "model2").mkdir()
            (models_dir / "model2" / "config.json").write_text("{}")

            # Get models
            models = get_available_models(models_dir)

            assert len(models) == 2
            assert "model1" in models
            assert "model2" in models

    def test_load_model_config(self):
        """Test loading model configuration"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock config
            config = {"model_type": "vits", "n_speakers": 10, "sample_rate": 22050}

            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(config))

            # Load config
            loaded_config = load_model_config(config_path)

            assert loaded_config["model_type"] == "vits"
            assert loaded_config["n_speakers"] == 10
            assert loaded_config["sample_rate"] == 22050


class TestAudioPlayer:
    """Test audio player component"""

    def test_audio_player_render(self):
        """Test audio player rendering"""
        player = AudioPlayer()

        # Create dummy audio
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))

        # Mock streamlit
        with patch("streamlit.audio") as mock_audio:
            player.render(audio, sample_rate=22050)
            mock_audio.assert_called_once()

    def test_audio_player_with_controls(self):
        """Test audio player with controls"""
        player = AudioPlayer(show_download=True)

        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))

        with patch("streamlit.audio") as mock_audio:
            with patch("streamlit.download_button") as mock_download:
                player.render(audio, sample_rate=22050, label="Test Audio")

                mock_audio.assert_called_once()
                mock_download.assert_called_once()


class TestSpeakerSelector:
    """Test speaker selector component"""

    def test_speaker_selector_single(self):
        """Test single speaker selection"""
        speakers = ["Speaker 1", "Speaker 2", "Speaker 3"]
        selector = SpeakerSelector(speakers)

        with patch("streamlit.selectbox") as mock_selectbox:
            mock_selectbox.return_value = "Speaker 2"

            selected = selector.render()

            assert selected == "Speaker 2"
            mock_selectbox.assert_called_once()

    def test_speaker_selector_multiple(self):
        """Test multiple speaker selection"""
        speakers = ["Speaker 1", "Speaker 2", "Speaker 3"]
        selector = SpeakerSelector(speakers, allow_multiple=True)

        with patch("streamlit.multiselect") as mock_multiselect:
            mock_multiselect.return_value = ["Speaker 1", "Speaker 3"]

            selected = selector.render()

            assert selected == ["Speaker 1", "Speaker 3"]
            mock_multiselect.assert_called_once()


class TestEmotionController:
    """Test emotion controller component"""

    def test_emotion_controller_render(self):
        """Test emotion controller rendering"""
        controller = EmotionController()

        with patch("streamlit.select_slider") as mock_slider:
            mock_slider.return_value = "happy"

            emotion = controller.render()

            assert emotion == "happy"
            mock_slider.assert_called_once()

    def test_emotion_controller_with_intensity(self):
        """Test emotion controller with intensity"""
        controller = EmotionController(show_intensity=True)

        with patch("streamlit.select_slider") as mock_emotion:
            with patch("streamlit.slider") as mock_intensity:
                mock_emotion.return_value = "angry"
                mock_intensity.return_value = 0.8

                emotion, intensity = controller.render()

                assert emotion == "angry"
                assert intensity == 0.8


class TestStyleSelector:
    """Test style selector component"""

    def test_style_selector_render(self):
        """Test style selector rendering"""
        styles = ["normal", "energetic", "calm", "dramatic"]
        selector = StyleSelector(styles)

        with patch("streamlit.radio") as mock_radio:
            mock_radio.return_value = "energetic"

            style = selector.render()

            assert style == "energetic"
            mock_radio.assert_called_once()


class TestSynthesisPage:
    """Test synthesis page"""

    @patch("streamlit.text_area")
    @patch("streamlit.button")
    @patch("src.tools.web_ui.pages.synthesis.st")
    def test_synthesis_page_render(self, mock_st, mock_button, mock_text_area):
        """Test synthesis page rendering"""
        # Mock inputs
        mock_text_area.return_value = "こんにちは"
        mock_button.return_value = True

        # Mock TTS
        mock_tts = Mock()
        mock_tts.synthesize.return_value = np.zeros(22050)

        with patch(
            "src.tools.web_ui.pages.synthesis.load_tts_model", return_value=mock_tts
        ):
            render_synthesis_page()

            # Check TTS was called
            mock_tts.synthesize.assert_called()


class TestVoiceMorphingPage:
    """Test voice morphing page"""

    @patch("streamlit.text_area")
    @patch("streamlit.button")
    @patch("src.tools.web_ui.pages.voice_morphing.st")
    def test_voice_morphing_render(self, mock_st, mock_button, mock_text_area):
        """Test voice morphing page rendering"""
        mock_text_area.return_value = "テストテキスト"
        mock_button.return_value = True

        # Mock TTS
        mock_tts = Mock()
        mock_tts.morph_voices.return_value = np.zeros(22050)

        with patch(
            "src.tools.web_ui.pages.voice_morphing.load_tts_model",
            return_value=mock_tts,
        ):
            render_voice_morphing_page()

            # Check morphing was called
            mock_tts.morph_voices.assert_called()


class TestVoiceCloningPage:
    """Test voice cloning page"""

    @patch("streamlit.file_uploader")
    @patch("streamlit.text_area")
    @patch("streamlit.button")
    @patch("src.tools.web_ui.pages.voice_cloning.st")
    def test_voice_cloning_render(
        self, mock_st, mock_button, mock_text_area, mock_uploader
    ):
        """Test voice cloning page rendering"""
        # Mock file upload
        mock_file = Mock()
        mock_file.read.return_value = b"\x00" * 1000
        mock_uploader.return_value = mock_file

        mock_text_area.return_value = "クローンテスト"
        mock_button.return_value = True

        # Mock TTS
        mock_tts = Mock()
        mock_tts.clone_voice.return_value = np.zeros(22050)

        with patch(
            "src.tools.web_ui.pages.voice_cloning.load_tts_model", return_value=mock_tts
        ):
            render_voice_cloning_page()

            # Check cloning was called
            mock_tts.clone_voice.assert_called()


class TestBatchProcessingPage:
    """Test batch processing page"""

    @patch("streamlit.file_uploader")
    @patch("streamlit.button")
    @patch("src.tools.web_ui.pages.batch_processing.st")
    def test_batch_processing_render(self, mock_st, mock_button, mock_uploader):
        """Test batch processing page rendering"""
        # Mock CSV file
        csv_content = "text,speaker_id\nこんにちは,0\nありがとう,1"
        mock_file = Mock()
        mock_file.read.return_value = csv_content.encode()
        mock_uploader.return_value = mock_file

        mock_button.return_value = True

        # Mock TTS
        mock_tts = Mock()
        mock_tts.batch_synthesize.return_value = [np.zeros(22050), np.zeros(22050)]

        with patch(
            "src.tools.web_ui.pages.batch_processing.load_tts_model",
            return_value=mock_tts,
        ):
            render_batch_processing_page()

            # Check batch synthesis was called
            mock_tts.batch_synthesize.assert_called()


class TestModelManagementPage:
    """Test model management page"""

    @patch("streamlit.selectbox")
    @patch("streamlit.button")
    @patch("src.tools.web_ui.pages.model_management.st")
    def test_model_management_render(self, mock_st, mock_button, mock_selectbox):
        """Test model management page rendering"""
        mock_selectbox.return_value = "model1"

        with patch(
            "src.tools.web_ui.pages.model_management.get_available_models",
            return_value=["model1", "model2"],
        ):
            render_model_management_page()

            # Check selectbox was called
            mock_selectbox.assert_called()


class TestSettingsPage:
    """Test settings page"""

    @patch("streamlit.number_input")
    @patch("streamlit.checkbox")
    @patch("streamlit.button")
    @patch("src.tools.web_ui.pages.settings.st")
    def test_settings_render(self, mock_st, mock_button, mock_checkbox, mock_number):
        """Test settings page rendering"""
        mock_number.return_value = 22050
        mock_checkbox.return_value = True
        mock_button.return_value = True

        render_settings_page()

        # Check settings inputs were rendered
        mock_number.assert_called()
        mock_checkbox.assert_called()


class TestAppIntegration:
    """Test full app integration"""

    @patch("streamlit.set_page_config")
    @patch("streamlit.sidebar")
    def test_app_main(self, mock_sidebar, mock_config):
        """Test main app function"""
        # Mock sidebar navigation
        mock_sidebar.radio.return_value = "音声合成"

        # Create app test instance
        at = AppTest.from_function(app_main)

        # Run app
        at.run()

        # Check app ran without errors
        assert not at.exception


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
