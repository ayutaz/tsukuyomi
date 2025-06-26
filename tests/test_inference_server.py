"""Tests for inference server."""

import base64
import io
import json
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from src.server.inference_server import (
    InferenceServer,
    MorphRequest,
    ServerConfig,
    TTSRequest,
)


class TestInferenceServer:
    """Test suite for inference server."""

    @pytest.fixture()
    def config(self):
        """Create test configuration."""
        return ServerConfig(
            cache_enabled=False,  # Disable cache for tests
            rate_limit=1000,  # High limit for tests
            enable_gpu=False,  # CPU for tests
        )

    @pytest.fixture()
    def mock_tts(self):
        """Create mock TTS system."""
        mock = Mock()
        mock.device = "cpu"

        # Mock synthesize method
        mock.synthesize.return_value = np.random.randn(48000)  # 1 second audio

        # Mock morph_voices method
        mock.morph_voices.return_value = np.random.randn(48000)

        # Mock clone_voice method
        mock.clone_voice.return_value = np.random.randn(48000)

        return mock

    @pytest.fixture()
    def server(self, config, mock_tts):
        """Create server instance with mocked TTS."""
        with patch("src.server.inference_server.TsukuyomiTTS", return_value=mock_tts):
            server = InferenceServer(config)
            return server

    @pytest.fixture()
    def client(self, server):
        """Create test client."""
        return TestClient(server.app)

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Tsukuyomi TTS API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["device"] == "cpu"
        assert data["models_loaded"] is True
        assert "metrics" in data

    def test_synthesize_basic(self, client):
        """Test basic synthesis."""
        request_data = {
            "text": "こんにちは",
            "speaker_id": 0,
            "emotion": "neutral",
            "style": "normal",
            "speed": 1.0,
            "pitch_shift": 0.0,
            "energy": 1.0,
            "output_format": "wav",
        }

        response = client.post("/synthesize", json=request_data)
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"

        # Check audio data
        audio_data = response.content
        audio, sr = sf.read(io.BytesIO(audio_data))
        assert len(audio) > 0
        assert sr == 48000

    def test_synthesize_with_parameters(self, client):
        """Test synthesis with various parameters."""
        test_cases = [
            {"emotion": "happy", "style": "energetic"},
            {"speed": 1.5, "pitch_shift": 2.0},
            {"energy": 0.8, "speaker_id": 5},
        ]

        for params in test_cases:
            request_data = {
                "text": "テスト",
                "speaker_id": params.get("speaker_id", 0),
                "emotion": params.get("emotion", "neutral"),
                "style": params.get("style", "normal"),
                "speed": params.get("speed", 1.0),
                "pitch_shift": params.get("pitch_shift", 0.0),
                "energy": params.get("energy", 1.0),
                "output_format": "wav",
            }

            response = client.post("/synthesize", json=request_data)
            assert response.status_code == 200

    def test_synthesize_validation(self, client):
        """Test request validation."""
        # Test invalid speed
        request_data = {"text": "test", "speed": 3.0}  # Out of range
        response = client.post("/synthesize", json=request_data)
        assert response.status_code == 422

        # Test invalid pitch
        request_data = {"text": "test", "pitch_shift": 15.0}  # Out of range
        response = client.post("/synthesize", json=request_data)
        assert response.status_code == 422

    def test_text_length_limit(self, client, server):
        """Test text length validation."""
        long_text = "あ" * (server.config.max_text_length + 1)
        request_data = {"text": long_text, "speaker_id": 0}

        response = client.post("/synthesize", json=request_data)
        assert response.status_code == 400
        assert "Text too long" in response.json()["detail"]

    def test_morph_voices(self, client):
        """Test voice morphing endpoint."""
        request_data = {
            "text": "声をモーフィングします",
            "speaker_ids": [0, 1, 2],
            "speaker_weights": [0.5, 0.3, 0.2],
            "speed": 1.0,
            "output_format": "wav",
        }

        response = client.post("/morph", json=request_data)
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"

    def test_morph_with_emotions(self, client):
        """Test morphing with emotions."""
        request_data = {
            "text": "感情もブレンド",
            "speaker_ids": [0, 1],
            "speaker_weights": [0.6, 0.4],
            "emotion_ids": ["happy", "excited"],
            "emotion_weights": [0.7, 0.3],
            "speed": 1.0,
            "output_format": "wav",
        }

        response = client.post("/morph", json=request_data)
        assert response.status_code == 200

    def test_batch_synthesis(self, client):
        """Test batch synthesis endpoint."""
        request_data = {
            "requests": [
                {"text": "最初のテキスト", "speaker_id": 0, "emotion": "neutral"},
                {"text": "二番目のテキスト", "speaker_id": 1, "emotion": "happy"},
            ],
            "parallel": True,
        }

        response = client.post("/batch", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2

        # Check each result
        for i, result in enumerate(data["results"]):
            assert result["index"] == i
            assert "audio" in result
            assert result["format"] == "wav"

            # Decode and verify audio
            audio_bytes = base64.b64decode(result["audio"])
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            assert len(audio) > 0

    def test_batch_size_limit(self, client, server):
        """Test batch size limit."""
        # Create requests exceeding limit
        requests = [
            {"text": f"Text {i}", "speaker_id": 0}
            for i in range(server.config.max_batch_size + 1)
        ]

        request_data = {"requests": requests, "parallel": True}

        response = client.post("/batch", json=request_data)
        assert response.status_code == 400
        assert "Batch size exceeds maximum" in response.json()["detail"]

    def test_speakers_endpoint(self, client):
        """Test speakers listing."""
        response = client.get("/speakers")
        assert response.status_code == 200

        data = response.json()
        assert "speakers" in data
        assert len(data["speakers"]) > 0

        # Check speaker format
        speaker = data["speakers"][0]
        assert "id" in speaker
        assert "name" in speaker
        assert "gender" in speaker
        assert "language" in speaker

    def test_emotions_endpoint(self, client):
        """Test emotions listing."""
        response = client.get("/emotions")
        assert response.status_code == 200

        data = response.json()
        assert "emotions" in data
        assert "neutral" in data["emotions"]
        assert "happy" in data["emotions"]

    def test_styles_endpoint(self, client):
        """Test styles listing."""
        response = client.get("/styles")
        assert response.status_code == 200

        data = response.json()
        assert "styles" in data
        assert "normal" in data["styles"]
        assert "energetic" in data["styles"]

    @pytest.mark.asyncio()
    async def test_rate_limiting(self, client, server):
        """Test rate limiting."""
        # Set low rate limit for testing
        server.config.rate_limit = 2
        server.request_history.clear()

        # Make requests up to limit
        for i in range(2):
            response = client.post("/synthesize", json={"text": f"test{i}"})
            assert response.status_code == 200

        # Next request should be rate limited
        response = client.post("/synthesize", json={"text": "test3"})
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]

    def test_metrics_tracking(self, server, client):
        """Test metrics are tracked correctly."""
        # Reset metrics
        server.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_latency": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Make successful request
        response = client.post("/synthesize", json={"text": "test"})
        assert response.status_code == 200

        assert server.metrics["total_requests"] == 1
        assert server.metrics["successful_requests"] == 1
        assert server.metrics["failed_requests"] == 0
        assert server.metrics["average_latency"] > 0

    def test_error_handling(self, client, mock_tts):
        """Test error handling."""
        # Make TTS raise an exception
        mock_tts.synthesize.side_effect = Exception("TTS Error")

        response = client.post("/synthesize", json={"text": "test"})
        assert response.status_code == 500
        assert "TTS Error" in response.json()["detail"]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_initialization(self):
        """Test GPU initialization when available."""
        config = ServerConfig(enable_gpu=True)
        with patch("src.server.inference_server.TsukuyomiTTS") as mock_tts_class:
            server = InferenceServer(config)
            mock_tts_class.assert_called_with(device="cuda", checkpoint_dir=None)


class TestCaching:
    """Test caching functionality."""

    @pytest.fixture()
    def redis_mock(self):
        """Mock Redis client."""
        mock = Mock()
        mock.exists.return_value = False
        mock.get.return_value = None
        mock.setex.return_value = True
        return mock

    @pytest.fixture()
    def server_with_cache(self, mock_tts, redis_mock):
        """Create server with mocked cache."""
        config = ServerConfig(cache_enabled=True)
        with patch("src.server.inference_server.TsukuyomiTTS", return_value=mock_tts):
            with patch(
                "src.server.inference_server.redis.Redis", return_value=redis_mock
            ):
                server = InferenceServer(config)
                server.cache = redis_mock
                return server

    @pytest.fixture()
    def client_with_cache(self, server_with_cache):
        """Create test client with cache enabled."""
        return TestClient(server_with_cache.app)

    def test_cache_miss(self, client_with_cache, server_with_cache):
        """Test cache miss scenario."""
        response = client_with_cache.post("/synthesize", json={"text": "test"})
        assert response.status_code == 200

        # Check cache was queried
        assert server_with_cache.cache.exists.called
        assert server_with_cache.metrics["cache_misses"] == 1
        assert server_with_cache.metrics["cache_hits"] == 0

    def test_cache_hit(self, client_with_cache, server_with_cache):
        """Test cache hit scenario."""
        # Setup cache to return data
        cached_audio = np.random.randn(48000).astype(np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, cached_audio, 48000, format="WAV")
        cached_bytes = buffer.getvalue()

        server_with_cache.cache.exists.return_value = True
        server_with_cache.cache.get.return_value = cached_bytes.decode("latin-1")

        response = client_with_cache.post("/synthesize", json={"text": "test"})
        assert response.status_code == 200

        # Verify cache hit
        assert server_with_cache.metrics["cache_hits"] == 1
        assert server_with_cache.metrics["cache_misses"] == 0

        # TTS should not be called on cache hit
        assert not server_with_cache.tts.synthesize.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
