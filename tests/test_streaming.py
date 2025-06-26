"""
Tests for streaming and real-time inference
"""

import asyncio
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from src.inference.realtime_streaming import (
    ChunkedAttention,
    OptimizedVocoder,
    StreamingConfig,
    StreamingPipeline,
    StreamingVITS,
    measure_latency,
)
from src.models.bigvgan import BigVGANv2
from src.models.vits import VITS


class TestChunkedAttention:
    """Test chunked attention mechanism"""

    def test_chunked_attention_creation(self):
        """Test chunked attention creation"""
        attention = ChunkedAttention(
            embed_dim=256, num_heads=8, chunk_size=64, memory_size=128
        )

        assert attention.embed_dim == 256
        assert attention.num_heads == 8
        assert attention.chunk_size == 64
        assert attention.memory_size == 128
        assert attention.memory_buffer.shape == (1, 128, 256)

    def test_forward_without_memory(self):
        """Test forward pass without memory"""
        attention = ChunkedAttention(embed_dim=256, num_heads=8, chunk_size=64)

        # Create input
        x = torch.randn(2, 64, 256)

        # Forward pass
        output, memory = attention(x, use_memory=False)

        assert output.shape == x.shape
        assert memory.shape == attention.memory_buffer.shape

    def test_forward_with_memory(self):
        """Test forward pass with memory"""
        attention = ChunkedAttention(
            embed_dim=256, num_heads=8, chunk_size=64, memory_size=128
        )

        # Process multiple chunks
        x1 = torch.randn(2, 64, 256)
        x2 = torch.randn(2, 64, 256)

        # First chunk
        output1, _ = attention(x1, use_memory=True)
        assert attention.memory_ptr > 0

        # Second chunk (should use memory from first)
        output2, _ = attention(x2, use_memory=True)

        assert output1.shape == x1.shape
        assert output2.shape == x2.shape

    def test_memory_reset(self):
        """Test memory reset"""
        attention = ChunkedAttention(embed_dim=256, num_heads=8, chunk_size=64)

        # Process chunk
        x = torch.randn(2, 64, 256)
        attention(x, use_memory=True)

        assert attention.memory_ptr > 0

        # Reset memory
        attention.reset_memory()

        assert attention.memory_ptr == 0
        assert torch.all(attention.memory_buffer == 0)


class TestStreamingVITS:
    """Test streaming VITS model"""

    @pytest.fixture()
    def streaming_vits(self):
        """Create streaming VITS model"""
        base_vits = VITS(
            n_vocab=100,
            n_speakers=2,
            hidden_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=2,
        )

        streaming_vits = StreamingVITS(
            base_vits=base_vits, chunk_size=64, overlap_size=16
        )

        return streaming_vits

    def test_streaming_vits_creation(self, streaming_vits):
        """Test streaming VITS creation"""
        assert streaming_vits.chunk_size == 64
        assert streaming_vits.overlap_size == 16
        assert streaming_vits.encoder_state is None
        assert streaming_vits.decoder_state is None
        assert streaming_vits.prev_overlap is None

    def test_encode_chunk(self, streaming_vits):
        """Test chunk encoding"""
        # Create text chunk
        text_chunk = torch.randint(0, 100, (1, 20))
        speaker_id = torch.tensor([0])

        # Encode chunk
        encoded = streaming_vits.encode_chunk(text_chunk, speaker_id)

        assert isinstance(encoded, torch.Tensor)
        assert encoded.dim() == 3

    def test_decode_chunk(self, streaming_vits):
        """Test chunk decoding"""
        # Create encoded chunk
        encoded_chunk = torch.randn(1, 192, 10)

        # Decode chunk
        mel_chunk = streaming_vits.decode_chunk(encoded_chunk)

        assert isinstance(mel_chunk, torch.Tensor)
        assert mel_chunk.dim() == 3

    def test_overlap_processing(self, streaming_vits):
        """Test overlap processing between chunks"""
        # Process two consecutive chunks
        encoded1 = torch.randn(1, 192, 10)
        encoded2 = torch.randn(1, 192, 10)

        # First chunk
        mel1 = streaming_vits.decode_chunk(encoded1)
        assert streaming_vits.prev_overlap is not None

        # Second chunk (should apply crossfade)
        mel2 = streaming_vits.decode_chunk(encoded2)

        assert mel1.shape == encoded1.shape
        assert mel2.shape == encoded2.shape

    def test_state_reset(self, streaming_vits):
        """Test state reset"""
        # Process chunk to create state
        encoded = torch.randn(1, 192, 10)
        streaming_vits.decode_chunk(encoded)

        assert streaming_vits.prev_overlap is not None

        # Reset state
        streaming_vits.reset_state()

        assert streaming_vits.encoder_state is None
        assert streaming_vits.decoder_state is None
        assert streaming_vits.prev_overlap is None


class TestStreamingPipeline:
    """Test streaming pipeline"""

    @pytest.fixture()
    def pipeline(self):
        """Create streaming pipeline"""
        # Create mock models
        acoustic_model = Mock()
        acoustic_model.eval = Mock()
        acoustic_model.infer = Mock(return_value=torch.randn(1, 80, 100))

        vocoder = Mock()
        vocoder.eval = Mock()
        vocoder.forward = Mock(return_value=torch.randn(1, 1, 8000))

        config = StreamingConfig(
            chunk_size=256,
            use_cuda_streams=False,  # Disable for testing
            enable_jit=False,
        )

        pipeline = StreamingPipeline(
            acoustic_model=acoustic_model, vocoder=vocoder, config=config, device="cpu"
        )

        return pipeline

    def test_pipeline_creation(self, pipeline):
        """Test pipeline creation"""
        assert pipeline.config.chunk_size == 256
        assert not pipeline.config.use_cuda_streams
        assert len(pipeline.text_buffer) == 0
        assert len(pipeline.mel_buffer) == 0
        assert len(pipeline.audio_buffer) == 0

    def test_process_text_chunk(self, pipeline):
        """Test processing single text chunk"""
        text_chunk = "これはテストです"
        speaker_id = 0

        # Process chunk
        audio = pipeline.process_text_chunk(text_chunk, speaker_id)

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0

    @pytest.mark.asyncio()
    async def test_stream_synthesis(self, pipeline):
        """Test streaming synthesis"""

        # Create async text generator
        async def text_generator():
            texts = ["これは", "ストリーミング", "テストです"]
            for text in texts:
                yield text

        # Collect audio chunks
        audio_chunks = []
        async for audio_chunk in pipeline.stream_synthesis(text_generator()):
            audio_chunks.append(audio_chunk)

        assert len(audio_chunks) > 0
        for chunk in audio_chunks:
            assert isinstance(chunk, np.ndarray)
            assert chunk.dtype == np.float32

    def test_processing_thread(self, pipeline):
        """Test background processing thread"""
        # Start processing thread
        pipeline.start_processing_thread()

        assert pipeline.processing_thread is not None
        assert pipeline.processing_thread.is_alive()

        # Add item to buffer
        pipeline.text_buffer.append("テスト")

        # Wait a bit
        time.sleep(0.1)

        # Stop thread
        pipeline.stop_processing_thread()

        assert pipeline.processing_thread is None


class TestOptimizedVocoder:
    """Test optimized vocoder for streaming"""

    @pytest.fixture()
    def optimized_vocoder(self):
        """Create optimized vocoder"""
        base_vocoder = Mock()
        base_vocoder.forward = Mock(return_value=torch.randn(1, 1, 256))

        # Mock modules for cache initialization
        conv1 = Mock()
        conv1.kernel_size = (7,)
        conv1.in_channels = 80

        base_vocoder.named_modules = Mock(return_value=[("conv1", conv1)])

        return OptimizedVocoder(base_vocoder, cache_size=16)

    def test_vocoder_creation(self, optimized_vocoder):
        """Test optimized vocoder creation"""
        assert optimized_vocoder.cache_size == 16
        assert len(optimized_vocoder.conv_caches) > 0

    def test_incremental_forward(self, optimized_vocoder):
        """Test incremental forward pass"""
        # Single mel frame
        mel_frame = torch.randn(1, 80, 1)

        # Forward pass
        audio_frame = optimized_vocoder.forward_incremental(mel_frame)

        assert isinstance(audio_frame, torch.Tensor)
        assert audio_frame.dim() == 3

    def test_cache_reset(self, optimized_vocoder):
        """Test cache reset"""
        # Process frame to populate cache
        mel_frame = torch.randn(1, 80, 1)
        optimized_vocoder.forward_incremental(mel_frame)

        # Reset cache
        optimized_vocoder.reset_cache()

        # Check caches are zeroed
        for cache in optimized_vocoder.conv_caches.values():
            assert torch.all(cache == 0)


class TestLatencyMeasurement:
    """Test latency measurement utilities"""

    def test_measure_latency(self):
        """Test latency measurement"""
        # Create mock pipeline
        pipeline = Mock()
        pipeline.process_text_chunk = Mock(return_value=np.random.randn(1000))

        # Measure latency
        latency_stats = measure_latency(pipeline, test_text="テスト", num_iterations=5)

        assert isinstance(latency_stats, dict)
        assert "mean_latency_ms" in latency_stats
        assert "std_latency_ms" in latency_stats
        assert "min_latency_ms" in latency_stats
        assert "max_latency_ms" in latency_stats
        assert "p95_latency_ms" in latency_stats

        # Check values are reasonable
        assert latency_stats["mean_latency_ms"] >= 0
        assert latency_stats["min_latency_ms"] <= latency_stats["mean_latency_ms"]
        assert latency_stats["max_latency_ms"] >= latency_stats["mean_latency_ms"]


class TestStreamingConfig:
    """Test streaming configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = StreamingConfig()

        assert config.chunk_size == 256
        assert config.lookahead_chunks == 2
        assert config.mel_chunk_size == 64
        assert config.audio_chunk_size == 4096
        assert config.buffer_size == 10
        assert config.max_latency_ms == 100.0
        assert config.use_cuda_streams == True
        assert config.num_cuda_streams == 3
        assert config.enable_jit == True

    def test_custom_config(self):
        """Test custom configuration"""
        config = StreamingConfig(
            chunk_size=512, max_latency_ms=50.0, use_cuda_streams=False
        )

        assert config.chunk_size == 512
        assert config.max_latency_ms == 50.0
        assert config.use_cuda_streams == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
