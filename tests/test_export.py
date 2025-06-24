"""Tests for ONNX export functionality."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import onnx
import pytest
import torch
import torch.nn as nn

from src.export.onnx_export import (
    ExportConfig,
    ONNXExporter,
    UnityExporter,
    export_for_deployment,
)


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Linear(100, hidden_size)
        self.decoder = nn.Linear(hidden_size, 80)

    def forward(
        self,
        phoneme_ids: torch.Tensor,
        phoneme_lengths: torch.Tensor,
        speaker_ids: torch.Tensor,
        emotion_ids: torch.Tensor,
        style_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = phoneme_ids.shape

        # Mock processing
        hidden = self.encoder(phoneme_ids.float())
        mel_outputs = self.decoder(hidden)

        # Mock outputs
        durations = torch.ones(batch_size, seq_len)
        pitch = torch.randn(batch_size, seq_len)
        energy = torch.randn(batch_size, seq_len)

        return mel_outputs, durations, pitch, energy


class TestONNXExporter:
    """Test ONNX exporter."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def export_config(self, temp_dir):
        """Create export configuration."""
        return ExportConfig(
            model_path="dummy_model.pt",
            output_dir=str(temp_dir),
            optimize=True,
            quantize=False,
        )

    @pytest.fixture
    def exporter(self, export_config):
        """Create exporter instance."""
        return ONNXExporter(export_config)

    def test_export_acoustic_model(self, exporter, temp_dir):
        """Test acoustic model export."""
        model = MockModel()

        onnx_path = exporter.export_model(model, "acoustic_model", "test_model")

        assert onnx_path.exists()
        assert onnx_path.suffix == ".onnx"

        # Check metadata
        metadata_path = temp_dir / "test_model_acoustic_metadata.json"
        assert metadata_path.exists()

    def test_export_with_optimization(self, exporter):
        """Test export with optimization."""
        model = MockModel()

        # Mock onnx optimizer
        with patch("onnx.optimizer.optimize") as mock_optimize:
            mock_optimize.return_value = Mock()

            onnx_path = exporter.export_model(model, "acoustic_model", "test_optimized")

            # Should call optimizer
            if exporter.config.optimize:
                assert mock_optimize.called

    def test_dynamic_axes(self, exporter, temp_dir):
        """Test dynamic axes in export."""
        model = MockModel()

        onnx_path = exporter.export_model(model, "acoustic_model", "test_dynamic")

        # Load and check model
        onnx_model = onnx.load(str(onnx_path))

        # Check dynamic dimensions
        for input_info in onnx_model.graph.input:
            if input_info.name == "phoneme_ids":
                dims = input_info.type.tensor_type.shape.dim
                # Should have dynamic batch and sequence dimensions
                assert any(d.dim_param for d in dims)

    def test_validation(self, exporter):
        """Test model validation after export."""
        model = MockModel()

        # Export should validate
        onnx_path = exporter.export_model(model, "acoustic_model", "test_validation")

        # Should not raise
        onnx.checker.check_model(str(onnx_path))

    def test_unity_export(self, export_config, temp_dir):
        """Test Unity-specific export."""
        exporter = ONNXExporter(export_config)
        unity_exporter = UnityExporter(exporter)

        acoustic_model = MockModel()
        vocoder_model = MockModel()

        results = unity_exporter.export_for_unity(
            acoustic_model, vocoder_model, "tsukuyomi_test"
        )

        assert "acoustic_model" in results
        assert "vocoder" in results
        assert "unity_wrapper" in results

        # Check Unity wrapper was created
        wrapper_path = results["unity_wrapper"]
        assert wrapper_path.exists()
        assert wrapper_path.suffix == ".cs"


class TestExportIntegration:
    """Integration tests for export functionality."""

    @pytest.fixture
    def checkpoint_path(self, tmp_path):
        """Create mock checkpoint."""
        checkpoint = {
            "acoustic_model": MockModel(),
            "vocoder": MockModel(),
            "model_state_dict": MockModel().state_dict(),
        }

        path = tmp_path / "checkpoint.pt"
        torch.save(checkpoint, path)
        return str(path)

    def test_export_for_deployment(self, checkpoint_path, tmp_path):
        """Test high-level export function."""
        with patch("src.export.onnx_export.ONNXExporter.export_model") as mock_export:
            mock_export.return_value = tmp_path / "model.onnx"

            results = export_for_deployment(
                checkpoint_path,
                str(tmp_path),
                targets=["acoustic_model"],
                optimization_level="O2",
            )

            assert "acoustic_model" in results
            assert mock_export.called
