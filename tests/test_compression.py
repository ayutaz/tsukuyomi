"""
Tests for model compression and quantization
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.tools.model_compression import (
    ModelCompressor,
    QuantizationConfig,
    PruningConfig,
    DistillationConfig,
    quantize_model,
    prune_model,
    distill_model,
    optimize_for_mobile,
    benchmark_model,
)


class DummyModel(nn.Module):
    """Dummy model for testing"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(80, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(256, 512, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.mean(dim=-1)  # Global average pooling
        x = self.fc(x)
        return x


class TestQuantization:
    """Test model quantization"""

    def test_quantization_config(self):
        """Test quantization configuration"""
        config = QuantizationConfig(
            backend="qnnpack", mode="dynamic", dtype=torch.qint8
        )

        assert config.backend == "qnnpack"
        assert config.mode == "dynamic"
        assert config.dtype == torch.qint8

    def test_dynamic_quantization(self):
        """Test dynamic quantization"""
        model = DummyModel()
        config = QuantizationConfig(mode="dynamic")

        # Get original size
        original_size = sum(p.numel() * p.element_size() for p in model.parameters())

        # Quantize model
        quantized_model = quantize_model(model, config)

        # Check model is quantized
        assert any("quantized" in str(m) for m in quantized_model.modules())

        # Test forward pass
        dummy_input = torch.randn(1, 80, 100)
        with torch.no_grad():
            output = quantized_model(dummy_input)

        assert output.shape == (1, 128)

    def test_static_quantization(self):
        """Test static quantization"""
        model = DummyModel()
        model.eval()

        config = QuantizationConfig(mode="static")

        # Create calibration data
        def calibration_fn():
            for _ in range(10):
                yield torch.randn(1, 80, 100)

        # Quantize model
        quantized_model = quantize_model(
            model, config, calibration_data=calibration_fn()
        )

        # Test forward pass
        dummy_input = torch.randn(1, 80, 100)
        with torch.no_grad():
            output = quantized_model(dummy_input)

        assert output.shape == (1, 128)

    def test_qat_quantization(self):
        """Test quantization-aware training"""
        model = DummyModel()
        config = QuantizationConfig(mode="qat")

        # Prepare for QAT
        qat_model = quantize_model(model, config, prepare_qat=True)

        # Check QAT modules are inserted
        has_fake_quant = any("FakeQuant" in str(m) for m in qat_model.modules())
        assert has_fake_quant

        # Simulate training
        optimizer = torch.optim.Adam(qat_model.parameters())
        dummy_input = torch.randn(2, 80, 100)
        dummy_target = torch.randn(2, 128)

        output = qat_model(dummy_input)
        loss = nn.MSELoss()(output, dummy_target)
        loss.backward()
        optimizer.step()

        # Convert to quantized model
        quantized_model = quantize_model(qat_model, config, finalize_qat=True)

        # Test quantized model
        with torch.no_grad():
            output = quantized_model(dummy_input)

        assert output.shape == (2, 128)


class TestPruning:
    """Test model pruning"""

    def test_pruning_config(self):
        """Test pruning configuration"""
        config = PruningConfig(method="magnitude", sparsity=0.5, structured=False)

        assert config.method == "magnitude"
        assert config.sparsity == 0.5
        assert config.structured == False

    def test_magnitude_pruning(self):
        """Test magnitude-based pruning"""
        model = DummyModel()
        config = PruningConfig(method="magnitude", sparsity=0.5)

        # Get original number of parameters
        original_params = sum(p.numel() for p in model.parameters())

        # Prune model
        pruned_model = prune_model(model, config)

        # Check sparsity
        total_params = 0
        zero_params = 0
        for module in pruned_model.modules():
            if hasattr(module, "weight"):
                total_params += module.weight.numel()
                zero_params += (module.weight == 0).sum().item()

        sparsity = zero_params / total_params
        assert sparsity >= 0.4  # Allow some tolerance

        # Test forward pass
        dummy_input = torch.randn(1, 80, 100)
        with torch.no_grad():
            output = pruned_model(dummy_input)

        assert output.shape == (1, 128)

    def test_structured_pruning(self):
        """Test structured pruning"""
        model = DummyModel()
        config = PruningConfig(method="l1_structured", sparsity=0.3, structured=True)

        # Prune model
        pruned_model = prune_model(model, config)

        # Check that some channels are completely pruned
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Conv1d):
                # Check if some output channels are zero
                weight_norms = module.weight.data.norm(dim=(1, 2))
                assert (weight_norms == 0).any()

        # Test forward pass
        dummy_input = torch.randn(1, 80, 100)
        with torch.no_grad():
            output = pruned_model(dummy_input)

        assert output.shape == (1, 128)

    def test_iterative_pruning(self):
        """Test iterative pruning"""
        model = DummyModel()
        config = PruningConfig(
            method="magnitude", sparsity=0.8, iterative=True, iterations=3
        )

        # Prune model iteratively
        pruned_model = prune_model(model, config)

        # Check high sparsity is achieved
        total_params = 0
        zero_params = 0
        for module in pruned_model.modules():
            if hasattr(module, "weight"):
                total_params += module.weight.numel()
                zero_params += (module.weight == 0).sum().item()

        sparsity = zero_params / total_params
        assert sparsity >= 0.7  # Allow some tolerance


class TestDistillation:
    """Test knowledge distillation"""

    def test_distillation_config(self):
        """Test distillation configuration"""
        config = DistillationConfig(
            temperature=4.0, alpha=0.7, student_architecture="small"
        )

        assert config.temperature == 4.0
        assert config.alpha == 0.7
        assert config.student_architecture == "small"

    def test_distillation_process(self):
        """Test distillation process"""
        # Create teacher and student models
        teacher = DummyModel()

        # Smaller student model
        class StudentModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(80, 128, 3, padding=1)
                self.relu = nn.ReLU()
                self.fc = nn.Linear(128, 128)

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = x.mean(dim=-1)
                x = self.fc(x)
                return x

        student = StudentModel()

        config = DistillationConfig(temperature=4.0, alpha=0.7)

        # Create dummy dataset
        dataset = [(torch.randn(80, 100), torch.randn(128)) for _ in range(10)]

        # Distill knowledge
        distilled_student = distill_model(
            teacher=teacher,
            student=student,
            train_dataset=dataset,
            config=config,
            epochs=2,
        )

        # Test student model
        dummy_input = torch.randn(1, 80, 100)
        with torch.no_grad():
            teacher_output = teacher(dummy_input)
            student_output = distilled_student(dummy_input)

        assert student_output.shape == teacher_output.shape


class TestModelCompressor:
    """Test integrated model compressor"""

    def test_compressor_creation(self):
        """Test model compressor creation"""
        model = DummyModel()
        compressor = ModelCompressor(model)

        assert compressor.original_model == model
        assert compressor.compressed_model is None

    def test_compression_pipeline(self):
        """Test full compression pipeline"""
        model = DummyModel()
        compressor = ModelCompressor(model)

        # Configure compression
        compression_config = {
            "quantization": {"enabled": True, "mode": "dynamic"},
            "pruning": {"enabled": True, "sparsity": 0.5},
        }

        # Compress model
        compressed_model = compressor.compress(compression_config)

        assert compressed_model is not None

        # Test compressed model
        dummy_input = torch.randn(1, 80, 100)
        with torch.no_grad():
            output = compressed_model(dummy_input)

        assert output.shape == (1, 128)

    def test_compression_ratio(self):
        """Test compression ratio calculation"""
        model = DummyModel()
        compressor = ModelCompressor(model)

        # Compress with quantization
        compressed_model = compressor.compress(
            {"quantization": {"enabled": True, "mode": "dynamic"}}
        )

        # Calculate compression ratio
        ratio = compressor.get_compression_ratio()

        assert ratio > 1.0  # Should be compressed

    def test_save_load_compressed(self):
        """Test saving and loading compressed model"""
        model = DummyModel()
        compressor = ModelCompressor(model)

        # Compress model
        compressed_model = compressor.compress(
            {"quantization": {"enabled": True, "mode": "dynamic"}}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save compressed model
            save_path = Path(tmpdir) / "compressed_model.pt"
            compressor.save_compressed_model(save_path)

            assert save_path.exists()

            # Load compressed model
            loaded_model = ModelCompressor.load_compressed_model(
                save_path, model_class=DummyModel
            )

            # Test loaded model
            dummy_input = torch.randn(1, 80, 100)
            with torch.no_grad():
                output = loaded_model(dummy_input)

            assert output.shape == (1, 128)


class TestMobileOptimization:
    """Test mobile optimization"""

    def test_mobile_optimization(self):
        """Test optimization for mobile deployment"""
        model = DummyModel()

        # Optimize for mobile
        mobile_model = optimize_for_mobile(model, backend="cpu", optimize_for_size=True)

        # Test optimized model
        dummy_input = torch.randn(1, 80, 100)
        with torch.no_grad():
            output = mobile_model(dummy_input)

        assert output.shape == (1, 128)

    def test_torchscript_export(self):
        """Test TorchScript export for mobile"""
        model = DummyModel()
        model.eval()

        # Export to TorchScript
        dummy_input = torch.randn(1, 80, 100)
        scripted_model = torch.jit.trace(model, dummy_input)

        # Optimize for mobile
        mobile_model = optimize_for_mobile(scripted_model, backend="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save mobile model
            save_path = Path(tmpdir) / "mobile_model.pt"
            mobile_model._save_for_lite_interpreter(str(save_path))

            assert save_path.exists()


class TestBenchmarking:
    """Test model benchmarking"""

    def test_benchmark_model(self):
        """Test model benchmarking"""
        model = DummyModel()
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 80, 100)

        # Benchmark model
        results = benchmark_model(model, dummy_input, num_runs=10, warmup_runs=2)

        assert "mean_latency" in results
        assert "std_latency" in results
        assert "min_latency" in results
        assert "max_latency" in results
        assert "throughput" in results
        assert "memory_usage" in results

        # Check values are reasonable
        assert results["mean_latency"] > 0
        assert results["throughput"] > 0

    def test_compare_models(self):
        """Test comparing original and compressed models"""
        original_model = DummyModel()

        # Create compressed version
        compressor = ModelCompressor(original_model)
        compressed_model = compressor.compress(
            {"quantization": {"enabled": True, "mode": "dynamic"}}
        )

        # Benchmark both
        dummy_input = torch.randn(1, 80, 100)

        original_results = benchmark_model(original_model, dummy_input)
        compressed_results = benchmark_model(compressed_model, dummy_input)

        # Compressed should be faster
        assert (
            compressed_results["mean_latency"] <= original_results["mean_latency"] * 1.1
        )  # Allow 10% tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
