"""
Tests for edge device optimization
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import onnx
import onnxruntime as ort

from src.tools.edge_optimization import (
    EdgeOptimizer,
    EdgeConfig,
    ONNXConverter,
    TFLiteConverter,
    CoreMLConverter,
    OpenVINOConverter,
    EdgeBenchmark,
    optimize_for_edge,
    convert_to_onnx,
    convert_to_tflite,
    convert_to_coreml,
    convert_to_openvino,
)


class DummyTTSModel(torch.nn.Module):
    """Dummy TTS model for testing"""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(256, 128)
        self.decoder = torch.nn.Linear(128, 80)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.relu(x)
        x = self.decoder(x)
        return x


class TestEdgeConfig:
    """Test edge configuration"""

    def test_default_config(self):
        """Test default edge configuration"""
        config = EdgeConfig()

        assert config.target_device == "cpu"
        assert config.optimization_level == 2
        assert config.quantization == True
        assert config.fp16 == False
        assert config.batch_size == 1

    def test_custom_config(self):
        """Test custom edge configuration"""
        config = EdgeConfig(
            target_device="gpu",
            optimization_level=3,
            quantization=False,
            fp16=True,
            batch_size=4,
        )

        assert config.target_device == "gpu"
        assert config.optimization_level == 3
        assert config.quantization == False
        assert config.fp16 == True
        assert config.batch_size == 4


class TestONNXConverter:
    """Test ONNX conversion"""

    def test_onnx_conversion(self):
        """Test basic ONNX conversion"""
        model = DummyTTSModel()
        converter = ONNXConverter()

        # Create dummy input
        dummy_input = torch.randn(1, 256)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"

            # Convert to ONNX
            converter.convert(
                model,
                dummy_input,
                output_path,
                input_names=["input"],
                output_names=["output"],
            )

            assert output_path.exists()

            # Verify ONNX model
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)

    def test_onnx_optimization(self):
        """Test ONNX optimization"""
        model = DummyTTSModel()
        converter = ONNXConverter()

        dummy_input = torch.randn(1, 256)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"

            # Convert with optimization
            converter.convert(
                model, dummy_input, output_path, optimize=True, optimization_level=2
            )

            # Load and check optimized model
            onnx_model = onnx.load(str(output_path))

            # Test with ONNX Runtime
            ort_session = ort.InferenceSession(str(output_path))

            # Run inference
            input_name = ort_session.get_inputs()[0].name
            output = ort_session.run(None, {input_name: dummy_input.numpy()})

            assert output[0].shape == (1, 80)

    def test_dynamic_shapes(self):
        """Test ONNX conversion with dynamic shapes"""
        model = DummyTTSModel()
        converter = ONNXConverter()

        dummy_input = torch.randn(1, 256)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"

            # Convert with dynamic axes
            converter.convert(
                model,
                dummy_input,
                output_path,
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )

            # Test with different batch sizes
            ort_session = ort.InferenceSession(str(output_path))

            for batch_size in [1, 2, 4]:
                test_input = torch.randn(batch_size, 256)
                input_name = ort_session.get_inputs()[0].name
                output = ort_session.run(None, {input_name: test_input.numpy()})
                assert output[0].shape == (batch_size, 80)


class TestTFLiteConverter:
    """Test TensorFlow Lite conversion"""

    @patch("tensorflow.lite.TFLiteConverter")
    def test_tflite_conversion(self, mock_converter_class):
        """Test basic TFLite conversion"""
        model = DummyTTSModel()
        converter = TFLiteConverter()

        # Mock TFLite converter
        mock_converter = Mock()
        mock_converter.convert.return_value = b"fake_tflite_model"
        mock_converter_class.from_concrete_functions.return_value = mock_converter

        dummy_input = torch.randn(1, 256)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.tflite"

            # Convert to TFLite
            converter.convert(model, dummy_input, output_path)

            assert output_path.exists()

    @patch("tensorflow.lite.TFLiteConverter")
    def test_tflite_quantization(self, mock_converter_class):
        """Test TFLite conversion with quantization"""
        model = DummyTTSModel()
        converter = TFLiteConverter()

        # Mock TFLite converter
        mock_converter = Mock()
        mock_converter.convert.return_value = b"fake_quantized_model"
        mock_converter_class.from_concrete_functions.return_value = mock_converter

        dummy_input = torch.randn(1, 256)

        # Create calibration dataset
        calibration_data = [torch.randn(1, 256) for _ in range(10)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_quantized.tflite"

            # Convert with quantization
            converter.convert(
                model,
                dummy_input,
                output_path,
                quantization="int8",
                calibration_data=calibration_data,
            )

            # Check quantization was configured
            assert mock_converter.optimizations
            assert mock_converter.representative_dataset is not None


class TestCoreMLConverter:
    """Test Core ML conversion"""

    @patch("coremltools.convert")
    def test_coreml_conversion(self, mock_convert):
        """Test basic Core ML conversion"""
        model = DummyTTSModel()
        converter = CoreMLConverter()

        # Mock Core ML model
        mock_ml_model = Mock()
        mock_ml_model.save = Mock()
        mock_convert.return_value = mock_ml_model

        dummy_input = torch.randn(1, 256)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.mlmodel"

            # Convert to Core ML
            converter.convert(
                model, dummy_input, output_path, minimum_deployment_target="iOS14"
            )

            mock_ml_model.save.assert_called_once()

    @patch("coremltools.convert")
    def test_coreml_neural_engine(self, mock_convert):
        """Test Core ML conversion for Neural Engine"""
        model = DummyTTSModel()
        converter = CoreMLConverter()

        # Mock Core ML model
        mock_ml_model = Mock()
        mock_convert.return_value = mock_ml_model

        dummy_input = torch.randn(1, 256)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_ne.mlmodel"

            # Convert for Neural Engine
            converter.convert(
                model,
                dummy_input,
                output_path,
                compute_precision="float16",
                convert_to="neuralnetwork",
            )

            # Check conversion options
            mock_convert.assert_called()
            call_kwargs = mock_convert.call_args[1]
            assert "compute_precision" in call_kwargs


class TestOpenVINOConverter:
    """Test OpenVINO conversion"""

    @patch("openvino.runtime.Core")
    def test_openvino_conversion(self, mock_core_class):
        """Test basic OpenVINO conversion"""
        model = DummyTTSModel()
        converter = OpenVINOConverter()

        # Mock OpenVINO
        mock_core = Mock()
        mock_model = Mock()
        mock_core.read_model.return_value = mock_model
        mock_core_class.return_value = mock_core

        dummy_input = torch.randn(1, 256)

        with tempfile.TemporaryDirectory() as tmpdir:
            # First convert to ONNX
            onnx_path = Path(tmpdir) / "temp.onnx"
            torch.onnx.export(model, dummy_input, onnx_path)

            output_path = Path(tmpdir) / "model.xml"

            # Convert to OpenVINO
            converter.convert(onnx_path, output_path, precision="FP16")

            assert output_path.exists()


class TestEdgeOptimizer:
    """Test integrated edge optimizer"""

    def test_optimizer_creation(self):
        """Test edge optimizer creation"""
        model = DummyTTSModel()
        config = EdgeConfig()

        optimizer = EdgeOptimizer(model, config)

        assert optimizer.model == model
        assert optimizer.config == config

    def test_optimization_pipeline(self):
        """Test full optimization pipeline"""
        model = DummyTTSModel()
        config = EdgeConfig(
            target_device="cpu", optimization_level=2, quantization=True
        )

        optimizer = EdgeOptimizer(model, config)

        # Optimize model
        optimized_model = optimizer.optimize()

        # Test optimized model
        dummy_input = torch.randn(1, 256)
        with torch.no_grad():
            output = optimized_model(dummy_input)

        assert output.shape == (1, 80)

    def test_multi_backend_export(self):
        """Test exporting to multiple backends"""
        model = DummyTTSModel()
        config = EdgeConfig()

        optimizer = EdgeOptimizer(model, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Export to multiple formats
            exported_files = optimizer.export_all_formats(
                output_dir,
                dummy_input=torch.randn(1, 256),
                formats=["onnx", "torchscript"],
            )

            assert "onnx" in exported_files
            assert "torchscript" in exported_files
            assert exported_files["onnx"].exists()
            assert exported_files["torchscript"].exists()


class TestEdgeBenchmark:
    """Test edge device benchmarking"""

    def test_benchmark_creation(self):
        """Test benchmark creation"""
        benchmark = EdgeBenchmark(device="cpu")

        assert benchmark.device == "cpu"
        assert benchmark.results == {}

    def test_latency_measurement(self):
        """Test latency measurement"""
        model = DummyTTSModel()
        benchmark = EdgeBenchmark()

        dummy_input = torch.randn(1, 256)

        # Measure latency
        latency_stats = benchmark.measure_latency(
            model, dummy_input, num_runs=10, warmup_runs=2
        )

        assert "mean" in latency_stats
        assert "std" in latency_stats
        assert "min" in latency_stats
        assert "max" in latency_stats
        assert "p95" in latency_stats

        assert latency_stats["mean"] > 0

    def test_memory_measurement(self):
        """Test memory usage measurement"""
        model = DummyTTSModel()
        benchmark = EdgeBenchmark()

        # Measure memory
        memory_stats = benchmark.measure_memory(model)

        assert "model_size_mb" in memory_stats
        assert "peak_memory_mb" in memory_stats
        assert memory_stats["model_size_mb"] > 0

    def test_power_estimation(self):
        """Test power consumption estimation"""
        model = DummyTTSModel()
        benchmark = EdgeBenchmark()

        dummy_input = torch.randn(1, 256)

        # Estimate power (mock for testing)
        with patch.object(benchmark, "_get_power_usage", return_value=2.5):
            power_stats = benchmark.estimate_power(
                model, dummy_input, duration_seconds=1.0
            )

        assert "average_power_w" in power_stats
        assert "energy_consumption_j" in power_stats
        assert power_stats["average_power_w"] == 2.5

    def test_comprehensive_benchmark(self):
        """Test comprehensive benchmarking"""
        model = DummyTTSModel()
        benchmark = EdgeBenchmark()

        dummy_input = torch.randn(1, 256)

        # Run comprehensive benchmark
        results = benchmark.run_comprehensive(
            model,
            dummy_input,
            include_power=False,  # Disable power measurement for testing
        )

        assert "latency" in results
        assert "memory" in results
        assert "throughput" in results

        # Generate report
        report = benchmark.generate_report()

        assert isinstance(report, dict)
        assert "device" in report
        assert "results" in report


class TestOptimizationHelpers:
    """Test optimization helper functions"""

    def test_optimize_for_edge(self):
        """Test edge optimization helper"""
        model = DummyTTSModel()

        # Optimize for edge
        optimized_model = optimize_for_edge(
            model,
            target_device="mobile",
            quantization=True,
            pruning=True,
            pruning_sparsity=0.5,
        )

        # Test optimized model
        dummy_input = torch.randn(1, 256)
        with torch.no_grad():
            output = optimized_model(dummy_input)

        assert output.shape == (1, 80)

    def test_format_specific_conversion(self):
        """Test format-specific conversion helpers"""
        model = DummyTTSModel()
        dummy_input = torch.randn(1, 256)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test ONNX conversion
            onnx_path = Path(tmpdir) / "model.onnx"
            convert_to_onnx(model, dummy_input, onnx_path)
            assert onnx_path.exists()

            # Test TorchScript conversion
            ts_model = torch.jit.trace(model, dummy_input)
            assert ts_model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
