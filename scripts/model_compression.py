#!/usr/bin/env python3
"""Model Compression and Quantization for Tsukuyomi TTS

This script provides:
- Model pruning (structured and unstructured)
- Quantization (dynamic, static, QAT)
- Knowledge distillation
- Model optimization for deployment
- Performance benchmarking
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelCompressor:
    """Main class for model compression operations"""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

    def prune_model(
        self, model: nn.Module, pruning_config: Dict, structured: bool = False
    ) -> nn.Module:
        """Apply pruning to model

        Args:
            model: PyTorch model
            pruning_config: Pruning configuration
            structured: Whether to use structured pruning

        Returns:
            Pruned model
        """
        logger.info(
            f"Applying {'structured' if structured else 'unstructured'} pruning..."
        )

        # Get pruning parameters
        sparsity = pruning_config.get("sparsity", 0.5)
        exclude_modules = pruning_config.get("exclude_modules", [])

        if structured:
            return self._structured_pruning(model, sparsity, exclude_modules)
        else:
            return self._unstructured_pruning(model, sparsity, exclude_modules)

    def _unstructured_pruning(
        self, model: nn.Module, sparsity: float, exclude_modules: List[str]
    ) -> nn.Module:
        """Apply unstructured pruning (weight-level)"""
        parameters_to_prune = []

        for name, module in model.named_modules():
            # Skip excluded modules
            if any(excl in name for excl in exclude_modules):
                continue

            # Prune linear and conv layers
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                parameters_to_prune.append((module, "weight"))

        # Apply global magnitude pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )

        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

        # Calculate actual sparsity
        total_params = 0
        pruned_params = 0

        for name, param in model.named_parameters():
            if "weight" in name:
                total_params += param.numel()
                pruned_params += (param == 0).sum().item()

        actual_sparsity = pruned_params / total_params
        logger.info(f"Actual sparsity: {actual_sparsity:.2%}")

        return model

    def _structured_pruning(
        self, model: nn.Module, sparsity: float, exclude_modules: List[str]
    ) -> nn.Module:
        """Apply structured pruning (channel/filter-level)"""
        for name, module in model.named_modules():
            # Skip excluded modules
            if any(excl in name for excl in exclude_modules):
                continue

            if isinstance(module, nn.Conv1d):
                # Prune output channels
                prune.ln_structured(module, name="weight", amount=sparsity, n=2, dim=0)
                prune.remove(module, "weight")

            elif isinstance(module, nn.Linear):
                # Prune output features
                prune.ln_structured(module, name="weight", amount=sparsity, n=2, dim=0)
                prune.remove(module, "weight")

        return model

    def quantize_model(
        self,
        model: nn.Module,
        quantization_config: Dict,
        calibration_data: Optional[torch.utils.data.DataLoader] = None,
    ) -> nn.Module:
        """Apply quantization to model

        Args:
            model: PyTorch model
            quantization_config: Quantization configuration
            calibration_data: Data for calibration (static quantization)

        Returns:
            Quantized model
        """
        method = quantization_config.get("method", "dynamic")
        backend = quantization_config.get("backend", "fbgemm")

        logger.info(f"Applying {method} quantization with {backend} backend...")

        if method == "dynamic":
            return self._dynamic_quantization(model, backend)
        elif method == "static":
            if calibration_data is None:
                raise ValueError("Calibration data required for static quantization")
            return self._static_quantization(model, backend, calibration_data)
        elif method == "qat":
            return self._quantization_aware_training(model, backend)
        else:
            raise ValueError(f"Unknown quantization method: {method}")

    def _dynamic_quantization(self, model: nn.Module, backend: str) -> nn.Module:
        """Apply dynamic quantization"""
        # Set quantization backend
        torch.backends.quantized.engine = backend

        # Specify modules to quantize
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            qconfig_spec={
                nn.Linear: torch.quantization.default_dynamic_qconfig,
                nn.LSTM: torch.quantization.default_dynamic_qconfig,
                nn.GRU: torch.quantization.default_dynamic_qconfig,
            },
            dtype=torch.qint8,
        )

        return quantized_model

    def _static_quantization(
        self,
        model: nn.Module,
        backend: str,
        calibration_data: torch.utils.data.DataLoader,
    ) -> nn.Module:
        """Apply static quantization"""
        # Set backend
        torch.backends.quantized.engine = backend

        # Prepare model for quantization
        model.eval()

        # Fuse modules
        model = self._fuse_modules(model)

        # Specify quantization configuration
        model.qconfig = torch.quantization.get_default_qconfig(backend)

        # Prepare model
        torch.quantization.prepare(model, inplace=True)

        # Calibration
        logger.info("Running calibration...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(calibration_data):
                if batch_idx >= 100:  # Use first 100 batches for calibration
                    break
                model(batch)

        # Convert to quantized model
        quantized_model = torch.quantization.convert(model, inplace=True)

        return quantized_model

    def _quantization_aware_training(self, model: nn.Module, backend: str) -> nn.Module:
        """Prepare model for quantization-aware training"""
        # Set backend
        torch.backends.quantized.engine = backend

        # Fuse modules
        model = self._fuse_modules(model)

        # Set qconfig
        model.qconfig = torch.quantization.get_default_qat_qconfig(backend)

        # Prepare for QAT
        torch.quantization.prepare_qat(model, inplace=True)

        logger.info("Model prepared for quantization-aware training")
        logger.info(
            "Train the model, then call torch.quantization.convert() to get quantized model"
        )

        return model

    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """Fuse modules for better quantization"""
        # This is model-specific - example for common patterns
        for name, module in model.named_children():
            if isinstance(module, nn.Sequential):
                # Look for conv-bn-relu patterns
                for idx in range(len(module) - 2):
                    if (
                        isinstance(module[idx], (nn.Conv1d, nn.Conv2d))
                        and isinstance(module[idx + 1], nn.BatchNorm1d)
                        and isinstance(module[idx + 2], nn.ReLU)
                    ):
                        # Fuse conv-bn-relu
                        torch.quantization.fuse_modules(
                            module, [str(idx), str(idx + 1), str(idx + 2)], inplace=True
                        )
        return model

    def distill_model(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        distillation_config: Dict,
    ) -> nn.Module:
        """Knowledge distillation from teacher to student model

        Args:
            teacher_model: Teacher model (larger)
            student_model: Student model (smaller)
            train_loader: Training data
            val_loader: Validation data
            distillation_config: Distillation configuration

        Returns:
            Trained student model
        """
        logger.info("Starting knowledge distillation...")

        # Configuration
        temperature = distillation_config.get("temperature", 5.0)
        alpha = distillation_config.get("alpha", 0.7)  # Weight for distillation loss
        epochs = distillation_config.get("epochs", 50)
        learning_rate = distillation_config.get("learning_rate", 1e-4)

        # Move models to device
        teacher_model = teacher_model.to(self.device)
        student_model = student_model.to(self.device)
        teacher_model.eval()

        # Optimizer
        optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)

        # Loss functions
        ce_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        # Training loop
        for epoch in range(epochs):
            student_model.train()
            total_loss = 0

            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)

                # Teacher predictions
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)

                # Student predictions
                student_outputs = student_model(inputs)

                # Distillation loss
                if isinstance(student_outputs, dict) and "logits" in student_outputs:
                    # Classification task
                    teacher_logits = teacher_outputs["logits"]
                    student_logits = student_outputs["logits"]

                    # Soft targets
                    soft_targets = nn.functional.softmax(
                        teacher_logits / temperature, dim=-1
                    )
                    soft_predictions = nn.functional.log_softmax(
                        student_logits / temperature, dim=-1
                    )

                    distillation_loss = nn.functional.kl_div(
                        soft_predictions, soft_targets, reduction="batchmean"
                    ) * (temperature**2)

                    # Hard targets
                    student_loss = ce_loss(student_logits, targets)

                else:
                    # Regression task or feature matching
                    distillation_loss = mse_loss(student_outputs, teacher_outputs)
                    student_loss = mse_loss(student_outputs, targets)

                # Combined loss
                loss = alpha * distillation_loss + (1 - alpha) * student_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Validation
            if epoch % 10 == 0:
                val_loss = self._validate_student(student_model, val_loader)
                logger.info(
                    f"Epoch {epoch}/{epochs} - "
                    f"Train Loss: {total_loss/len(train_loader):.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )

        return student_model

    def _validate_student(
        self, model: nn.Module, val_loader: torch.utils.data.DataLoader
    ) -> float:
        """Validate student model"""
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)

                outputs = model(inputs)

                if isinstance(outputs, dict) and "logits" in outputs:
                    loss = nn.functional.cross_entropy(outputs["logits"], targets)
                else:
                    loss = nn.functional.mse_loss(outputs, targets)

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def optimize_for_mobile(self, model: nn.Module) -> torch.jit.ScriptModule:
        """Optimize model for mobile deployment

        Args:
            model: PyTorch model

        Returns:
            Optimized TorchScript model
        """
        logger.info("Optimizing model for mobile...")

        # Convert to eval mode
        model.eval()

        # Trace or script the model
        example_input = self._get_example_input(model)

        try:
            # Try tracing first (faster)
            traced_model = torch.jit.trace(model, example_input)
        except:
            # Fall back to scripting
            traced_model = torch.jit.script(model)

        # Optimize for mobile
        optimized_model = torch.jit.optimize_for_mobile(traced_model)

        return optimized_model

    def _get_example_input(self, model: nn.Module) -> torch.Tensor:
        """Get example input for model tracing"""
        # This should be customized based on model architecture
        # Default example for sequence model
        batch_size = 1
        seq_length = 100
        input_dim = 256

        return torch.randn(batch_size, seq_length, input_dim)

    def benchmark_model(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        test_data: torch.utils.data.DataLoader,
        metrics: List[str] = ["size", "latency", "accuracy"],
    ) -> Dict:
        """Benchmark compressed model against original

        Args:
            original_model: Original model
            compressed_model: Compressed model
            test_data: Test dataset
            metrics: Metrics to evaluate

        Returns:
            Benchmark results
        """
        logger.info("Benchmarking models...")
        results = {}

        # Model size comparison
        if "size" in metrics:
            original_size = self._get_model_size(original_model)
            compressed_size = self._get_model_size(compressed_model)

            results["size"] = {
                "original_mb": original_size,
                "compressed_mb": compressed_size,
                "compression_ratio": original_size / compressed_size,
                "size_reduction": 1 - (compressed_size / original_size),
            }

        # Latency comparison
        if "latency" in metrics:
            original_latency = self._measure_latency(original_model, test_data)
            compressed_latency = self._measure_latency(compressed_model, test_data)

            results["latency"] = {
                "original_ms": original_latency,
                "compressed_ms": compressed_latency,
                "speedup": original_latency / compressed_latency,
                "latency_reduction": 1 - (compressed_latency / original_latency),
            }

        # Accuracy comparison
        if "accuracy" in metrics:
            original_outputs = self._get_model_outputs(original_model, test_data)
            compressed_outputs = self._get_model_outputs(compressed_model, test_data)

            # Calculate accuracy metrics
            mse = mean_squared_error(original_outputs, compressed_outputs)
            max_error = np.max(np.abs(original_outputs - compressed_outputs))

            results["accuracy"] = {
                "mse": float(mse),
                "max_error": float(max_error),
                "relative_error": float(mse / np.mean(original_outputs**2)),
            }

        return results

    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB"""
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024 / 1024

        return size_mb

    def _measure_latency(
        self,
        model: nn.Module,
        test_data: torch.utils.data.DataLoader,
        num_runs: int = 100,
    ) -> float:
        """Measure average inference latency"""
        model.eval()

        # Warmup
        for _ in range(10):
            batch = next(iter(test_data))
            with torch.no_grad():
                _ = model(batch["input"].to(self.device))

        # Measure
        latencies = []

        for i, batch in enumerate(test_data):
            if i >= num_runs:
                break

            input_data = batch["input"].to(self.device)

            start_time = time.time()
            with torch.no_grad():
                _ = model(input_data)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

            latencies.append((end_time - start_time) * 1000)  # Convert to ms

        return np.mean(latencies)

    def _get_model_outputs(
        self, model: nn.Module, test_data: torch.utils.data.DataLoader
    ) -> np.ndarray:
        """Get model outputs for comparison"""
        model.eval()
        outputs = []

        with torch.no_grad():
            for batch in test_data:
                input_data = batch["input"].to(self.device)
                output = model(input_data)

                if isinstance(output, dict):
                    output = output["output"]

                outputs.append(output.cpu().numpy())

        return np.concatenate(outputs)

    def export_compressed_model(
        self,
        model: nn.Module,
        export_path: str,
        format: str = "onnx",
        optimize: bool = True,
    ):
        """Export compressed model

        Args:
            model: Compressed model
            export_path: Path to save model
            format: Export format ('onnx', 'torchscript', 'tflite')
            optimize: Whether to apply format-specific optimizations
        """
        logger.info(f"Exporting model to {format} format...")

        if format == "onnx":
            self._export_onnx(model, export_path, optimize)
        elif format == "torchscript":
            self._export_torchscript(model, export_path, optimize)
        elif format == "tflite":
            self._export_tflite(model, export_path, optimize)
        else:
            raise ValueError(f"Unknown export format: {format}")

    def _export_onnx(self, model: nn.Module, export_path: str, optimize: bool):
        """Export to ONNX format"""
        model.eval()

        # Get example input
        example_input = self._get_example_input(model)

        # Export
        torch.onnx.export(
            model,
            example_input,
            export_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "sequence"},
                "output": {0: "batch_size", 1: "sequence"},
            },
        )

        if optimize:
            # Optimize ONNX model
            from onnxruntime.transformers import optimizer

            opt_model = optimizer.optimize_model(
                export_path,
                model_type="bert",  # Adjust based on your model
                num_heads=12,
                hidden_size=768,
            )
            opt_model.save_model_to_file(
                export_path.replace(".onnx", "_optimized.onnx")
            )

    def _export_torchscript(self, model: nn.Module, export_path: str, optimize: bool):
        """Export to TorchScript format"""
        model.eval()

        # Convert to TorchScript
        example_input = self._get_example_input(model)

        try:
            scripted_model = torch.jit.trace(model, example_input)
        except:
            scripted_model = torch.jit.script(model)

        if optimize:
            scripted_model = torch.jit.optimize_for_mobile(scripted_model)

        # Save
        scripted_model.save(export_path)

    def _export_tflite(self, model: nn.Module, export_path: str, optimize: bool):
        """Export to TFLite format (requires TensorFlow)"""
        try:
            import tensorflow as tf
        except ImportError:
            logger.error("TensorFlow required for TFLite export")
            return

        # First export to ONNX
        onnx_path = export_path.replace(".tflite", ".onnx")
        self._export_onnx(model, onnx_path, optimize=False)

        # Convert ONNX to TF
        # This is simplified - actual conversion may need onnx-tf
        logger.warning("TFLite conversion requires additional tools (onnx-tf)")


def create_student_model(
    teacher_model: nn.Module, compression_ratio: float = 0.5
) -> nn.Module:
    """Create a smaller student model based on teacher architecture

    Args:
        teacher_model: Teacher model
        compression_ratio: How much to compress (0.5 = half the size)

    Returns:
        Student model
    """
    # This is a simplified example - should be customized based on model architecture
    student_model = type(teacher_model)()

    # Reduce hidden dimensions
    for name, module in student_model.named_modules():
        if isinstance(module, nn.Linear):
            # Reduce dimensions
            in_features = int(module.in_features * compression_ratio)
            out_features = int(module.out_features * compression_ratio)

            # Replace module
            new_module = nn.Linear(
                in_features, out_features, bias=module.bias is not None
            )
            setattr(student_model, name.split(".")[-1], new_module)

        elif isinstance(module, nn.Conv1d):
            # Reduce channels
            in_channels = int(module.in_channels * compression_ratio)
            out_channels = int(module.out_channels * compression_ratio)

            new_module = nn.Conv1d(
                in_channels,
                out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                bias=module.bias is not None,
            )
            setattr(student_model, name.split(".")[-1], new_module)

    return student_model


def main():
    parser = argparse.ArgumentParser(description="Model Compression Tool")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output-dir", type=str, default="compressed_models", help="Output directory"
    )
    parser.add_argument(
        "--compression-methods",
        nargs="+",
        choices=["prune", "quantize", "distill"],
        default=["quantize"],
        help="Compression methods to apply",
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.5, help="Pruning sparsity (0-1)"
    )
    parser.add_argument(
        "--quantization-method",
        choices=["dynamic", "static", "qat"],
        default="dynamic",
        help="Quantization method",
    )
    parser.add_argument(
        "--export-format",
        choices=["onnx", "torchscript", "tflite"],
        default="onnx",
        help="Export format",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location="cpu")

    # Extract model from checkpoint
    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
    else:
        model_state = checkpoint

    # Create model instance (this should be customized)
    # For now, assuming VITS model
    from src.models.vits import VITS

    model = VITS()  # Add appropriate config
    model.load_state_dict(model_state)

    # Initialize compressor
    compressor = ModelCompressor()

    # Store models for comparison
    original_model = model
    compressed_model = model

    # Apply compression methods
    for method in args.compression_methods:
        if method == "prune":
            logger.info(f"Applying pruning with sparsity {args.sparsity}")
            pruning_config = {
                "sparsity": args.sparsity,
                "exclude_modules": ["embedding", "output"],
            }
            compressed_model = compressor.prune_model(
                compressed_model, pruning_config, structured=False
            )

        elif method == "quantize":
            logger.info(f"Applying {args.quantization_method} quantization")
            quantization_config = {
                "method": args.quantization_method,
                "backend": "fbgemm" if torch.cuda.is_available() else "qnnpack",
            }

            # For static quantization, we'd need calibration data
            compressed_model = compressor.quantize_model(
                compressed_model, quantization_config
            )

        elif method == "distill":
            logger.info("Creating student model for distillation")
            student_model = create_student_model(
                compressed_model, compression_ratio=0.5
            )

            # Note: Actual distillation requires training data
            logger.warning("Knowledge distillation requires training data - skipping")
            compressed_model = student_model

    # Export compressed model
    export_path = output_dir / f"compressed_model.{args.export_format}"
    compressor.export_compressed_model(
        compressed_model, str(export_path), format=args.export_format, optimize=True
    )
    logger.info(f"Exported compressed model to {export_path}")

    # Benchmark if requested
    if args.benchmark:
        logger.info("Running benchmarks...")

        # Create dummy test data
        test_data = []
        for _ in range(10):
            test_data.append({"input": torch.randn(1, 100, 256)})
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)

        results = compressor.benchmark_model(
            original_model, compressed_model, test_loader, metrics=["size", "latency"]
        )

        # Save results
        with open(output_dir / "benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Print summary
        print("\n=== Compression Results ===")
        if "size" in results:
            print(
                f"Model size: {results['size']['original_mb']:.2f} MB -> "
                f"{results['size']['compressed_mb']:.2f} MB "
                f"({results['size']['size_reduction']:.1%} reduction)"
            )
        if "latency" in results:
            print(
                f"Inference latency: {results['latency']['original_ms']:.2f} ms -> "
                f"{results['latency']['compressed_ms']:.2f} ms "
                f"({results['latency']['speedup']:.2f}x speedup)"
            )

    logger.info("Compression complete!")


if __name__ == "__main__":
    main()
