"""
Benchmarking System for Tsukuyomi TTS

Performance benchmarking including:
- Inference speed (RTF)
- Memory usage
- Quality metrics
- Multi-GPU scaling
"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import GPUtil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import torch
import torch.nn as nn
from tqdm import tqdm

from .metrics import ComprehensiveEvaluator

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""

    model_path: str
    test_dataset: str
    output_dir: str = "benchmark_results"
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None
    num_speakers: List[int] = None
    devices: List[str] = None
    num_warmup_runs: int = 10
    num_benchmark_runs: int = 100
    compute_quality_metrics: bool = True
    profile_memory: bool = True
    test_quantization: bool = False
    test_onnx: bool = False

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16, 32]
        if self.sequence_lengths is None:
            self.sequence_lengths = [50, 100, 200, 400]
        if self.num_speakers is None:
            self.num_speakers = [1, 10, 100, 500]
        if self.devices is None:
            self.devices = ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    # Configuration
    batch_size: int
    sequence_length: int
    num_speakers: int
    device: str
    model_type: str

    # Performance metrics
    inference_time_mean: float
    inference_time_std: float
    rtf_mean: float
    rtf_std: float
    throughput: float  # samples per second

    # Memory metrics
    peak_memory_mb: Optional[float] = None
    gpu_memory_mb: Optional[float] = None

    # Quality metrics
    quality_scores: Optional[Dict[str, float]] = None

    # System info
    cpu_percent: Optional[float] = None
    gpu_utilization: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PerformanceBenchmark:
    """Comprehensive performance benchmarking system."""

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize benchmark system.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize evaluator if needed
        self.evaluator = None
        if config.compute_quality_metrics:
            self.evaluator = ComprehensiveEvaluator()

        # Results storage
        self.results = []

    def benchmark_model(
        self, model: nn.Module, model_type: str = "pytorch"
    ) -> List[BenchmarkResult]:
        """
        Run comprehensive benchmarks on a model.

        Args:
            model: Model to benchmark
            model_type: Type of model (pytorch, onnx, etc.)

        Returns:
            List of benchmark results
        """
        logger.info(f"Starting benchmark for {model_type} model")

        results = []

        # Test different configurations
        for device in self.config.devices:
            if device.startswith("cuda") and not torch.cuda.is_available():
                continue

            logger.info(f"Benchmarking on {device}")

            # Move model to device
            if model_type == "pytorch":
                model = model.to(device)
                model.eval()

            for batch_size in self.config.batch_sizes:
                for seq_len in self.config.sequence_lengths:
                    result = self._benchmark_configuration(
                        model, batch_size, seq_len, device, model_type
                    )
                    results.append(result)

        self.results.extend(results)
        return results

    def _benchmark_configuration(
        self,
        model: nn.Module,
        batch_size: int,
        sequence_length: int,
        device: str,
        model_type: str,
    ) -> BenchmarkResult:
        """Benchmark a specific configuration."""
        logger.info(
            f"Testing: batch_size={batch_size}, seq_len={sequence_length}, device={device}"
        )

        # Create dummy inputs
        inputs = self._create_dummy_inputs(batch_size, sequence_length, device)

        # Warmup
        logger.info("Running warmup...")
        for _ in range(self.config.num_warmup_runs):
            with torch.no_grad():
                if model_type == "pytorch":
                    _ = model(**inputs)
                elif model_type == "onnx":
                    _ = self._run_onnx_model(model, inputs)

        # Clear GPU cache
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Benchmark
        logger.info("Running benchmark...")
        inference_times = []
        memory_usage = []

        # Start monitoring
        initial_memory = self._get_memory_usage(device)

        for _ in tqdm(range(self.config.num_benchmark_runs)):
            # Record CPU/GPU state
            cpu_before = psutil.cpu_percent(interval=None)

            # Time inference
            start_time = time.perf_counter()

            with torch.no_grad():
                if model_type == "pytorch":
                    outputs = model(**inputs)
                elif model_type == "onnx":
                    outputs = self._run_onnx_model(model, inputs)

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            inference_time = end_time - start_time
            inference_times.append(inference_time)

            # Record memory
            current_memory = self._get_memory_usage(device)
            memory_usage.append(current_memory - initial_memory)

        # Calculate statistics
        inference_times = np.array(inference_times)

        # Audio duration (assuming 50ms per frame)
        audio_duration = sequence_length * 0.05 * batch_size

        # RTF calculation
        rtf_values = inference_times / audio_duration

        # Create result
        result = BenchmarkResult(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_speakers=1,  # TODO: Add speaker variation testing
            device=device,
            model_type=model_type,
            inference_time_mean=float(np.mean(inference_times)),
            inference_time_std=float(np.std(inference_times)),
            rtf_mean=float(np.mean(rtf_values)),
            rtf_std=float(np.std(rtf_values)),
            throughput=float(batch_size / np.mean(inference_times)),
            peak_memory_mb=float(np.max(memory_usage)) if memory_usage else None,
            cpu_percent=psutil.cpu_percent(interval=0.1),
        )

        # GPU specific metrics
        if device.startswith("cuda"):
            gpu_id = int(device.split(":")[-1])
            gpus = GPUtil.getGPUs()
            if gpu_id < len(gpus):
                result.gpu_memory_mb = gpus[gpu_id].memoryUsed
                result.gpu_utilization = gpus[gpu_id].load * 100

        # Quality metrics (on subset)
        if self.config.compute_quality_metrics and batch_size == 1:
            result.quality_scores = self._compute_quality_metrics(outputs)

        return result

    def _create_dummy_inputs(
        self, batch_size: int, sequence_length: int, device: str
    ) -> Dict[str, torch.Tensor]:
        """Create dummy inputs for benchmarking."""
        inputs = {
            "phoneme_ids": torch.randint(0, 100, (batch_size, sequence_length)).to(
                device
            ),
            "phoneme_lengths": torch.full((batch_size,), sequence_length).to(device),
            "speaker_ids": torch.randint(0, 10, (batch_size,)).to(device),
            "emotion_ids": torch.randint(0, 7, (batch_size,)).to(device),
            "style_ids": torch.randint(0, 10, (batch_size,)).to(device),
        }

        return inputs

    def _get_memory_usage(self, device: str) -> float:
        """Get current memory usage in MB."""
        if device == "cpu":
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        elif device.startswith("cuda"):
            return torch.cuda.memory_allocated(device) / 1024 / 1024
        return 0.0

    def _run_onnx_model(self, session, inputs: Dict[str, torch.Tensor]) -> Any:
        """Run ONNX model inference."""
        # Convert torch tensors to numpy
        onnx_inputs = {}
        for name, tensor in inputs.items():
            onnx_inputs[name] = tensor.cpu().numpy()

        # Run inference
        outputs = session.run(None, onnx_inputs)
        return outputs

    def _compute_quality_metrics(self, outputs: Any) -> Dict[str, float]:
        """Compute quality metrics for outputs."""
        # This is a placeholder - actual implementation would use real evaluation
        return {"mock_quality": 0.85, "mock_accuracy": 0.92}

    def generate_report(self):
        """Generate comprehensive benchmark report."""
        if not self.results:
            logger.warning("No results to report")
            return

        # Convert to DataFrame
        df = pd.DataFrame([r.to_dict() for r in self.results])

        # Save raw results
        df.to_csv(self.output_dir / "benchmark_results.csv", index=False)

        # Generate visualizations
        self._plot_results(df)

        # Generate summary report
        self._generate_summary_report(df)

    def _plot_results(self, df: pd.DataFrame):
        """Generate visualization plots."""
        plt.style.use("seaborn-v0_8-darkgrid")

        # RTF vs Batch Size
        fig, ax = plt.subplots(figsize=(10, 6))
        for device in df["device"].unique():
            device_df = df[df["device"] == device]
            for seq_len in self.config.sequence_lengths:
                seq_df = device_df[device_df["sequence_length"] == seq_len]
                ax.plot(
                    seq_df["batch_size"],
                    seq_df["rtf_mean"],
                    marker="o",
                    label=f"{device} - seq_len={seq_len}",
                )

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Real-Time Factor (RTF)")
        ax.set_title("RTF Performance vs Batch Size")
        ax.axhline(y=1.0, color="r", linestyle="--", label="Real-time threshold")
        ax.legend()
        ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(self.output_dir / "rtf_vs_batch_size.png", dpi=150)
        plt.close()

        # Memory usage
        if "peak_memory_mb" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot_df = df.pivot_table(
                values="peak_memory_mb",
                index="batch_size",
                columns="sequence_length",
                aggfunc="mean",
            )
            sns.heatmap(pivot_df, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
            ax.set_title("Peak Memory Usage (MB)")
            plt.tight_layout()
            plt.savefig(self.output_dir / "memory_heatmap.png", dpi=150)
            plt.close()

        # Throughput comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        device_throughput = df.groupby("device")["throughput"].mean()
        device_throughput.plot(kind="bar", ax=ax)
        ax.set_ylabel("Throughput (samples/second)")
        ax.set_title("Average Throughput by Device")
        plt.tight_layout()
        plt.savefig(self.output_dir / "throughput_comparison.png", dpi=150)
        plt.close()

    def _generate_summary_report(self, df: pd.DataFrame):
        """Generate text summary report."""
        report = []
        report.append("# Tsukuyomi TTS Benchmark Report\n")
        report.append(f"Generated on: {pd.Timestamp.now()}\n")

        # Overall statistics
        report.append("## Overall Performance\n")
        report.append(
            f"- Mean RTF: {df['rtf_mean'].mean():.3f} ± {df['rtf_mean'].std():.3f}\n"
        )
        report.append(f"- Best RTF: {df['rtf_mean'].min():.3f}\n")
        report.append(f"- Mean Throughput: {df['throughput'].mean():.1f} samples/sec\n")

        # Device comparison
        report.append("\n## Device Comparison\n")
        device_summary = (
            df.groupby("device")
            .agg(
                {
                    "rtf_mean": ["mean", "min"],
                    "throughput": "mean",
                    "peak_memory_mb": "mean",
                }
            )
            .round(3)
        )
        report.append(device_summary.to_string())

        # Optimal configurations
        report.append("\n\n## Optimal Configurations\n")

        # Best RTF configuration
        best_rtf_idx = df["rtf_mean"].idxmin()
        best_rtf = df.loc[best_rtf_idx]
        report.append("\n### Best RTF Configuration\n")
        report.append(f"- Device: {best_rtf['device']}\n")
        report.append(f"- Batch Size: {best_rtf['batch_size']}\n")
        report.append(f"- Sequence Length: {best_rtf['sequence_length']}\n")
        report.append(f"- RTF: {best_rtf['rtf_mean']:.3f}\n")

        # Best throughput configuration
        best_throughput_idx = df["throughput"].idxmax()
        best_throughput = df.loc[best_throughput_idx]
        report.append("\n### Best Throughput Configuration\n")
        report.append(f"- Device: {best_throughput['device']}\n")
        report.append(f"- Batch Size: {best_throughput['batch_size']}\n")
        report.append(
            f"- Throughput: {best_throughput['throughput']:.1f} samples/sec\n"
        )

        # Save report
        with open(self.output_dir / "benchmark_report.md", "w") as f:
            f.writelines(report)

        logger.info(f"Report saved to {self.output_dir}")


def benchmark_tts_system(
    model_path: str, test_dataset: str, output_dir: str = "benchmark_results"
) -> Dict[str, Any]:
    """
    High-level function to benchmark TTS system.

    Args:
        model_path: Path to model checkpoint
        test_dataset: Path to test dataset
        output_dir: Output directory for results

    Returns:
        Dictionary with benchmark results
    """
    config = BenchmarkConfig(
        model_path=model_path, test_dataset=test_dataset, output_dir=output_dir
    )

    benchmark = PerformanceBenchmark(config)

    # Load model
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    model = checkpoint["model"]

    # Run benchmarks
    results = benchmark.benchmark_model(model, "pytorch")

    # Test quantized version if requested
    if config.test_quantization:
        logger.info("Testing quantized model...")
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        quantized_results = benchmark.benchmark_model(
            quantized_model, "pytorch_quantized"
        )
        results.extend(quantized_results)

    # Test ONNX version if requested
    if config.test_onnx:
        logger.info("Testing ONNX model...")
        # Load ONNX runtime session
        import onnxruntime as ort

        onnx_path = Path(model_path).parent / "model.onnx"
        if onnx_path.exists():
            session = ort.InferenceSession(str(onnx_path))
            onnx_results = benchmark.benchmark_model(session, "onnx")
            results.extend(onnx_results)

    # Generate report
    benchmark.generate_report()

    return {
        "results": results,
        "summary": {
            "mean_rtf": np.mean([r.rtf_mean for r in results]),
            "best_rtf": min(r.rtf_mean for r in results),
            "mean_throughput": np.mean([r.throughput for r in results]),
        },
    }
