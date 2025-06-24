"""
ONNX Export Module for Tsukuyomi TTS

Exports trained models to ONNX format for deployment in Unity and other environments.
Includes optimization for mobile and edge devices.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers import optimizer

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for ONNX export."""

    model_path: str
    output_dir: str
    opset_version: int = 16
    optimize: bool = True
    quantize: bool = False
    simplify: bool = True
    target_device: str = "cpu"  # cpu, gpu, mobile
    batch_size: int = 1
    max_sequence_length: int = 512
    fp16: bool = False
    int8_quantization: bool = False
    graph_optimization_level: int = 99
    use_external_data: bool = True
    external_data_threshold: int = 1024  # bytes


class ONNXExporter:
    """Export Tsukuyomi TTS models to ONNX format."""

    def __init__(self, config: ExportConfig):
        """
        Initialize ONNX exporter.

        Args:
            config: Export configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Supported export targets
        self.export_targets = {
            "acoustic_model": self._export_acoustic_model,
            "vocoder": self._export_vocoder,
            "phoneme_encoder": self._export_phoneme_encoder,
            "full_pipeline": self._export_full_pipeline,
        }

    def export_model(
        self,
        model: nn.Module,
        target: str,
        model_name: str,
        sample_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Path:
        """
        Export a model to ONNX format.

        Args:
            model: PyTorch model to export
            target: Export target type
            model_name: Name for the exported model
            sample_inputs: Sample inputs for tracing

        Returns:
            Path to exported ONNX model
        """
        if target not in self.export_targets:
            raise ValueError(f"Unknown export target: {target}")

        logger.info(f"Exporting {target} model: {model_name}")

        # Prepare model for export
        model.eval()
        if self.config.target_device == "cpu":
            model = model.cpu()
        else:
            model = model.cuda()

        # Export using target-specific method
        onnx_path = self.export_targets[target](model, model_name, sample_inputs)

        # Optimize if requested
        if self.config.optimize:
            onnx_path = self._optimize_model(onnx_path)

        # Quantize if requested
        if self.config.quantize:
            onnx_path = self._quantize_model(onnx_path)

        # Validate exported model
        self._validate_model(onnx_path, sample_inputs)

        # Generate metadata
        self._generate_metadata(onnx_path, model_name, target)

        logger.info(f"Successfully exported model to: {onnx_path}")
        return onnx_path

    def _export_acoustic_model(
        self,
        model: nn.Module,
        model_name: str,
        sample_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Path:
        """Export acoustic model to ONNX."""
        if sample_inputs is None:
            # Create dummy inputs
            batch_size = self.config.batch_size
            seq_len = 100

            sample_inputs = {
                "phoneme_ids": torch.randint(0, 100, (batch_size, seq_len)),
                "phoneme_lengths": torch.tensor([seq_len] * batch_size),
                "speaker_ids": torch.randint(0, 10, (batch_size,)),
                "emotion_ids": torch.randint(0, 7, (batch_size,)),
                "style_ids": torch.randint(0, 10, (batch_size,)),
            }

        # Define input/output names
        input_names = list(sample_inputs.keys())
        output_names = ["mel_outputs", "durations", "pitch", "energy"]

        # Dynamic axes for variable sequence length
        dynamic_axes = {
            "phoneme_ids": {0: "batch_size", 1: "sequence_length"},
            "phoneme_lengths": {0: "batch_size"},
            "mel_outputs": {0: "batch_size", 2: "mel_length"},
            "durations": {0: "batch_size", 1: "sequence_length"},
        }

        # Export path
        onnx_path = self.output_dir / f"{model_name}_acoustic.onnx"

        # Export to ONNX
        torch.onnx.export(
            model,
            tuple(sample_inputs.values()),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=self.config.opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )

        return onnx_path

    def _export_vocoder(
        self,
        model: nn.Module,
        model_name: str,
        sample_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Path:
        """Export vocoder to ONNX."""
        if sample_inputs is None:
            # Create dummy mel spectrogram
            batch_size = self.config.batch_size
            mel_channels = 80
            mel_length = 200

            sample_inputs = {
                "mel_spectrogram": torch.randn(batch_size, mel_channels, mel_length)
            }

        input_names = ["mel_spectrogram"]
        output_names = ["audio_waveform"]

        dynamic_axes = {
            "mel_spectrogram": {0: "batch_size", 2: "mel_length"},
            "audio_waveform": {0: "batch_size", 1: "audio_length"},
        }

        onnx_path = self.output_dir / f"{model_name}_vocoder.onnx"

        # For vocoder, we might need to handle the model differently
        class VocoderWrapper(nn.Module):
            def __init__(self, vocoder):
                super().__init__()
                self.vocoder = vocoder

            def forward(self, mel_spectrogram):
                return self.vocoder(mel_spectrogram)

        wrapped_model = VocoderWrapper(model)

        torch.onnx.export(
            wrapped_model,
            (sample_inputs["mel_spectrogram"],),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=self.config.opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )

        return onnx_path

    def _export_phoneme_encoder(
        self,
        model: nn.Module,
        model_name: str,
        sample_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Path:
        """Export phoneme encoder to ONNX."""
        if sample_inputs is None:
            batch_size = self.config.batch_size
            seq_len = 50

            sample_inputs = {
                "text_ids": torch.randint(0, 1000, (batch_size, seq_len)),
                "text_lengths": torch.tensor([seq_len] * batch_size),
            }

        input_names = ["text_ids", "text_lengths"]
        output_names = ["phoneme_ids", "phoneme_embeddings"]

        dynamic_axes = {
            "text_ids": {0: "batch_size", 1: "text_length"},
            "phoneme_ids": {0: "batch_size", 1: "phoneme_length"},
            "phoneme_embeddings": {0: "batch_size", 1: "phoneme_length"},
        }

        onnx_path = self.output_dir / f"{model_name}_phoneme_encoder.onnx"

        torch.onnx.export(
            model,
            tuple(sample_inputs.values()),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=self.config.opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )

        return onnx_path

    def _export_full_pipeline(
        self,
        model: nn.Module,
        model_name: str,
        sample_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Path:
        """Export full TTS pipeline as single ONNX model."""
        # For full pipeline, we export each component separately
        # This provides more flexibility for deployment
        logger.info(
            "Full pipeline export - exporting components separately for flexibility"
        )

        output_paths = {}

        # Export text encoder if available
        if hasattr(model, "text_encoder"):
            text_encoder_path = self._export_phoneme_encoder(
                model.text_encoder, f"{model_name}_text_encoder", sample_inputs
            )
            output_paths["text_encoder"] = str(text_encoder_path)

        # Export acoustic model
        if hasattr(model, "acoustic_model"):
            acoustic_path = self._export_acoustic_model(
                model.acoustic_model, f"{model_name}_acoustic", sample_inputs
            )
            output_paths["acoustic_model"] = str(acoustic_path)

        # Export vocoder
        if hasattr(model, "vocoder"):
            vocoder_path = self._export_vocoder(
                model.vocoder, f"{model_name}_vocoder", sample_inputs
            )
            output_paths["vocoder"] = str(vocoder_path)

        # Create pipeline config
        pipeline_config = {
            "pipeline_name": model_name,
            "components": output_paths,
            "config": {
                "sample_rate": getattr(model, "sample_rate", 22050),
                "hop_size": getattr(model, "hop_size", 256),
                "n_mels": getattr(model, "n_mels", 80),
            },
        }

        config_path = self.output_dir / f"{model_name}_pipeline.json"
        with open(config_path, "w") as f:
            json.dump(pipeline_config, f, indent=2)

        logger.info(f"Pipeline components exported. Config saved to {config_path}")
        return config_path

    def _optimize_model(self, onnx_path: Path) -> Path:
        """
        Optimize ONNX model for inference.

        Args:
            onnx_path: Path to ONNX model

        Returns:
            Path to optimized model
        """
        logger.info("Optimizing ONNX model...")

        optimized_path = onnx_path.parent / f"{onnx_path.stem}_optimized.onnx"

        # Load model
        model = onnx.load(str(onnx_path))

        # Apply optimizations based on target device
        if self.config.target_device == "mobile":
            # Mobile-specific optimizations
            from onnxruntime.transformers import optimizer

            optimized_model = optimizer.optimize_model(
                str(onnx_path),
                model_type="bert",  # Adjust based on actual model
                num_heads=0,  # Will be auto-detected
                hidden_size=0,  # Will be auto-detected
                optimization_options=optimizer.FusionOptions("bert"),
            )
            optimized_model.save_model_to_file(str(optimized_path))
        else:
            # General optimizations
            import onnx.optimizer

            passes = [
                "eliminate_nop_pad",
                "eliminate_nop_transpose",
                "eliminate_unused_initializer",
                "fuse_add_bias_into_conv",
                "fuse_bn_into_conv",
                "fuse_consecutive_squeezes",
                "fuse_consecutive_transposes",
                "fuse_matmul_add_bias_into_gemm",
                "fuse_pad_into_conv",
                "fuse_transpose_into_gemm",
            ]

            optimized_model = onnx.optimizer.optimize(model, passes)
            onnx.save(optimized_model, str(optimized_path))

        # Simplify if requested
        if self.config.simplify:
            try:
                import onnxsim

                simplified_model, check = onnxsim.simplify(optimized_model)
                if check:
                    onnx.save(simplified_model, str(optimized_path))
                    logger.info("Model simplified successfully")
            except Exception as e:
                logger.warning(f"Failed to simplify model: {e}")

        return optimized_path

    def _quantize_model(self, onnx_path: Path) -> Path:
        """
        Quantize ONNX model to INT8.

        Args:
            onnx_path: Path to ONNX model

        Returns:
            Path to quantized model
        """
        logger.info("Quantizing ONNX model...")

        quantized_path = onnx_path.parent / f"{onnx_path.stem}_quantized.onnx"

        if self.config.int8_quantization:
            # Dynamic quantization to INT8
            quantize_dynamic(
                str(onnx_path), str(quantized_path), weight_type=QuantType.QInt8
            )
        else:
            # Could implement other quantization strategies
            logger.warning("Only INT8 quantization is currently supported")
            return onnx_path

        return quantized_path

    def _validate_model(
        self, onnx_path: Path, sample_inputs: Optional[Dict[str, torch.Tensor]]
    ):
        """
        Validate exported ONNX model.

        Args:
            onnx_path: Path to ONNX model
            sample_inputs: Sample inputs for validation
        """
        logger.info("Validating ONNX model...")

        # Check model validity
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)

        # Run inference test
        try:
            providers = (
                ["CUDAExecutionProvider"]
                if self.config.target_device == "gpu"
                else ["CPUExecutionProvider"]
            )
            session = ort.InferenceSession(str(onnx_path), providers=providers)

            if sample_inputs:
                # Prepare inputs
                ort_inputs = {}
                for input_meta in session.get_inputs():
                    if input_meta.name in sample_inputs:
                        tensor = sample_inputs[input_meta.name]
                        if isinstance(tensor, torch.Tensor):
                            tensor = tensor.numpy()
                        ort_inputs[input_meta.name] = tensor

                # Run inference
                outputs = session.run(None, ort_inputs)
                logger.info(
                    f"Validation successful. Output shapes: {[out.shape for out in outputs]}"
                )
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise

    def _generate_metadata(self, onnx_path: Path, model_name: str, target: str):
        """
        Generate metadata file for exported model.

        Args:
            onnx_path: Path to ONNX model
            model_name: Model name
            target: Export target type
        """
        metadata = {
            "model_name": model_name,
            "target": target,
            "export_config": {
                "opset_version": self.config.opset_version,
                "optimized": self.config.optimize,
                "quantized": self.config.quantize,
                "target_device": self.config.target_device,
                "fp16": self.config.fp16,
            },
            "input_info": {},
            "output_info": {},
            "file_size_mb": onnx_path.stat().st_size / (1024 * 1024),
        }

        # Get input/output information
        model = onnx.load(str(onnx_path))

        for input_meta in model.graph.input:
            metadata["input_info"][input_meta.name] = {
                "shape": [
                    d.dim_value if d.dim_value > 0 else "dynamic"
                    for d in input_meta.type.tensor_type.shape.dim
                ],
                "dtype": onnx.TensorProto.DataType.Name(
                    input_meta.type.tensor_type.elem_type
                ),
            }

        for output_meta in model.graph.output:
            metadata["output_info"][output_meta.name] = {
                "shape": [
                    d.dim_value if d.dim_value > 0 else "dynamic"
                    for d in output_meta.type.tensor_type.shape.dim
                ],
                "dtype": onnx.TensorProto.DataType.Name(
                    output_meta.type.tensor_type.elem_type
                ),
            }

        # Save metadata
        metadata_path = onnx_path.parent / f"{onnx_path.stem}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to: {metadata_path}")


class UnityExporter:
    """Specialized exporter for Unity integration."""

    def __init__(self, base_exporter: ONNXExporter):
        """
        Initialize Unity exporter.

        Args:
            base_exporter: Base ONNX exporter
        """
        self.base_exporter = base_exporter
        self.unity_dir = base_exporter.output_dir / "unity"
        self.unity_dir.mkdir(parents=True, exist_ok=True)

    def export_for_unity(
        self,
        acoustic_model: nn.Module,
        vocoder: nn.Module,
        model_name: str = "tsukuyomi",
    ) -> Dict[str, Path]:
        """
        Export models optimized for Unity.

        Args:
            acoustic_model: Acoustic model
            vocoder: Vocoder model
            model_name: Base name for exported models

        Returns:
            Dictionary with paths to exported models
        """
        logger.info("Exporting models for Unity...")

        # Configure for Unity (mobile optimization)
        original_config = self.base_exporter.config
        self.base_exporter.config.target_device = "mobile"
        self.base_exporter.config.optimize = True
        self.base_exporter.config.quantize = True
        self.base_exporter.config.int8_quantization = True

        # Export acoustic model
        acoustic_path = self.base_exporter.export_model(
            acoustic_model, "acoustic_model", f"{model_name}_unity"
        )

        # Export vocoder
        vocoder_path = self.base_exporter.export_model(
            vocoder, "vocoder", f"{model_name}_unity"
        )

        # Generate Unity C# wrapper
        self._generate_unity_wrapper(model_name, acoustic_path, vocoder_path)

        # Restore original config
        self.base_exporter.config = original_config

        return {
            "acoustic_model": acoustic_path,
            "vocoder": vocoder_path,
            "unity_wrapper": self.unity_dir / f"{model_name}TTS.cs",
        }

    def _generate_unity_wrapper(
        self, model_name: str, acoustic_path: Path, vocoder_path: Path
    ):
        """Generate C# wrapper for Unity."""
        wrapper_code = f"""
using System;
using System.Linq;
using Unity.Barracuda;
using UnityEngine;

namespace TsukuyomiTTS
{{
    public class {model_name}TTS : MonoBehaviour
    {{
        [SerializeField] private NNModel acousticModel;
        [SerializeField] private NNModel vocoderModel;
        
        private IWorker acousticWorker;
        private IWorker vocoderWorker;
        
        void Start()
        {{
            // Load models
            var acousticNN = ModelLoader.Load(acousticModel);
            var vocoderNN = ModelLoader.Load(vocoderModel);
            
            // Create workers
            acousticWorker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, acousticNN);
            vocoderWorker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, vocoderNN);
        }}
        
        public AudioClip Synthesize(string text, int speakerId = 0)
        {{
            // Convert text to phoneme IDs (implement based on your phonemizer)
            var phonemeIds = TextToPhonemes(text);
            
            // Prepare inputs
            var inputs = new Dictionary<string, Tensor>();
            inputs["phoneme_ids"] = new Tensor(1, phonemeIds.Length, phonemeIds);
            inputs["phoneme_lengths"] = new Tensor(1, 1, new float[] {{ phonemeIds.Length }});
            inputs["speaker_ids"] = new Tensor(1, 1, new float[] {{ speakerId }});
            
            // Run acoustic model
            acousticWorker.Execute(inputs);
            var melOutput = acousticWorker.PeekOutput("mel_outputs");
            
            // Run vocoder
            var vocoderInputs = new Dictionary<string, Tensor>();
            vocoderInputs["mel_spectrogram"] = melOutput;
            vocoderWorker.Execute(vocoderInputs);
            var audioOutput = vocoderWorker.PeekOutput("audio_waveform");
            
            // Convert to AudioClip
            return TensorToAudioClip(audioOutput, 48000);
        }}
        
        private int[] TextToPhonemes(string text)
        {{
            // Implement phoneme conversion
            // This is a placeholder - integrate with your phonemizer
            return new int[0];
        }}
        
        private AudioClip TensorToAudioClip(Tensor tensor, int sampleRate)
        {{
            var samples = tensor.ToReadOnlyArray();
            var audioClip = AudioClip.Create("{model_name}TTS", samples.Length, 1, sampleRate, false);
            audioClip.SetData(samples, 0);
            return audioClip;
        }}
        
        void OnDestroy()
        {{
            acousticWorker?.Dispose();
            vocoderWorker?.Dispose();
        }}
    }}
}}
"""

        wrapper_path = self.unity_dir / f"{model_name}TTS.cs"
        with open(wrapper_path, "w") as f:
            f.write(wrapper_code)

        logger.info(f"Unity wrapper generated at: {wrapper_path}")


def export_for_deployment(
    checkpoint_path: str,
    output_dir: str,
    targets: List[str] = ["acoustic_model", "vocoder"],
    optimization_level: str = "O2",
) -> Dict[str, Path]:
    """
    High-level function to export models for deployment.

    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory
        targets: List of export targets
        optimization_level: O1 (basic), O2 (aggressive), O3 (extreme)

    Returns:
        Dictionary mapping target names to exported paths
    """
    # Configure based on optimization level
    config = ExportConfig(
        model_path=checkpoint_path,
        output_dir=output_dir,
        optimize=optimization_level in ["O2", "O3"],
        quantize=optimization_level == "O3",
        target_device="gpu" if torch.cuda.is_available() else "cpu",
    )

    exporter = ONNXExporter(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    exported_models = {}

    for target in targets:
        if target in checkpoint:
            model = checkpoint[target]
            exported_path = exporter.export_model(
                model, target, Path(checkpoint_path).stem
            )
            exported_models[target] = exported_path

    return exported_models
