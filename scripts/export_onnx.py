#!/usr/bin/env python3
"""
Export Tsukuyomi TTS models to ONNX format

Usage:
    python scripts/export_onnx.py --checkpoint path/to/checkpoint.pt --output-dir exported_models/
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.export.onnx_export import ExportConfig, ONNXExporter
from src.models.hifigan import HiFiGAN
from src.models.vits import VITS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Load model from checkpoint"""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine model type from checkpoint
    if "acoustic_model_state_dict" in checkpoint:
        # Full training checkpoint
        model_state = checkpoint["acoustic_model_state_dict"]
        config = checkpoint.get("config", {})
    elif "model_state_dict" in checkpoint:
        # Standard checkpoint format
        model_state = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
    else:
        # Direct state dict
        model_state = checkpoint
        config = {}
    
    # Create model based on architecture
    # For now, assume VITS architecture
    model = VITS(
        n_vocab=config.get("n_vocab", 100),
        n_speakers=config.get("n_speakers", 1),
        hidden_channels=config.get("hidden_channels", 192),
        filter_channels=config.get("filter_channels", 768),
        n_heads=config.get("n_heads", 2),
        n_layers=config.get("n_layers", 6),
        kernel_size=config.get("kernel_size", 3),
        p_dropout=config.get("p_dropout", 0.1),
    )
    
    # Load state dict
    model.load_state_dict(model_state, strict=False)
    model.eval()
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Export Tsukuyomi TTS models to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exported_models",
        help="Output directory for ONNX models"
    )
    parser.add_argument(
        "--model-type",
        choices=["vits", "hifigan", "bigvgan", "full"],
        default="vits",
        help="Type of model to export"
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=16,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize ONNX model"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize model to INT8"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for export"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--target-device",
        choices=["cpu", "gpu", "mobile"],
        default="cpu",
        help="Target device for optimization"
    )
    
    args = parser.parse_args()
    
    # Create export config
    export_config = ExportConfig(
        model_path=args.checkpoint,
        output_dir=args.output_dir,
        opset_version=args.opset_version,
        optimize=args.optimize,
        quantize=args.quantize,
        target_device=args.target_device,
        batch_size=args.batch_size,
        max_sequence_length=args.max_length,
    )
    
    # Create exporter
    exporter = ONNXExporter(export_config)
    
    # Load model
    device = "cuda" if torch.cuda.is_available() and args.target_device == "gpu" else "cpu"
    
    if args.model_type == "vits":
        model, config = load_checkpoint(args.checkpoint, device)
        
        # Create sample inputs for VITS
        sample_inputs = {
            "text": torch.randint(0, 100, (args.batch_size, 50)),
            "text_lengths": torch.tensor([50] * args.batch_size),
            "speaker_ids": torch.tensor([0] * args.batch_size) if config.get("n_speakers", 1) > 1 else None,
        }
        
        # Export model
        output_path = exporter.export_model(
            model,
            target="acoustic_model",
            model_name="tsukuyomi_vits",
            sample_inputs=sample_inputs
        )
        
    elif args.model_type == "hifigan":
        # Load HiFi-GAN vocoder
        vocoder = HiFiGAN()
        if Path(args.checkpoint).exists():
            state_dict = torch.load(args.checkpoint, map_location=device)
            if "generator_state_dict" in state_dict:
                vocoder.load_state_dict(state_dict["generator_state_dict"])
            else:
                vocoder.load_state_dict(state_dict)
        vocoder.eval()
        
        # Sample inputs for vocoder
        sample_inputs = {
            "mel": torch.randn(args.batch_size, 80, 100)  # [B, n_mels, T]
        }
        
        output_path = exporter.export_model(
            vocoder,
            target="vocoder",
            model_name="tsukuyomi_hifigan",
            sample_inputs=sample_inputs
        )
        
    elif args.model_type == "full":
        # Export full pipeline
        model, config = load_checkpoint(args.checkpoint, device)
        
        # For full pipeline, we need both text and mel inputs
        sample_inputs = {
            "text": torch.randint(0, 100, (args.batch_size, 50)),
            "text_lengths": torch.tensor([50] * args.batch_size),
            "mel": torch.randn(args.batch_size, 80, 100),
        }
        
        output_path = exporter.export_model(
            model,
            target="full_pipeline",
            model_name="tsukuyomi_full",
            sample_inputs=sample_inputs
        )
    
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    logger.info(f"Export completed! Model saved to: {output_path}")
    
    # Print usage instructions
    print("\n" + "="*50)
    print("Export successful! To use the ONNX model:")
    print("\nPython (onnxruntime):")
    print("  import onnxruntime as ort")
    print(f"  session = ort.InferenceSession('{output_path}')")
    print("  outputs = session.run(None, inputs)")
    print("\nUnity (Sentis):")
    print(f"  1. Copy {output_path} to Assets/")
    print("  2. Use Unity.Sentis.ModelLoader.Load()")
    print("="*50)


if __name__ == "__main__":
    main()
