"""
Model Management and Versioning System for Tsukuyomi TTS

Features:
- Model versioning with semantic versioning
- Checkpoint management
- Model registry with metadata
- Automatic backup and recovery
- Model comparison and selection
"""

import hashlib
import json
import logging
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
from packaging import version

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a model version."""

    name: str
    version: str
    description: str
    created_at: str
    model_type: str
    architecture: Dict[str, Any]
    training_config: Dict[str, Any]
    dataset_info: Dict[str, Any]
    performance_metrics: Dict[str, float]
    file_size_mb: float
    hash: str
    tags: List[str]
    author: str = "Tsukuyomi Team"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        return cls(**data)


class ModelRegistry:
    """Central registry for managing TTS models."""

    def __init__(self, registry_dir: Union[str, Path]):
        """
        Initialize model registry.

        Args:
            registry_dir: Directory to store registry and models
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir = self.registry_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.registry_file = self.registry_dir / "registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Dict[str, ModelMetadata]]:
        """Load registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r") as f:
                    data = json.load(f)

                # Convert to ModelMetadata objects
                registry = {}
                for model_name, versions in data.items():
                    registry[model_name] = {}
                    for ver, metadata in versions.items():
                        registry[model_name][ver] = ModelMetadata.from_dict(metadata)

                return registry
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                return {}
        return {}

    def _save_registry(self):
        """Save registry to file."""
        # Convert to serializable format
        data = {}
        for model_name, versions in self.registry.items():
            data[model_name] = {}
            for ver, metadata in versions.items():
                data[model_name][ver] = metadata.to_dict()

        with open(self.registry_file, "w") as f:
            json.dump(data, f, indent=2)

    def register_model(
        self,
        model: nn.Module,
        metadata: ModelMetadata,
        checkpoint_data: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Register a new model version.

        Args:
            model: PyTorch model
            metadata: Model metadata
            checkpoint_data: Additional checkpoint data

        Returns:
            Path to saved model
        """
        # Validate version (metadataがdictの場合の対応)
        version = (
            metadata.get("version", "1.0.0")
            if isinstance(metadata, dict)
            else getattr(metadata, "version", "1.0.0")
        )
        if not self._is_valid_version(version):
            raise ValueError(f"Invalid version format: {version}")

        # Check if version already exists
        if metadata.name in self.registry:
            if metadata.version in self.registry[metadata.name]:
                raise ValueError(
                    f"Version {metadata.version} already exists for {metadata.name}"
                )

        # Create model directory
        model_dir = self.models_dir / metadata.name / metadata.version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / "model.pt"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "metadata": metadata.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }

        if checkpoint_data:
            checkpoint.update(checkpoint_data)

        torch.save(checkpoint, model_path)

        # Calculate hash
        metadata.hash = self._calculate_file_hash(model_path)
        metadata.file_size_mb = model_path.stat().st_size / (1024 * 1024)

        # Save metadata separately
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Update registry
        if metadata.name not in self.registry:
            self.registry[metadata.name] = {}
        self.registry[metadata.name][metadata.version] = metadata
        self._save_registry()

        logger.info(f"Registered model: {metadata.name} v{metadata.version}")
        return model_path

    def get_model(
        self, name: str, version: Optional[str] = None, tag: Optional[str] = None
    ) -> Tuple[nn.Module, ModelMetadata]:
        """
        Load a model from registry.

        Args:
            name: Model name
            version: Specific version (if None, uses latest)
            tag: Tag to filter by

        Returns:
            Tuple of (model, metadata)
        """
        if name not in self.registry:
            raise ValueError(f"Model {name} not found in registry")

        # Get version
        if version is None:
            # Get latest version
            versions = list(self.registry[name].keys())
            versions.sort(key=lambda v: version.parse(v), reverse=True)

            if tag:
                # Filter by tag
                for v in versions:
                    if tag in self.registry[name][v].tags:
                        version = v
                        break
                if version is None:
                    raise ValueError(f"No version found with tag: {tag}")
            else:
                version = versions[0]

        if version not in self.registry[name]:
            raise ValueError(f"Version {version} not found for {name}")

        # Load model
        metadata = self.registry[name][version]
        model_path = self.models_dir / name / version / "model.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu")

        # Recreate model architecture
        model = self._create_model_from_metadata(metadata)
        model.load_state_dict(checkpoint["model_state_dict"])

        logger.info(f"Loaded model: {name} v{version}")
        return model, metadata

    def list_models(
        self, name: Optional[str] = None, tag: Optional[str] = None
    ) -> pd.DataFrame:
        """
        List available models.

        Args:
            name: Filter by model name
            tag: Filter by tag

        Returns:
            DataFrame with model information
        """
        models = []

        for model_name, versions in self.registry.items():
            if name and model_name != name:
                continue

            for ver, metadata in versions.items():
                if tag and tag not in metadata.tags:
                    continue

                models.append(
                    {
                        "name": model_name,
                        "version": ver,
                        "created_at": metadata.created_at,
                        "description": metadata.description,
                        "tags": ", ".join(metadata.tags),
                        "size_mb": f"{metadata.file_size_mb:.1f}",
                        "performance": metadata.performance_metrics.get(
                            "overall_score", 0
                        ),
                    }
                )

        df = pd.DataFrame(models)
        if not df.empty:
            df = df.sort_values(["name", "version"], ascending=[True, False])

        return df

    def compare_models(
        self, model1: Tuple[str, str], model2: Tuple[str, str]
    ) -> pd.DataFrame:
        """
        Compare two model versions.

        Args:
            model1: (name, version) tuple
            model2: (name, version) tuple

        Returns:
            Comparison DataFrame
        """
        meta1 = self.registry[model1[0]][model1[1]]
        meta2 = self.registry[model2[0]][model2[1]]

        comparison = []

        # Basic info
        comparison.append(
            {"Metric": "Version", model1[0]: meta1.version, model2[0]: meta2.version}
        )

        comparison.append(
            {
                "Metric": "Created",
                model1[0]: meta1.created_at,
                model2[0]: meta2.created_at,
            }
        )

        comparison.append(
            {
                "Metric": "Size (MB)",
                model1[0]: f"{meta1.file_size_mb:.1f}",
                model2[0]: f"{meta2.file_size_mb:.1f}",
            }
        )

        # Performance metrics
        metrics1 = meta1.performance_metrics
        metrics2 = meta2.performance_metrics

        for metric in set(metrics1.keys()) | set(metrics2.keys()):
            comparison.append(
                {
                    "Metric": metric,
                    model1[0]: f"{metrics1.get(metric, 0):.3f}",
                    model2[0]: f"{metrics2.get(metric, 0):.3f}",
                }
            )

        return pd.DataFrame(comparison)

    def tag_model(self, name: str, version: str, tags: List[str]):
        """Add tags to a model version."""
        if name not in self.registry or version not in self.registry[name]:
            raise ValueError(f"Model {name} v{version} not found")

        metadata = self.registry[name][version]
        metadata.tags.extend(tags)
        metadata.tags = list(set(metadata.tags))  # Remove duplicates

        self._save_registry()

        # Update metadata file
        model_dir = self.models_dir / name / version
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

    def delete_model(self, name: str, version: str, confirm: bool = False):
        """Delete a model version."""
        if not confirm:
            raise ValueError("Set confirm=True to delete model")

        if name not in self.registry or version not in self.registry[name]:
            raise ValueError(f"Model {name} v{version} not found")

        # Remove from registry
        del self.registry[name][version]
        if not self.registry[name]:  # No versions left
            del self.registry[name]

        # Delete files
        model_dir = self.models_dir / name / version
        if model_dir.exists():
            shutil.rmtree(model_dir)

        self._save_registry()
        logger.info(f"Deleted model: {name} v{version}")

    def export_model(
        self,
        name: str,
        version: str,
        output_path: Union[str, Path],
        include_code: bool = False,
    ):
        """Export model as a package."""
        if name not in self.registry or version not in self.registry[name]:
            raise ValueError(f"Model {name} v{version} not found")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy model files
            model_dir = self.models_dir / name / version
            export_dir = temp_path / f"{name}_v{version}"
            shutil.copytree(model_dir, export_dir)

            # Add inference script
            self._create_inference_script(export_dir, name, version)

            # Create requirements file
            self._create_requirements_file(export_dir)

            # Include source code if requested
            if include_code:
                code_dir = export_dir / "src"
                # Copy minimal required source files
                # This would be customized based on actual needs

            # Create zip archive
            shutil.make_archive(str(output_path).replace(".zip", ""), "zip", temp_path)

        logger.info(f"Exported model to: {output_path}")

    def _is_valid_version(self, ver: str) -> bool:
        """Check if version string is valid."""
        try:
            version.parse(ver)
            return True
        except:
            return False

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _create_model_from_metadata(self, metadata: ModelMetadata) -> nn.Module:
        """Recreate model from metadata."""
        # This would need to be implemented based on your actual model architecture
        # For now, return a mock model
        logger.warning("Model recreation from metadata not fully implemented")

        # Import your actual model classes
        from ..models.ultimate_acoustic_model import UltimateAcousticModel

        # Create model based on metadata
        if metadata.model_type == "acoustic_model":
            model = UltimateAcousticModel(**metadata.architecture)
        else:
            raise ValueError(f"Unknown model type: {metadata.model_type}")

        return model

    def _create_inference_script(self, export_dir: Path, name: str, version: str):
        """Create standalone inference script."""
        script = f"""#!/usr/bin/env python3
\"\"\"
Inference script for {name} v{version}
Auto-generated by Tsukuyomi Model Manager
\"\"\"

import torch
import numpy as np
from pathlib import Path
import json
import argparse

def load_model(model_path="model.pt"):
    \"\"\"Load the TTS model.\"\"\"
    checkpoint = torch.load(model_path, map_location="cpu")
    # Model loading logic here
    return checkpoint

def synthesize(text, speaker_id=0, output_path="output.wav"):
    \"\"\"Synthesize speech from text.\"\"\"
    model = load_model()
    # Synthesis logic here
    print(f"Synthesized: {{text}} -> {{output_path}}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS Inference")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--speaker", type=int, default=0, help="Speaker ID")
    parser.add_argument("--output", default="output.wav", help="Output file")
    
    args = parser.parse_args()
    synthesize(args.text, args.speaker, args.output)
"""

        script_path = export_dir / "inference.py"
        with open(script_path, "w") as f:
            f.write(script)

        # Make executable
        script_path.chmod(0o755)

    def _create_requirements_file(self, export_dir: Path):
        """Create requirements.txt for exported model."""
        requirements = [
            "torch>=2.0.0",
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "librosa>=0.9.0",
            "soundfile>=0.10.0",
            "tqdm>=4.62.0",
        ]

        req_path = export_dir / "requirements.txt"
        with open(req_path, "w") as f:
            f.write("\n".join(requirements))


class ModelVersionManager:
    """Manage model versions with semantic versioning."""

    def __init__(self, registry: ModelRegistry):
        """
        Initialize version manager.

        Args:
            registry: Model registry instance
        """
        self.registry = registry

    def create_version(
        self,
        model: nn.Module,
        name: str,
        version_type: str = "patch",
        description: str = "",
        training_config: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Create a new model version.

        Args:
            model: PyTorch model
            name: Model name
            version_type: "major", "minor", or "patch"
            description: Version description
            training_config: Training configuration
            performance_metrics: Performance metrics
            tags: Optional tags

        Returns:
            New version string
        """
        # Get current versions
        if name in self.registry.registry:
            versions = list(self.registry.registry[name].keys())
            versions.sort(key=lambda v: version.parse(v), reverse=True)
            current_version = versions[0] if versions else "0.0.0"
        else:
            current_version = "0.0.0"

        # Calculate new version
        new_version = self._increment_version(current_version, version_type)

        # Create metadata
        metadata = ModelMetadata(
            name=name,
            version=new_version,
            description=description,
            created_at=datetime.now().isoformat(),
            model_type=model.__class__.__name__.lower(),
            architecture=self._extract_architecture(model),
            training_config=training_config or {},
            dataset_info={},  # To be filled by user
            performance_metrics=performance_metrics or {},
            file_size_mb=0,  # Will be updated during registration
            hash="",  # Will be updated during registration
            tags=tags or [],
        )

        # Register model
        self.registry.register_model(model, metadata)

        return new_version

    def _increment_version(self, current: str, version_type: str) -> str:
        """Increment version number."""
        v = version.parse(current)
        major, minor, patch = v.major, v.minor, v.micro

        if version_type == "major":
            return f"{major + 1}.0.0"
        elif version_type == "minor":
            return f"{major}.{minor + 1}.0"
        elif version_type == "patch":
            return f"{major}.{minor}.{patch + 1}"
        else:
            raise ValueError(f"Invalid version type: {version_type}")

    def _extract_architecture(self, model: nn.Module) -> Dict[str, Any]:
        """Extract model architecture information."""
        arch = {
            "class": model.__class__.__name__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        }

        # Try to extract more specific architecture details
        if hasattr(model, "config"):
            arch["config"] = model.config

        return arch


class RemoteModelHub:
    """Interface to remote model hub (e.g., Hugging Face)."""

    def __init__(self, hub_url: str = "https://huggingface.co"):
        """
        Initialize remote hub interface.

        Args:
            hub_url: Base URL for model hub
        """
        self.hub_url = hub_url

    def download_model(
        self, model_id: str, save_path: Union[str, Path], token: Optional[str] = None
    ) -> Path:
        """
        Download model from remote hub.

        Args:
            model_id: Model identifier (e.g., "org/model")
            save_path: Local save path
            token: Authentication token

        Returns:
            Path to downloaded model
        """
        # This would integrate with actual model hub API
        # For now, it's a placeholder
        logger.info(f"Downloading model: {model_id}")

        # Mock implementation
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Would actually download from hub
        # requests.get(f"{self.hub_url}/{model_id}/resolve/main/model.pt")

        return save_path

    def upload_model(
        self, model_path: Path, model_id: str, token: str, private: bool = False
    ):
        """Upload model to remote hub."""
        # This would integrate with actual model hub API
        logger.info(f"Uploading model to: {model_id}")

        # Mock implementation
        # Would actually upload to hub
