"""Tests for model management and versioning."""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.utils.model_manager import (
    ModelMetadata,
    ModelRegistry,
    ModelVersionManager,
    RemoteModelHub,
)


class DummyModel(nn.Module):
    """Dummy model for testing."""

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.linear(x)


class TestModelMetadata:
    """Test ModelMetadata class."""

    def test_metadata_creation(self):
        """Test creating metadata."""
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            description="Test model",
            created_at=datetime.now().isoformat(),
            model_type="acoustic_model",
            architecture={"hidden_size": 256},
            training_config={"batch_size": 32},
            dataset_info={"hours": 100},
            performance_metrics={"mos": 4.5},
            file_size_mb=150.5,
            hash="abc123",
            tags=["test", "v1"],
        )

        assert metadata.name == "test_model"
        assert metadata.version == "1.0.0"
        assert metadata.tags == ["test", "v1"]

    def test_metadata_serialization(self):
        """Test metadata serialization."""
        metadata = ModelMetadata(
            name="test",
            version="1.0.0",
            description="Test",
            created_at="2024-01-01",
            model_type="test",
            architecture={},
            training_config={},
            dataset_info={},
            performance_metrics={},
            file_size_mb=10.0,
            hash="test",
            tags=[],
        )

        # To dict
        data = metadata.to_dict()
        assert isinstance(data, dict)
        assert data["name"] == "test"

        # From dict
        metadata2 = ModelMetadata.from_dict(data)
        assert metadata2.name == metadata.name
        assert metadata2.version == metadata.version


class TestModelRegistry:
    """Test ModelRegistry class."""

    @pytest.fixture
    def temp_registry_dir(self):
        """Create temporary registry directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def registry(self, temp_registry_dir):
        """Create registry instance."""
        return ModelRegistry(temp_registry_dir)

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return ModelMetadata(
            name="tsukuyomi",
            version="1.0.0",
            description="Test TTS model",
            created_at=datetime.now().isoformat(),
            model_type="acoustic_model",
            architecture={"hidden_size": 256},
            training_config={"epochs": 100},
            dataset_info={"hours": 100},
            performance_metrics={"mos": 4.5, "rtf": 0.05},
            file_size_mb=0,  # Will be updated
            hash="",  # Will be updated
            tags=["production", "v1"],
        )

    def test_register_model(self, registry, sample_metadata):
        """Test registering a model."""
        model = DummyModel()

        model_path = registry.register_model(model, sample_metadata)

        assert model_path.exists()
        assert "tsukuyomi" in registry.registry
        assert "1.0.0" in registry.registry["tsukuyomi"]

        # Check files were created
        model_dir = registry.models_dir / "tsukuyomi" / "1.0.0"
        assert (model_dir / "model.pt").exists()
        assert (model_dir / "metadata.json").exists()

    def test_get_model(self, registry, sample_metadata):
        """Test retrieving a model."""
        # Register model first
        model = DummyModel()
        registry.register_model(model, sample_metadata)

        # Get model
        loaded_model, loaded_metadata = registry.get_model("tsukuyomi", "1.0.0")

        assert isinstance(loaded_model, nn.Module)
        assert loaded_metadata.name == "tsukuyomi"
        assert loaded_metadata.version == "1.0.0"

    def test_get_latest_model(self, registry):
        """Test getting latest model version."""
        model = DummyModel()

        # Register multiple versions
        for version in ["1.0.0", "1.1.0", "1.0.1", "2.0.0"]:
            metadata = ModelMetadata(
                name="test",
                version=version,
                description=f"Version {version}",
                created_at=datetime.now().isoformat(),
                model_type="test",
                architecture={},
                training_config={},
                dataset_info={},
                performance_metrics={},
                file_size_mb=0,
                hash="",
                tags=[],
            )
            registry.register_model(model, metadata)

        # Get latest (should be 2.0.0)
        loaded_model, loaded_metadata = registry.get_model("test")
        assert loaded_metadata.version == "2.0.0"

    def test_list_models(self, registry, sample_metadata):
        """Test listing models."""
        model = DummyModel()

        # Register some models
        registry.register_model(model, sample_metadata)

        # Register another version
        sample_metadata.version = "1.1.0"
        sample_metadata.tags = ["experimental"]
        registry.register_model(model, sample_metadata)

        # List all
        df = registry.list_models()
        assert len(df) == 2
        assert "tsukuyomi" in df["name"].values

        # List by tag
        df_prod = registry.list_models(tag="production")
        assert len(df_prod) == 1
        assert df_prod.iloc[0]["version"] == "1.0.0"

    def test_compare_models(self, registry):
        """Test comparing models."""
        model = DummyModel()

        # Register two versions with different metrics
        metadata1 = ModelMetadata(
            name="test",
            version="1.0.0",
            description="Version 1",
            created_at=datetime.now().isoformat(),
            model_type="test",
            architecture={},
            training_config={},
            dataset_info={},
            performance_metrics={"mos": 4.3, "rtf": 0.08},
            file_size_mb=100,
            hash="",
            tags=[],
        )

        metadata2 = ModelMetadata(
            name="test",
            version="2.0.0",
            description="Version 2",
            created_at=datetime.now().isoformat(),
            model_type="test",
            architecture={},
            training_config={},
            dataset_info={},
            performance_metrics={"mos": 4.5, "rtf": 0.05},
            file_size_mb=120,
            hash="",
            tags=[],
        )

        registry.register_model(model, metadata1)
        registry.register_model(model, metadata2)

        # Compare
        comparison = registry.compare_models(("test", "1.0.0"), ("test", "2.0.0"))

        assert len(comparison) > 0
        assert "Version" in comparison["Metric"].values
        assert "mos" in comparison["Metric"].values

    def test_tag_model(self, registry, sample_metadata):
        """Test adding tags to model."""
        model = DummyModel()
        registry.register_model(model, sample_metadata)

        # Add tags
        registry.tag_model("tsukuyomi", "1.0.0", ["stable", "recommended"])

        # Check tags were added
        metadata = registry.registry["tsukuyomi"]["1.0.0"]
        assert "stable" in metadata.tags
        assert "recommended" in metadata.tags
        assert "production" in metadata.tags  # Original tag

    def test_delete_model(self, registry, sample_metadata):
        """Test deleting a model."""
        model = DummyModel()
        model_path = registry.register_model(model, sample_metadata)

        # Delete
        registry.delete_model("tsukuyomi", "1.0.0", confirm=True)

        # Check deletion
        assert "tsukuyomi" not in registry.registry
        assert not model_path.exists()

    def test_export_model(self, registry, sample_metadata, tmp_path):
        """Test exporting a model."""
        model = DummyModel()
        registry.register_model(model, sample_metadata)

        # Export
        export_path = tmp_path / "export.zip"
        registry.export_model("tsukuyomi", "1.0.0", export_path)

        assert export_path.exists()


class TestModelVersionManager:
    """Test ModelVersionManager class."""

    @pytest.fixture
    def version_manager(self, registry):
        """Create version manager."""
        return ModelVersionManager(registry)

    def test_create_patch_version(self, version_manager):
        """Test creating patch version."""
        model = DummyModel()

        # Create initial version
        v1 = version_manager.create_version(model, "test", "patch", "Initial version")
        assert v1 == "0.0.1"

        # Create patch
        v2 = version_manager.create_version(model, "test", "patch", "Bug fix")
        assert v2 == "0.0.2"

    def test_create_minor_version(self, version_manager):
        """Test creating minor version."""
        model = DummyModel()

        # Create initial
        version_manager.create_version(model, "test", "minor", "v1")

        # Create minor
        v2 = version_manager.create_version(model, "test", "minor", "New feature")
        assert v2 == "0.2.0"

    def test_create_major_version(self, version_manager):
        """Test creating major version."""
        model = DummyModel()

        # Create some versions
        version_manager.create_version(model, "test", "minor")
        version_manager.create_version(model, "test", "patch")

        # Create major
        v3 = version_manager.create_version(model, "test", "major", "Breaking changes")
        assert v3 == "1.0.0"

    def test_version_with_metrics(self, version_manager):
        """Test creating version with performance metrics."""
        model = DummyModel()

        version = version_manager.create_version(
            model,
            "test",
            "minor",
            "Improved model",
            performance_metrics={"mos": 4.6, "rtf": 0.04, "speaker_similarity": 0.95},
            tags=["benchmark", "sota"],
        )

        # Check metadata
        registry = version_manager.registry
        metadata = registry.registry["test"][version]
        assert metadata.performance_metrics["mos"] == 4.6
        assert "sota" in metadata.tags


class TestRemoteModelHub:
    """Test RemoteModelHub class."""

    def test_hub_initialization(self):
        """Test hub initialization."""
        hub = RemoteModelHub()
        assert hub.hub_url == "https://huggingface.co"

    def test_download_model_mock(self, tmp_path):
        """Test download model (mocked)."""
        hub = RemoteModelHub()

        # Mock download
        save_path = tmp_path / "model.pt"
        result = hub.download_model("organization/model", save_path)

        # In real implementation, this would download
        assert isinstance(result, Path)
