"""
Tests for training components
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.hifigan import HiFiGAN
from src.models.vits import VITS
from src.training.fine_tuning import (
    AdapterConfig,
    LoRAConfig,
    apply_adapter,
    apply_lora,
)
from src.training.losses import DiscriminatorLoss, GeneratorLoss, VITSLoss
from src.training.metrics import AudioMetrics, MelSpectrogramMetrics
from src.training.trainer import VITSTrainer


class TestLosses:
    """Test loss functions"""

    def test_generator_loss(self):
        """Test generator loss computation"""
        loss_fn = GeneratorLoss()

        # Create dummy outputs
        disc_outputs = [torch.randn(2, 1, 10) for _ in range(3)]

        # Compute loss
        loss = loss_fn(disc_outputs)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.requires_grad

    def test_discriminator_loss(self):
        """Test discriminator loss computation"""
        loss_fn = DiscriminatorLoss()

        # Create dummy outputs
        disc_real_outputs = [torch.randn(2, 1, 10) for _ in range(3)]
        disc_fake_outputs = [torch.randn(2, 1, 10) for _ in range(3)]

        # Compute loss
        loss = loss_fn(disc_real_outputs, disc_fake_outputs)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_vits_loss(self):
        """Test VITS combined loss"""
        loss_fn = VITSLoss()

        # Create dummy model outputs
        outputs = {
            "y_hat": torch.randn(2, 1, 1000),
            "l_length": torch.tensor(5.0),
            "l_mel": torch.tensor(10.0),
            "ids_slice": torch.tensor([0, 1]),
            "z_p": torch.randn(2, 192, 10),
            "z_q": torch.randn(2, 192, 10),
            "m_p": torch.randn(2, 192, 10),
            "logs_p": torch.randn(2, 192, 10),
            "m_q": torch.randn(2, 192, 10),
            "logs_q": torch.randn(2, 192, 10),
            "z_mask": torch.ones(2, 1, 10),
        }

        y = torch.randn(2, 1, 1000)
        y_mel = torch.randn(2, 80, 10)
        y_hat_mel = torch.randn(2, 80, 10)

        # Compute loss
        total_loss, loss_dict = loss_fn(
            outputs=outputs, y=y, y_mel=y_mel, y_hat_mel=y_hat_mel
        )

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.dim() == 0
        assert isinstance(loss_dict, dict)
        assert "loss_mel" in loss_dict
        assert "loss_kl" in loss_dict
        assert "loss_dur" in loss_dict


class TestMetrics:
    """Test metric computation"""

    def test_mel_spectrogram_metrics(self):
        """Test mel spectrogram metrics"""
        metrics = MelSpectrogramMetrics()

        # Create dummy mel spectrograms
        pred_mel = torch.randn(2, 80, 100)
        target_mel = torch.randn(2, 80, 100)

        # Compute metrics
        result = metrics.compute(pred_mel, target_mel)

        assert isinstance(result, dict)
        assert "mel_mae" in result
        assert "mel_mse" in result
        assert "mel_l1" in result
        assert "mel_l2" in result
        assert all(isinstance(v, float) for v in result.values())

    def test_audio_metrics(self):
        """Test audio metrics"""
        metrics = AudioMetrics()

        # Create dummy audio
        pred_audio = torch.randn(2, 16000)
        target_audio = torch.randn(2, 16000)

        # Compute metrics
        result = metrics.compute(pred_audio, target_audio)

        assert isinstance(result, dict)
        assert "snr" in result
        assert "si_sdr" in result
        assert all(isinstance(v, float) for v in result.values())


class TestFineTuning:
    """Test fine-tuning methods"""

    def test_lora_application(self):
        """Test LoRA application to model"""
        # Create dummy model
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128)
        )

        # Apply LoRA
        config = LoRAConfig(r=8, alpha=16, target_modules=["Linear"])
        lora_layers = apply_lora(model, config)

        assert len(lora_layers) > 0
        for layer in lora_layers:
            assert hasattr(layer, "lora_A")
            assert hasattr(layer, "lora_B")
            assert hasattr(layer, "scaling")

    def test_adapter_application(self):
        """Test Adapter application to model"""
        # Create dummy model
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128)
        )

        # Apply adapters
        config = AdapterConfig(adapter_size=64)
        adapter_layers = apply_adapter(model, config)

        assert len(adapter_layers) > 0
        for layer in adapter_layers:
            assert hasattr(layer, "adapter")


class TestTrainer:
    """Test trainer functionality"""

    @pytest.fixture()
    def trainer_setup(self):
        """Setup trainer with dummy models"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy models
            model = VITS(
                n_vocab=100,
                n_speakers=2,
                hidden_channels=32,
                filter_channels=64,
                n_heads=2,
                n_layers=2,
            )

            vocoder = HiFiGAN(
                in_channels=80,
                upsample_initial_channel=32,
                upsample_rates=[8, 8],
                upsample_kernel_sizes=[16, 16],
            )

            # Create trainer
            trainer = VITSTrainer(
                model=model,
                vocoder=vocoder,
                output_dir=Path(tmpdir),
                config={
                    "batch_size": 2,
                    "learning_rate": 1e-4,
                    "num_epochs": 1,
                    "gradient_accumulation_steps": 1,
                    "eval_steps": 10,
                    "save_steps": 10,
                    "logging_steps": 1,
                    "warmup_steps": 10,
                    "num_workers": 0,
                    "pin_memory": False,
                    "mixed_precision": "no",
                    "gradient_checkpointing": False,
                },
            )

            yield trainer, tmpdir

    def test_trainer_initialization(self, trainer_setup):
        """Test trainer initialization"""
        trainer, _ = trainer_setup

        assert trainer is not None
        assert trainer.model is not None
        assert trainer.vocoder is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_training_step(self, trainer_setup):
        """Test single training step"""
        trainer, _ = trainer_setup

        # Create dummy batch
        batch = {
            "text": torch.randint(0, 100, (2, 20)),
            "text_lengths": torch.tensor([20, 18]),
            "mel": torch.randn(2, 80, 100),
            "mel_lengths": torch.tensor([100, 90]),
            "audio": torch.randn(2, 1, 8000),
            "audio_lengths": torch.tensor([8000, 7200]),
            "speaker_ids": torch.tensor([0, 1]),
        }

        # Run training step
        loss = trainer.training_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_validation_step(self, trainer_setup):
        """Test validation step"""
        trainer, _ = trainer_setup

        # Create dummy batch
        batch = {
            "text": torch.randint(0, 100, (2, 20)),
            "text_lengths": torch.tensor([20, 18]),
            "mel": torch.randn(2, 80, 100),
            "mel_lengths": torch.tensor([100, 90]),
            "audio": torch.randn(2, 1, 8000),
            "audio_lengths": torch.tensor([8000, 7200]),
            "speaker_ids": torch.tensor([0, 1]),
        }

        # Run validation step
        outputs = trainer.validation_step(batch, 0)

        assert isinstance(outputs, dict)
        assert "loss" in outputs
        assert "metrics" in outputs

    def test_checkpoint_saving(self, trainer_setup):
        """Test checkpoint saving"""
        trainer, tmpdir = trainer_setup

        # Save checkpoint
        trainer.save_checkpoint("test_checkpoint")

        # Check files exist
        checkpoint_path = Path(tmpdir) / "test_checkpoint"
        assert checkpoint_path.exists()
        assert (checkpoint_path / "model.pt").exists()
        assert (checkpoint_path / "vocoder.pt").exists()
        assert (checkpoint_path / "optimizer.pt").exists()
        assert (checkpoint_path / "scheduler.pt").exists()
        assert (checkpoint_path / "config.json").exists()

    def test_checkpoint_loading(self, trainer_setup):
        """Test checkpoint loading"""
        trainer, tmpdir = trainer_setup

        # Save checkpoint
        trainer.save_checkpoint("test_checkpoint")

        # Create new trainer and load checkpoint
        model = VITS(
            n_vocab=100,
            n_speakers=2,
            hidden_channels=32,
            filter_channels=64,
            n_heads=2,
            n_layers=2,
        )

        vocoder = HiFiGAN(
            in_channels=80,
            upsample_initial_channel=32,
            upsample_rates=[8, 8],
            upsample_kernel_sizes=[16, 16],
        )

        new_trainer = VITSTrainer(
            model=model, vocoder=vocoder, output_dir=Path(tmpdir), config=trainer.config
        )

        # Load checkpoint
        new_trainer.load_checkpoint("test_checkpoint")

        # Verify state is loaded
        assert new_trainer.global_step == trainer.global_step
        assert new_trainer.epoch == trainer.epoch


class TestDistributed:
    """Test distributed training utilities"""

    def test_ddp_wrapper(self):
        """Test DDP wrapper application"""
        # This test would require multiple GPUs
        # For now, just test that the wrapper can be imported
        from src.training.trainer import setup_distributed

        assert setup_distributed is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
