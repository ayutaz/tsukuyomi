#!/usr/bin/env python3
"""Main training script for Tsukuyomi TTS"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.data.ljspeech_dataset import create_ljspeech_datasets
from src.models.vits import VITS
from src.models.hifigan import HiFiGAN
from src.models.f0_bert import F0BERT
from src.models.xphonebert import XPhoneBERT
from src.training.losses import MultiTaskLoss, DiscriminatorLoss
from src.training.metrics import TrainingMetrics, get_evaluation_metrics
from src.training.finetune import (
    FineTuningConfig,
    load_checkpoint_for_finetuning,
    freeze_model_weights,
    save_finetuned_model,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTSTrainer:
    """Main trainer class for TTS models"""
    
    def __init__(self, config, rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        
        # Initialize models
        self.setup_models()
        
        # Initialize data loaders
        self.setup_data_loaders()
        
        # Initialize optimizers and schedulers
        self.setup_optimizers()
        
        # Initialize losses and metrics
        self.setup_losses_and_metrics()
        
        # Initialize logging
        if rank == 0:
            self.setup_logging()
            
        self.epoch = 0
        self.global_step = 0
        
    def setup_models(self):
        """Initialize all models"""
        logger.info("Setting up models...")
        
        # Text encoder (XPhoneBERT)
        if self.config.models.xphonebert.enabled:
            self.xphonebert = XPhoneBERT(
                model_name=self.config.models.xphonebert.model_name,
                hidden_size=self.config.models.xphonebert.hidden_size,
                num_layers=self.config.models.xphonebert.num_layers,
                num_heads=self.config.models.xphonebert.num_heads,
            ).to(self.device)
        else:
            self.xphonebert = None
            
        # F0 predictor
        if self.config.models.f0_bert.enabled:
            self.f0_bert = F0BERT(
                hidden_size=self.config.models.f0_bert.hidden_size,
                num_layers=self.config.models.f0_bert.num_layers,
                num_heads=self.config.models.f0_bert.num_heads,
                pitch_bins=self.config.models.f0_bert.pitch_bins,
            ).to(self.device)
        else:
            self.f0_bert = None
            
        # Main acoustic model
        if self.config.models.acoustic_model == "vits":
            self.acoustic_model = VITS(
                n_vocab=self.config.models.vits.n_vocab,
                n_speakers=self.config.models.vits.n_speakers,
                hidden_channels=self.config.models.vits.hidden_channels,
                filter_channels=self.config.models.vits.filter_channels,
                n_heads=self.config.models.vits.n_heads,
                n_layers=self.config.models.vits.n_layers,
                kernel_size=self.config.models.vits.kernel_size,
                p_dropout=self.config.models.vits.p_dropout,
                n_flows=self.config.models.vits.get("n_flows", 4),
                upsample_rates=self.config.models.bigvgan.upsample_rates,
                upsample_kernel_sizes=self.config.models.bigvgan.upsample_kernel_sizes,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown acoustic model: {self.config.models.acoustic_model}")
            
        # Vocoder (if separate from acoustic model)
        if self.config.models.get("separate_vocoder", False):
            self.vocoder = HiFiGAN(
                in_channels=self.config.models.bigvgan.num_mels,
                upsample_rates=self.config.models.bigvgan.upsample_rates,
                upsample_kernel_sizes=self.config.models.bigvgan.upsample_kernel_sizes,
                resblock_kernel_sizes=self.config.models.bigvgan.resblock_kernel_sizes,
                resblock_dilation_sizes=self.config.models.bigvgan.resblock_dilation_sizes,
            ).to(self.device)
        else:
            self.vocoder = None
            
        # Load checkpoint if specified
        if self.config.training.resume_from:
            self.load_checkpoint(self.config.training.resume_from)
            
        # Setup fine-tuning if specified
        if hasattr(self.config, "finetuning") and self.config.finetuning.enabled:
            self.setup_finetuning()
            
        # Distributed training
        if self.world_size > 1:
            self.wrap_distributed()
            
    def setup_finetuning(self):
        """Setup fine-tuning configuration"""
        ft_config = FineTuningConfig(
            base_model_path=self.config.finetuning.base_model_path,
            output_dir=self.config.training.checkpoint_dir,
            strategy=self.config.finetuning.strategy,
            lora_rank=self.config.finetuning.get("lora_rank", 64),
            freeze_modules=self.config.finetuning.get("freeze_modules", []),
            learning_rate=self.config.finetuning.get("learning_rate", 1e-4),
        )
        
        self.acoustic_model, _ = load_checkpoint_for_finetuning(
            self.acoustic_model,
            self.config.finetuning.base_model_path,
            ft_config
        )
        
        self.ft_config = ft_config
        
    def wrap_distributed(self):
        """Wrap models for distributed training"""
        if self.xphonebert is not None:
            self.xphonebert = DDP(self.xphonebert, device_ids=[self.rank])
        if self.f0_bert is not None:
            self.f0_bert = DDP(self.f0_bert, device_ids=[self.rank])
        self.acoustic_model = DDP(self.acoustic_model, device_ids=[self.rank])
        if self.vocoder is not None:
            self.vocoder = DDP(self.vocoder, device_ids=[self.rank])
            
    def setup_data_loaders(self):
        """Setup data loaders"""
        logger.info("Setting up data loaders...")
        
        self.train_loader, self.val_loader = create_ljspeech_datasets(
            data_dir=self.config.data.data_dir,
            sample_rate=self.config.data.sample_rate,
            n_mels=self.config.data.n_mels,
            batch_size=self.config.training.batch_size // self.world_size,
            num_workers=self.config.data.num_workers,
            n_fft=self.config.data.n_fft,
            hop_length=self.config.data.hop_length,
            win_length=self.config.data.win_length,
            f_min=self.config.data.get("f_min", 0),
            f_max=self.config.data.get("f_max", 8000),
        )
        
    def setup_optimizers(self):
        """Setup optimizers and schedulers"""
        logger.info("Setting up optimizers...")
        
        # Collect parameters
        params = []
        if self.xphonebert is not None:
            params.extend(self.xphonebert.parameters())
        if self.f0_bert is not None:
            params.extend(self.f0_bert.parameters())
        params.extend(self.acoustic_model.parameters())
        if self.vocoder is not None:
            params.extend(self.vocoder.parameters())
            
        # Filter trainable parameters
        trainable_params = [p for p in params if p.requires_grad]
        
        # Create optimizer
        opt_config = self.config.training.optimizers.default
        if opt_config.type == "adamw":
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=opt_config.lr,
                betas=(opt_config.beta1, opt_config.beta2),
                eps=opt_config.eps,
                weight_decay=opt_config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.type}")
            
        # Create scheduler
        sched_config = self.config.training.schedulers.default
        if sched_config.type == "exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=sched_config.gamma
            )
        elif sched_config.type == "cosine_annealing_warm_restarts":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=sched_config.T_0,
                T_mult=sched_config.T_mult,
                eta_min=sched_config.eta_min,
            )
        else:
            self.scheduler = None
            
        # Discriminator optimizers (for GAN training)
        if hasattr(self.acoustic_model, "mpd") and hasattr(self.acoustic_model, "msd"):
            disc_params = []
            disc_params.extend(self.acoustic_model.mpd.parameters())
            disc_params.extend(self.acoustic_model.msd.parameters())
            
            self.disc_optimizer = torch.optim.AdamW(
                disc_params,
                lr=opt_config.lr,
                betas=(opt_config.beta1, opt_config.beta2),
                eps=opt_config.eps,
                weight_decay=0.0,  # No weight decay for discriminator
            )
        else:
            self.disc_optimizer = None
            
    def setup_losses_and_metrics(self):
        """Setup loss functions and metrics"""
        logger.info("Setting up losses and metrics...")
        
        self.criterion = MultiTaskLoss(
            loss_weights=self.config.training.loss_weights,
            use_mel_loss=True,
            use_kl_loss=True,
            use_duration_loss=True,
            use_pitch_loss=self.f0_bert is not None,
            use_gan_loss=self.disc_optimizer is not None,
        )
        
        if self.disc_optimizer is not None:
            self.disc_criterion = DiscriminatorLoss(loss_type="hinge")
            
        self.train_metrics = TrainingMetrics()
        self.val_metrics = TrainingMetrics()
        
        self.eval_metrics = get_evaluation_metrics({
            "use_mcd": True,
            "use_pitch_correlation": self.f0_bert is not None,
            "use_speaker_similarity": self.config.models.vits.n_speakers > 1,
        })
        
    def setup_logging(self):
        """Setup logging and visualization"""
        self.writer = SummaryWriter(self.config.paths.tensorboard)
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Forward pass through models
        outputs = {}
        
        # Text encoding (if using XPhoneBERT)
        if self.xphonebert is not None:
            text_features = self.xphonebert(
                batch["text"],
                batch["text_lengths"]
            )
            outputs["text_features"] = text_features
        else:
            text_features = None
            
        # Pitch prediction (if using F0-BERT)
        if self.f0_bert is not None and text_features is not None:
            pitch_outputs = self.f0_bert(
                text_features,
                batch["text_lengths"]
            )
            outputs.update(pitch_outputs)
            
        # Main acoustic model forward
        acoustic_outputs = self.acoustic_model(
            text=batch["text"],
            text_lengths=batch["text_lengths"],
            mel=batch["mel"],
            mel_lengths=batch["mel_lengths"],
            speaker_ids=batch.get("speaker_id"),
        )
        outputs.update(acoustic_outputs)
        
        # Discriminator forward (if using GAN)
        if self.disc_optimizer is not None and "audio" in acoustic_outputs:
            # Discriminator on real audio
            real_audio = batch["audio"].unsqueeze(1)  # Add channel dimension
            disc_real = []
            disc_fake = []
            fmap_real = []
            fmap_fake = []
            
            # Multi-period discriminator
            y_d_r, y_d_g, fmap_r, fmap_g = self.acoustic_model.mpd(
                real_audio,
                acoustic_outputs["audio"].detach()
            )
            disc_real.extend(y_d_r)
            disc_fake.extend(y_d_g)
            fmap_real.extend(fmap_r)
            fmap_fake.extend(fmap_g)
            
            # Multi-scale discriminator
            y_d_r, y_d_g, fmap_r, fmap_g = self.acoustic_model.msd(
                real_audio,
                acoustic_outputs["audio"].detach()
            )
            disc_real.extend(y_d_r)
            disc_fake.extend(y_d_g)
            fmap_real.extend(fmap_r)
            fmap_fake.extend(fmap_g)
            
            # Discriminator loss
            disc_loss = self.disc_criterion(disc_real, disc_fake)
            
            # Update discriminator
            self.disc_optimizer.zero_grad()
            disc_loss.backward()
            self.disc_optimizer.step()
            
            # Generator discriminator outputs (for generator loss)
            disc_outputs = {
                "fake": disc_fake,
                "fmap_real": fmap_real,
                "fmap_fake": fmap_fake,
            }
        else:
            disc_outputs = None
            disc_loss = None
            
        # Calculate losses
        targets = {
            "mel": batch["mel"],
            "duration": batch.get("duration"),
            "pitch": batch.get("pitch"),
        }
        
        losses = self.criterion(outputs, targets, disc_outputs)
        
        # Backward pass
        self.optimizer.zero_grad()
        losses["total"].backward()
        
        # Gradient clipping
        if self.config.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.acoustic_model.parameters() if p.requires_grad],
                self.config.training.gradient_clip
            )
            
        self.optimizer.step()
        
        # Prepare metrics
        metrics = {k: v.item() for k, v in losses.items()}
        if disc_loss is not None:
            metrics["disc_loss"] = disc_loss.item()
            
        return metrics
        
    @torch.no_grad()
    def val_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single validation step"""
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Forward pass
        outputs = {}
        
        if self.xphonebert is not None:
            text_features = self.xphonebert(
                batch["text"],
                batch["text_lengths"]
            )
            outputs["text_features"] = text_features
        else:
            text_features = None
            
        if self.f0_bert is not None and text_features is not None:
            pitch_outputs = self.f0_bert(
                text_features,
                batch["text_lengths"]
            )
            outputs.update(pitch_outputs)
            
        acoustic_outputs = self.acoustic_model(
            text=batch["text"],
            text_lengths=batch["text_lengths"],
            mel=batch["mel"],
            mel_lengths=batch["mel_lengths"],
            speaker_ids=batch.get("speaker_id"),
        )
        outputs.update(acoustic_outputs)
        
        # Calculate losses
        targets = {
            "mel": batch["mel"],
            "duration": batch.get("duration"),
            "pitch": batch.get("pitch"),
        }
        
        losses = self.criterion(outputs, targets)
        
        # Calculate evaluation metrics
        eval_results = {}
        if "mcd" in self.eval_metrics and "mel" in outputs:
            eval_results["mcd"] = self.eval_metrics["mcd"](
                outputs["mel"],
                batch["mel"],
                batch["mel_lengths"]
            )
            
        if "pitch_correlation" in self.eval_metrics and "pitch" in outputs:
            eval_results["pitch_corr"] = self.eval_metrics["pitch_correlation"](
                outputs["pitch"],
                batch["pitch"]
            )
            
        # Prepare metrics
        metrics = {k: v.item() for k, v in losses.items()}
        metrics.update(eval_results)
        
        return metrics
        
    def train_epoch(self):
        """Train for one epoch"""
        self.acoustic_model.train()
        if self.xphonebert is not None:
            self.xphonebert.train()
        if self.f0_bert is not None:
            self.f0_bert.train()
            
        self.train_metrics.reset()
        
        for batch_idx, batch in enumerate(self.train_loader):
            metrics = self.train_step(batch)
            self.train_metrics.update(metrics)
            
            # Logging
            if self.rank == 0 and batch_idx % self.config.training.logging.log_interval == 0:
                avg_metrics = self.train_metrics.compute_average()
                self.log_metrics(avg_metrics, "train")
                
                logger.info(
                    f"Epoch {self.epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {avg_metrics['loss']:.4f}"
                )
                
            self.global_step += 1
            
            # Clear cache periodically
            if hasattr(self.config.training, "memory_optimization"):
                if batch_idx % self.config.training.memory_optimization.clear_cache_interval == 0:
                    torch.cuda.empty_cache()
                    
    @torch.no_grad()
    def validate(self):
        """Validate model"""
        self.acoustic_model.eval()
        if self.xphonebert is not None:
            self.xphonebert.eval()
        if self.f0_bert is not None:
            self.f0_bert.eval()
            
        self.val_metrics.reset()
        
        for batch in self.val_loader:
            metrics = self.val_step(batch)
            self.val_metrics.update(metrics)
            
        avg_metrics = self.val_metrics.compute_average()
        
        if self.rank == 0:
            self.log_metrics(avg_metrics, "val")
            logger.info(f"Validation - Loss: {avg_metrics['loss']:.4f}")
            
        return avg_metrics
        
    def log_metrics(self, metrics: Dict[str, float], prefix: str):
        """Log metrics to tensorboard"""
        for name, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{name}", value, self.global_step)
            
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "acoustic_model_state_dict": self.acoustic_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        
        if self.xphonebert is not None:
            checkpoint["xphonebert_state_dict"] = self.xphonebert.state_dict()
        if self.f0_bert is not None:
            checkpoint["f0_bert_state_dict"] = self.f0_bert.state_dict()
        if self.vocoder is not None:
            checkpoint["vocoder_state_dict"] = self.vocoder.state_dict()
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.disc_optimizer is not None:
            checkpoint["disc_optimizer_state_dict"] = self.disc_optimizer.state_dict()
            
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = Path(path).parent / "best_model.pt"
            torch.save(checkpoint, best_path)
            
        logger.info(f"Saved checkpoint to {path}")
        
    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.acoustic_model.load_state_dict(checkpoint["acoustic_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "xphonebert_state_dict" in checkpoint and self.xphonebert is not None:
            self.xphonebert.load_state_dict(checkpoint["xphonebert_state_dict"])
        if "f0_bert_state_dict" in checkpoint and self.f0_bert is not None:
            self.f0_bert.load_state_dict(checkpoint["f0_bert_state_dict"])
        if "vocoder_state_dict" in checkpoint and self.vocoder is not None:
            self.vocoder.load_state_dict(checkpoint["vocoder_state_dict"])
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "disc_optimizer_state_dict" in checkpoint and self.disc_optimizer is not None:
            self.disc_optimizer.load_state_dict(checkpoint["disc_optimizer_state_dict"])
            
        self.epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        best_val_loss = float("inf")
        
        for epoch in range(self.epoch, self.config.training.num_epochs):
            self.epoch = epoch
            
            # Update learning rate schedule
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Unfreeze weights if using progressive unfreezing
            if hasattr(self, "ft_config"):
                freeze_model_weights(self.acoustic_model, self.ft_config, epoch)
                
            # Train
            self.train_epoch()
            
            # Validate
            if epoch % self.config.training.get("val_interval", 1) == 0:
                val_metrics = self.validate()
                
                # Save checkpoint
                if self.rank == 0:
                    is_best = val_metrics["loss"] < best_val_loss
                    if is_best:
                        best_val_loss = val_metrics["loss"]
                        
                    if epoch % self.config.training.save_interval == 0:
                        checkpoint_path = (
                            Path(self.config.training.checkpoint_dir) / 
                            f"checkpoint_epoch_{epoch}.pt"
                        )
                        self.save_checkpoint(str(checkpoint_path), is_best)
                        
        logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train Tsukuyomi TTS")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Override data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use distributed training",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="Local rank for distributed training",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.output_dir:
        config.training.checkpoint_dir = args.output_dir
        config.paths.checkpoints = args.output_dir
    if args.resume:
        config.training.resume_from = args.resume
        
    # Setup distributed training
    if args.distributed:
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0
        
    # Create output directories
    if rank == 0:
        Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.paths.tensorboard).mkdir(parents=True, exist_ok=True)
        
    # Create trainer
    trainer = TTSTrainer(config, rank, world_size)
    
    # Train
    trainer.train()
    
    # Cleanup
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()