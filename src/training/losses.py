"""Loss functions for TTS training"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class MultiTaskLoss(nn.Module):
    """Multi-task loss for TTS training"""
    
    def __init__(
        self,
        loss_weights: Optional[Dict[str, float]] = None,
        use_mel_loss: bool = True,
        use_kl_loss: bool = True,
        use_duration_loss: bool = True,
        use_pitch_loss: bool = True,
        use_gan_loss: bool = True,
        use_feature_matching_loss: bool = True,
    ):
        super().__init__()
        
        # Default loss weights
        self.loss_weights = loss_weights or {
            "mel": 45.0,
            "kl": 1.0,
            "duration": 1.0,
            "pitch": 1.0,
            "gen": 1.0,
            "fm": 2.0,
        }
        
        self.use_mel_loss = use_mel_loss
        self.use_kl_loss = use_kl_loss
        self.use_duration_loss = use_duration_loss
        self.use_pitch_loss = use_pitch_loss
        self.use_gan_loss = use_gan_loss
        self.use_feature_matching_loss = use_feature_matching_loss
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        discriminator_outputs: Optional[Dict[str, List[torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate multi-task loss
        
        Args:
            predictions: Model predictions
            targets: Target values
            discriminator_outputs: Outputs from discriminators (for GAN loss)
            
        Returns:
            Dictionary of individual losses and total loss
        """
        losses = {}
        
        # Mel-spectrogram loss
        if self.use_mel_loss and "mel" in predictions and "mel" in targets:
            losses["mel"] = self.mel_loss(predictions["mel"], targets["mel"])
            
        # KL divergence loss (for VITS)
        if self.use_kl_loss and "kl" in predictions:
            losses["kl"] = predictions["kl"]
            
        # Duration loss
        if self.use_duration_loss and "duration" in predictions and "duration" in targets:
            losses["duration"] = self.duration_loss(
                predictions["duration"],
                targets["duration"],
                targets.get("duration_mask")
            )
            
        # Pitch loss
        if self.use_pitch_loss and "pitch" in predictions and "pitch" in targets:
            losses["pitch"] = self.pitch_loss(
                predictions["pitch"],
                targets["pitch"],
                targets.get("pitch_mask")
            )
            
        # GAN losses
        if self.use_gan_loss and discriminator_outputs is not None:
            gen_loss, fm_loss = self.gan_losses(discriminator_outputs)
            if gen_loss is not None:
                losses["gen"] = gen_loss
            if fm_loss is not None and self.use_feature_matching_loss:
                losses["fm"] = fm_loss
                
        # Calculate weighted total loss
        total_loss = torch.tensor(0.0, device=predictions.get("mel", torch.tensor(0)).device)
        for name, loss in losses.items():
            weight = self.loss_weights.get(name, 1.0)
            total_loss = total_loss + weight * loss
            
        losses["total"] = total_loss
        
        return losses
        
    def mel_loss(self, pred_mel: torch.Tensor, target_mel: torch.Tensor) -> torch.Tensor:
        """L1 loss for mel-spectrogram"""
        return F.l1_loss(pred_mel, target_mel)
        
    def duration_loss(
        self,
        pred_duration: torch.Tensor,
        target_duration: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """MSE loss for duration prediction"""
        if mask is not None:
            pred_duration = pred_duration * mask
            target_duration = target_duration * mask
            return F.mse_loss(pred_duration, target_duration, reduction="sum") / mask.sum()
        else:
            return F.mse_loss(pred_duration, target_duration)
            
    def pitch_loss(
        self,
        pred_pitch: torch.Tensor,
        target_pitch: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """MSE loss for pitch prediction"""
        if mask is not None:
            # Only calculate loss for voiced regions
            voiced_mask = (target_pitch > 0).float() * mask
            if voiced_mask.sum() > 0:
                pred_pitch = pred_pitch * voiced_mask
                target_pitch = target_pitch * voiced_mask
                return F.mse_loss(pred_pitch, target_pitch, reduction="sum") / voiced_mask.sum()
            else:
                return torch.tensor(0.0, device=pred_pitch.device)
        else:
            return F.mse_loss(pred_pitch, target_pitch)
            
    def gan_losses(
        self,
        discriminator_outputs: Dict[str, List[torch.Tensor]]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Calculate generator and feature matching losses"""
        gen_loss = None
        fm_loss = None
        
        # Generator loss (hinge loss)
        if "fake" in discriminator_outputs:
            gen_losses = []
            for d_fake in discriminator_outputs["fake"]:
                gen_losses.append(torch.mean((1 - d_fake) ** 2))
            gen_loss = sum(gen_losses) / len(gen_losses)
            
        # Feature matching loss
        if "fmap_real" in discriminator_outputs and "fmap_fake" in discriminator_outputs:
            fm_losses = []
            for fmap_r, fmap_f in zip(
                discriminator_outputs["fmap_real"],
                discriminator_outputs["fmap_fake"]
            ):
                for fr, ff in zip(fmap_r, fmap_f):
                    fm_losses.append(F.l1_loss(ff, fr.detach()))
            fm_loss = sum(fm_losses) / len(fm_losses)
            
        return gen_loss, fm_loss


class DiscriminatorLoss(nn.Module):
    """Discriminator loss for GAN training"""
    
    def __init__(self, loss_type: str = "hinge"):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(
        self,
        disc_real_outputs: List[torch.Tensor],
        disc_fake_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Calculate discriminator loss
        
        Args:
            disc_real_outputs: Discriminator outputs for real samples
            disc_fake_outputs: Discriminator outputs for fake samples
            
        Returns:
            Discriminator loss
        """
        if self.loss_type == "hinge":
            return self.hinge_loss(disc_real_outputs, disc_fake_outputs)
        elif self.loss_type == "lsgan":
            return self.lsgan_loss(disc_real_outputs, disc_fake_outputs)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
    def hinge_loss(
        self,
        disc_real_outputs: List[torch.Tensor],
        disc_fake_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """Hinge loss for discriminator"""
        real_losses = []
        fake_losses = []
        
        for dr in disc_real_outputs:
            real_losses.append(torch.mean(torch.clamp(1 - dr, min=0)))
            
        for df in disc_fake_outputs:
            fake_losses.append(torch.mean(torch.clamp(1 + df, min=0)))
            
        real_loss = sum(real_losses) / len(real_losses)
        fake_loss = sum(fake_losses) / len(fake_losses)
        
        return real_loss + fake_loss
        
    def lsgan_loss(
        self,
        disc_real_outputs: List[torch.Tensor],
        disc_fake_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """Least squares GAN loss for discriminator"""
        real_losses = []
        fake_losses = []
        
        for dr in disc_real_outputs:
            real_losses.append(torch.mean((dr - 1) ** 2))
            
        for df in disc_fake_outputs:
            fake_losses.append(torch.mean(df ** 2))
            
        real_loss = sum(real_losses) / len(real_losses)
        fake_loss = sum(fake_losses) / len(fake_losses)
        
        return real_loss + fake_loss


class EmotionLoss(nn.Module):
    """Loss for emotion control"""
    
    def __init__(self, num_emotions: int = 10, temperature: float = 0.07):
        super().__init__()
        self.num_emotions = num_emotions
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(
        self,
        emotion_logits: torch.Tensor,
        emotion_labels: torch.Tensor,
        emotion_embeddings: Optional[torch.Tensor] = None,
        vad_predictions: Optional[torch.Tensor] = None,
        vad_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate emotion-related losses
        
        Args:
            emotion_logits: Predicted emotion logits [B, num_emotions]
            emotion_labels: Target emotion labels [B]
            emotion_embeddings: Emotion embeddings for contrastive loss [B, D]
            vad_predictions: Predicted VAD values [B, 3]
            vad_targets: Target VAD values [B, 3]
            
        Returns:
            Dictionary of emotion losses
        """
        losses = {}
        
        # Classification loss
        losses["emotion_ce"] = self.ce_loss(emotion_logits, emotion_labels)
        
        # Contrastive loss
        if emotion_embeddings is not None:
            losses["emotion_contrastive"] = self.contrastive_loss(
                emotion_embeddings,
                emotion_labels
            )
            
        # VAD regression loss
        if vad_predictions is not None and vad_targets is not None:
            losses["vad"] = F.mse_loss(vad_predictions, vad_targets)
            
        return losses
        
    def contrastive_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """InfoNCE contrastive loss for emotion embeddings"""
        batch_size = embeddings.size(0)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive pairs
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal
        mask.fill_diagonal_(0)
        
        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        exp_sim = exp_sim * (1 - torch.eye(batch_size, device=exp_sim.device))
        
        positive_sim = exp_sim * mask
        negative_sim = exp_sim * (1 - mask)
        
        positive_sim = positive_sim.sum(dim=1)
        negative_sim = negative_sim.sum(dim=1)
        
        loss = -torch.log(positive_sim / (positive_sim + negative_sim + 1e-8))
        
        return loss.mean()


class StyleTransferLoss(nn.Module):
    """Loss for style transfer"""
    
    def __init__(self, style_dim: int = 256):
        super().__init__()
        self.style_dim = style_dim
        
    def forward(
        self,
        pred_style: torch.Tensor,
        target_style: torch.Tensor,
        pred_audio: Optional[torch.Tensor] = None,
        target_audio: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate style transfer losses
        
        Args:
            pred_style: Predicted style embedding [B, style_dim]
            target_style: Target style embedding [B, style_dim]
            pred_audio: Predicted audio for perceptual loss
            target_audio: Target audio for perceptual loss
            
        Returns:
            Dictionary of style losses
        """
        losses = {}
        
        # Style embedding loss
        losses["style_l2"] = F.mse_loss(pred_style, target_style)
        
        # Cosine similarity loss
        pred_norm = F.normalize(pred_style, dim=1)
        target_norm = F.normalize(target_style, dim=1)
        cosine_sim = (pred_norm * target_norm).sum(dim=1)
        losses["style_cosine"] = 1 - cosine_sim.mean()
        
        # Perceptual loss (optional)
        if pred_audio is not None and target_audio is not None:
            losses["perceptual"] = self.perceptual_loss(pred_audio, target_audio)
            
        return losses
        
    def perceptual_loss(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor
    ) -> torch.Tensor:
        """Simple perceptual loss based on spectral features"""
        # Compute spectrograms
        pred_spec = torch.stft(
            pred_audio.squeeze(1),
            n_fft=1024,
            hop_length=256,
            return_complex=True
        ).abs()
        
        target_spec = torch.stft(
            target_audio.squeeze(1),
            n_fft=1024,
            hop_length=256,
            return_complex=True
        ).abs()
        
        # Multi-scale spectral loss
        loss = F.l1_loss(pred_spec, target_spec)
        
        # Add log-scale loss
        loss += F.l1_loss(
            torch.log(pred_spec + 1e-7),
            torch.log(target_spec + 1e-7)
        )
        
        return loss


def get_loss_function(config: Dict) -> nn.Module:
    """Factory function to create loss function based on config"""
    loss_type = config.get("loss_type", "multi_task")
    
    if loss_type == "multi_task":
        return MultiTaskLoss(
            loss_weights=config.get("loss_weights"),
            use_mel_loss=config.get("use_mel_loss", True),
            use_kl_loss=config.get("use_kl_loss", True),
            use_duration_loss=config.get("use_duration_loss", True),
            use_pitch_loss=config.get("use_pitch_loss", True),
            use_gan_loss=config.get("use_gan_loss", True),
            use_feature_matching_loss=config.get("use_feature_matching_loss", True),
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")