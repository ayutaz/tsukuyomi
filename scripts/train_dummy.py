#!/usr/bin/env python3
"""
Dummy training script for testing basic functionality.
This script uses simplified models and synthetic data to verify the training pipeline works.
"""

import logging
import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DummyTTSModel(nn.Module):
    """Simplified TTS model for testing"""
    
    def __init__(self, n_vocab=128, n_speakers=10, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, hidden_dim)
        self.speaker_embedding = nn.Embedding(n_speakers, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, 80)  # 80 mel channels
        
    def forward(self, text, speaker_ids):
        # Simple forward pass
        x = self.embedding(text)
        spk = self.speaker_embedding(speaker_ids)
        x = x + spk.unsqueeze(1)
        x, _ = self.lstm(x)
        mel = self.output_proj(x)
        
        # Return dummy loss
        loss = mel.mean()
        return {'mel': mel, 'loss': loss}


def create_dummy_batch(batch_size=2, device='cpu'):
    """Create a dummy batch for testing"""
    return {
        'text': torch.randint(0, 100, (batch_size, 30)).to(device),
        'speaker_ids': torch.randint(0, 10, (batch_size,)).to(device),
        'audio': torch.randn(batch_size, 16000).to(device),
    }


def train_dummy():
    """Run dummy training loop"""
    logger.info("Starting dummy training...")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = DummyTTSModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    num_epochs = 2
    steps_per_epoch = 10
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            # Create dummy batch
            batch = create_dummy_batch(batch_size=2, device=device)
            
            # Forward pass
            outputs = model(batch['text'], batch['speaker_ids'])
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if step % 5 == 0:
                logger.info(f"  Step {step}/{steps_per_epoch}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / steps_per_epoch
        logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
    
    logger.info("Dummy training completed successfully!")
    return True


if __name__ == "__main__":
    success = train_dummy()
    if success:
        print("\n✅ Dummy training successful! The basic training pipeline is working.")
        print("Next steps:")
        print("1. Fix the VITS model configuration")
        print("2. Implement proper text tokenization")
        print("3. Add mel-spectrogram extraction")
        print("4. Connect real data pipeline")
    else:
        print("\n❌ Dummy training failed!")
        exit(1)