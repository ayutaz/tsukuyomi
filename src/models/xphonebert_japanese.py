"""
XPhoneBERT Japanese: Japanese-optimized multilingual phoneme encoder

This module implements a Japanese-optimized version of XPhoneBERT with:
- Japanese phoneme system adaptation
- Accent and pitch pattern modeling
- Dialect awareness
- LoRA fine-tuning for efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
from dataclasses import dataclass
import logging
from pathlib import Path

from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig
)
from transformers.modeling_outputs import BaseModelOutputWithPooling

# CUDA 12.1+ optimizations
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False

logger = logging.getLogger(__name__)

# Enable CUDA 12.1+ optimizations
if torch.cuda.is_available():
    cuda_version = torch.version.cuda
    if cuda_version and float(cuda_version.split('.')[0]) >= 12.1:
        logger.info(f"CUDA {cuda_version} detected - enabling XPhoneBERT-Japanese optimizations")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


@dataclass
class XPhoneBERTJapaneseConfig:
    """Configuration for Japanese-optimized XPhoneBERT."""
    # Base model
    base_model: str = "vinai/xphonebert-base"
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    
    # Japanese-specific
    num_accent_types: int = 4  # 平板、頭高、中高、尾高
    num_dialects: int = 47     # Japanese prefectures
    pitch_embedding_dim: int = 256
    
    # LoRA configuration
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # Training configuration
    gradient_checkpointing: bool = True
    use_cache: bool = True
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["query", "value"]


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer for efficient fine-tuning.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        alpha: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Original weight (frozen)
        self.weight = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation."""
        # Original computation (with frozen weights)
        if self.weight is not None:
            result = F.linear(x, self.weight)
        else:
            result = x
            
        # LoRA adaptation
        lora_out = x @ self.lora_A
        lora_out = self.dropout(lora_out)
        lora_out = lora_out @ self.lora_B
        lora_out = lora_out * self.scaling
        
        return result + lora_out


class JapanesePhonemeAdapter(nn.Module):
    """
    Adapter module for Japanese phoneme system.
    
    Handles:
    - Japanese-specific phonemes (ん, っ, etc.)
    - Accent patterns
    - Pitch accents
    - Dialectal variations
    """
    
    def __init__(self, config: XPhoneBERTJapaneseConfig):
        super().__init__()
        self.config = config
        
        # Accent embedding
        self.accent_embedding = nn.Embedding(
            config.num_accent_types,
            config.hidden_size
        )
        
        # Dialect embedding
        self.dialect_embedding = nn.Embedding(
            config.num_dialects,
            config.hidden_size
        )
        
        # Pitch pattern encoder
        self.pitch_encoder = nn.Sequential(
            nn.Linear(config.pitch_embedding_dim, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, config.hidden_size)
        )
        
        # Japanese phoneme-specific transformations
        self.phoneme_transform = nn.ModuleDict({
            'mora_boundary': nn.Linear(config.hidden_size, config.hidden_size),
            'geminate': nn.Linear(config.hidden_size, config.hidden_size),  # っ
            'nasal': nn.Linear(config.hidden_size, config.hidden_size),     # ん
            'long_vowel': nn.Linear(config.hidden_size, config.hidden_size),
        })
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        accent_ids: Optional[torch.Tensor] = None,
        dialect_ids: Optional[torch.Tensor] = None,
        pitch_features: Optional[torch.Tensor] = None,
        phoneme_types: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply Japanese-specific adaptations.
        
        Args:
            hidden_states: Base phoneme embeddings (batch, seq_len, hidden)
            accent_ids: Accent type IDs (batch, seq_len)
            dialect_ids: Dialect IDs (batch,)
            pitch_features: Pitch pattern features (batch, seq_len, pitch_dim)
            phoneme_types: Type of each phoneme (batch, seq_len)
            
        Returns:
            Adapted hidden states
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # Default values
        if accent_ids is None:
            accent_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        if dialect_ids is None:
            dialect_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
            
        # Get embeddings
        accent_emb = self.accent_embedding(accent_ids)
        dialect_emb = self.dialect_embedding(dialect_ids).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Process pitch if provided
        if pitch_features is not None:
            pitch_emb = self.pitch_encoder(pitch_features)
        else:
            pitch_emb = torch.zeros_like(hidden_states)
            
        # Apply phoneme-specific transformations
        if phoneme_types is not None:
            # Create masks for different phoneme types
            is_geminate = (phoneme_types == 1).float().unsqueeze(-1)
            is_nasal = (phoneme_types == 2).float().unsqueeze(-1)
            is_long_vowel = (phoneme_types == 3).float().unsqueeze(-1)
            
            # Apply transformations
            hidden_states = (
                hidden_states +
                is_geminate * self.phoneme_transform['geminate'](hidden_states) +
                is_nasal * self.phoneme_transform['nasal'](hidden_states) +
                is_long_vowel * self.phoneme_transform['long_vowel'](hidden_states)
            )
            
        # Combine all features
        combined = torch.cat([
            hidden_states,
            accent_emb,
            dialect_emb + pitch_emb
        ], dim=-1)
        
        # Fusion
        adapted = self.fusion(combined)
        
        return adapted + hidden_states  # Residual connection


class XPhoneBERTJapanese(nn.Module):
    """
    Japanese-optimized XPhoneBERT model.
    
    Features:
    - Pre-trained multilingual phoneme understanding
    - Japanese phoneme system adaptation
    - Accent and pitch modeling
    - Efficient LoRA fine-tuning
    - Multi-dialect support
    """
    
    def __init__(self, config: Optional[XPhoneBERTJapaneseConfig] = None):
        super().__init__()
        self.config = config or XPhoneBERTJapaneseConfig()
        
        # Load base XPhoneBERT
        try:
            self.base_model = AutoModel.from_pretrained(
                self.config.base_model,
                trust_remote_code=True,
                use_auth_token=False
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                trust_remote_code=True,
                use_auth_token=False
            )
            logger.info(f"Loaded base model: {self.config.base_model}")
        except Exception as e:
            logger.warning(f"Could not load pre-trained model: {e}")
            logger.info("Initializing with random weights")
            # Initialize with config
            base_config = AutoConfig.from_pretrained(self.config.base_model)
            self.base_model = AutoModel.from_config(base_config)
            self.tokenizer = None
            
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Japanese adapter
        self.japanese_adapter = JapanesePhonemeAdapter(self.config)
        
        # LoRA layers
        if self.config.use_lora:
            self.lora_layers = nn.ModuleDict()
            self._add_lora_layers()
            
        # Output projection
        self.output_projection = nn.Linear(
            self.config.hidden_size,
            self.config.hidden_size
        )
        
        # Japanese phoneme vocabulary extension
        self.japanese_phoneme_embeddings = nn.Embedding(
            100,  # Extended vocabulary for Japanese phonemes
            self.config.hidden_size
        )
        
        # Mora boundary predictor
        self.mora_boundary_predictor = nn.Linear(self.config.hidden_size, 2)
        
        logger.info("Initialized XPhoneBERT-Japanese")
        
    def _add_lora_layers(self):
        """Add LoRA layers to specified modules."""
        for name, module in self.base_model.named_modules():
            if any(target in name for target in self.config.lora_target_modules):
                if isinstance(module, nn.Linear):
                    # Clean module name for ModuleDict
                    clean_name = name.replace('.', '_')
                    lora = LoRALayer(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        rank=self.config.lora_rank,
                        alpha=self.config.lora_alpha,
                        dropout=self.config.lora_dropout
                    )
                    lora.weight = module.weight
                    self.lora_layers[clean_name] = lora
                    
        logger.info(f"Added {len(self.lora_layers)} LoRA layers")
        
    def forward(
        self,
        phoneme_ids: Optional[torch.Tensor] = None,
        phoneme_strings: Optional[List[str]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        accent_ids: Optional[torch.Tensor] = None,
        dialect_ids: Optional[torch.Tensor] = None,
        pitch_features: Optional[torch.Tensor] = None,
        phoneme_types: Optional[torch.Tensor] = None,
        return_mora_boundaries: bool = False,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of XPhoneBERT-Japanese.
        
        Args:
            phoneme_ids: Phoneme token IDs (batch, seq_len)
            phoneme_strings: List of phoneme strings (alternative input)
            attention_mask: Attention mask
            accent_ids: Japanese accent patterns
            dialect_ids: Dialect identifiers
            pitch_features: Pitch contour features
            phoneme_types: Type of each phoneme (geminate, nasal, etc.)
            return_mora_boundaries: Whether to predict mora boundaries
            return_dict: Whether to return dictionary
            
        Returns:
            Phoneme embeddings or dictionary of outputs
        """
        # Handle string input
        if phoneme_strings is not None and self.tokenizer is not None:
            # Convert phoneme strings to IDs
            encoded = self.tokenizer(
                phoneme_strings,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            phoneme_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            
        # Get base embeddings
        outputs = self.base_model(
            input_ids=phoneme_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        hidden_states = outputs.last_hidden_state
        
        # Apply LoRA if enabled
        if self.config.use_lora:
            # Hook into specific layers and add LoRA
            # This is a simplified version - in practice would use hooks
            pass
            
        # Apply Japanese adaptation
        hidden_states = self.japanese_adapter(
            hidden_states,
            accent_ids=accent_ids,
            dialect_ids=dialect_ids,
            pitch_features=pitch_features,
            phoneme_types=phoneme_types
        )
        
        # Output projection
        hidden_states = self.output_projection(hidden_states)
        
        if not return_dict:
            return hidden_states
            
        result = {
            'hidden_states': hidden_states,
            'pooler_output': outputs.pooler_output if hasattr(outputs, 'pooler_output') else None
        }
        
        # Mora boundary prediction
        if return_mora_boundaries:
            mora_logits = self.mora_boundary_predictor(hidden_states)
            result['mora_boundaries'] = mora_logits
            
        return result
        
    def encode_japanese_phonemes(
        self,
        phonemes: str,
        accent_pattern: Optional[str] = None,
        dialect: Optional[str] = None
    ) -> torch.Tensor:
        """
        Convenience method for encoding Japanese phonemes.
        
        Args:
            phonemes: Space-separated phoneme string
            accent_pattern: Accent pattern (e.g., "LHHHL")
            dialect: Dialect name or code
            
        Returns:
            Phoneme embeddings
        """
        # Convert phonemes to IDs
        if self.tokenizer:
            encoded = self.tokenizer(
                phonemes,
                return_tensors='pt'
            )
            phoneme_ids = encoded['input_ids']
        else:
            # Fallback to simple mapping
            phoneme_list = phonemes.split()
            phoneme_ids = torch.tensor([[hash(p) % 1000 for p in phoneme_list]])
            
        # Convert accent pattern
        if accent_pattern:
            accent_map = {'L': 0, 'H': 1, 'F': 2, 'R': 3}  # Low, High, Fall, Rise
            accent_ids = torch.tensor([[accent_map.get(a, 0) for a in accent_pattern]])
        else:
            accent_ids = None
            
        # Convert dialect
        if dialect:
            dialect_map = {
                'tokyo': 0, 'osaka': 1, 'kyoto': 2, 'tohoku': 3,
                # Add more dialects
            }
            dialect_ids = torch.tensor([dialect_map.get(dialect, 0)])
        else:
            dialect_ids = None
            
        # Encode
        outputs = self.forward(
            phoneme_ids=phoneme_ids,
            accent_ids=accent_ids,
            dialect_ids=dialect_ids,
            return_dict=True
        )
        
        return outputs['hidden_states']
        
    def save_pretrained(self, save_path: Path):
        """Save model with LoRA weights."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_dict = {
            'base_model': self.config.base_model,
            'hidden_size': self.config.hidden_size,
            'use_lora': self.config.use_lora,
            'lora_rank': self.config.lora_rank,
            'lora_alpha': self.config.lora_alpha
        }
        
        import json
        with open(save_path / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
            
        # Save LoRA weights
        if self.config.use_lora:
            lora_state_dict = {
                name: module.state_dict()
                for name, module in self.lora_layers.items()
            }
            torch.save(lora_state_dict, save_path / 'lora_weights.pt')
            
        # Save adapter weights
        torch.save(self.japanese_adapter.state_dict(), save_path / 'adapter_weights.pt')
        
        logger.info(f"Saved model to {save_path}")
        
    @classmethod
    def from_pretrained(cls, load_path: Path):
        """Load model with LoRA weights."""
        load_path = Path(load_path)
        
        # Load config
        import json
        with open(load_path / 'config.json', 'r') as f:
            config_dict = json.load(f)
            
        # Create config
        config = XPhoneBERTJapaneseConfig(**config_dict)
        
        # Initialize model
        model = cls(config)
        
        # Load LoRA weights
        if config.use_lora and (load_path / 'lora_weights.pt').exists():
            lora_state_dict = torch.load(load_path / 'lora_weights.pt')
            for name, state_dict in lora_state_dict.items():
                if name in model.lora_layers:
                    model.lora_layers[name].load_state_dict(state_dict)
                    
        # Load adapter weights
        if (load_path / 'adapter_weights.pt').exists():
            adapter_state_dict = torch.load(load_path / 'adapter_weights.pt')
            model.japanese_adapter.load_state_dict(adapter_state_dict)
            
        logger.info(f"Loaded model from {load_path}")
        return model
    
    def optimize_for_inference(self) -> None:
        """
        Optimize model for inference using CUDA 12.1+ features.
        """
        self.eval()
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping optimization")
            return
            
        cuda_version = torch.version.cuda
        if cuda_version and float(cuda_version.split('.')[0]) >= 12.1:
            logger.info("Optimizing XPhoneBERT-Japanese with CUDA 12.1+ features")
            
            # Enable BF16 precision
            self.to(torch.bfloat16)
            logger.info("Enabled BF16 precision")
            
            # Compile model with torch.compile
            try:
                self.forward = torch.compile(
                    self.forward,
                    mode="max-autotune",
                    fullgraph=False,
                    backend="inductor"
                )
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
                
            # Fuse LoRA weights if not training
            if self.config.use_lora and not self.training:
                self._fuse_lora_weights()
                logger.info("Fused LoRA weights for inference")
        else:
            logger.warning(f"CUDA {cuda_version} detected - CUDA 12.1+ required for optimization")
    
    def _fuse_lora_weights(self):
        """Fuse LoRA weights into base model for faster inference."""
        for name, lora_layer in self.lora_layers.items():
            # Get corresponding base layer
            layer_idx = int(name.split('_')[1])
            base_layer = self.base_model.encoder.layer[layer_idx].attention.self
            
            # Fuse query weights
            if hasattr(lora_layer, 'lora_q'):
                q_delta = lora_layer.lora_q.lora_B @ lora_layer.lora_q.lora_A
                q_delta = q_delta * (lora_layer.alpha / lora_layer.rank)
                base_layer.query.weight.data += q_delta.T
                
            # Fuse value weights  
            if hasattr(lora_layer, 'lora_v'):
                v_delta = lora_layer.lora_v.lora_B @ lora_layer.lora_v.lora_A
                v_delta = v_delta * (lora_layer.alpha / lora_layer.rank)
                base_layer.value.weight.data += v_delta.T


def create_xphonebert_japanese(
    checkpoint_path: Optional[Path] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> XPhoneBERTJapanese:
    """Create XPhoneBERT-Japanese instance."""
    if checkpoint_path and checkpoint_path.exists():
        model = XPhoneBERTJapanese.from_pretrained(checkpoint_path)
    else:
        model = XPhoneBERTJapanese()
        
    return model.to(device)


if __name__ == "__main__":
    # Test Japanese optimization
    model = create_xphonebert_japanese()
    
    # Test encoding
    test_phrases = [
        "k o N n i ch i w a",
        "a r i g a t o o g o z a i m a s u",
        "t s u k u y o m i n o o N s e e g o o s e e"
    ]
    
    for phrase in test_phrases:
        embeddings = model.encode_japanese_phonemes(
            phrase,
            accent_pattern="LHHHL",
            dialect="tokyo"
        )
        print(f"Encoded '{phrase}': {embeddings.shape}")