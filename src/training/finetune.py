"""Fine-tuning utilities for Tsukuyomi TTS

Supports fine-tuning from pre-trained models with various strategies
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class FineTuningConfig:
    """Configuration for fine-tuning"""
    
    def __init__(
        self,
        # Model selection
        base_model_path: str,
        output_dir: str,
        
        # Fine-tuning strategy
        strategy: str = "full",  # "full", "lora", "adapter", "freeze_encoder"
        
        # LoRA settings
        lora_rank: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.1,
        lora_target_modules: List[str] = None,
        
        # Adapter settings
        adapter_size: int = 256,
        adapter_layers: List[int] = None,
        
        # Freezing settings
        freeze_modules: List[str] = None,
        unfreeze_after_epoch: int = -1,
        
        # Training settings
        learning_rate: float = 1e-4,
        min_learning_rate: float = 1e-6,
        warmup_steps: int = 1000,
        num_epochs: int = 50,
        batch_size: int = 16,
        gradient_accumulation_steps: int = 1,
        
        # Data settings
        new_speakers: Optional[List[int]] = None,
        speaker_embedding_dim: int = 256,
        
        # Regularization
        weight_decay: float = 0.01,
        gradient_clip: float = 1.0,
        label_smoothing: float = 0.1,
    ):
        self.base_model_path = base_model_path
        self.output_dir = output_dir
        self.strategy = strategy
        
        # LoRA
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or [
            "query", "key", "value", "dense"
        ]
        
        # Adapter
        self.adapter_size = adapter_size
        self.adapter_layers = adapter_layers
        
        # Freezing
        self.freeze_modules = freeze_modules or []
        self.unfreeze_after_epoch = unfreeze_after_epoch
        
        # Training
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_steps = warmup_steps
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Data
        self.new_speakers = new_speakers
        self.speaker_embedding_dim = speaker_embedding_dim
        
        # Regularization
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.label_smoothing = label_smoothing


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        alpha: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA weights
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=1)
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation"""
        lora_output = self.lora_dropout(x)
        lora_output = self.lora_A(lora_output)
        lora_output = self.lora_B(lora_output)
        
        return base_output + lora_output * self.scaling


class AdapterLayer(nn.Module):
    """Adapter layer for parameter-efficient fine-tuning"""
    
    def __init__(
        self,
        hidden_size: int,
        adapter_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Down-projection
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        
        # Non-linearity
        self.activation = nn.ReLU()
        
        # Up-projection
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Initialize
        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adapter"""
        residual = x
        
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        
        return residual + x


def apply_lora_to_model(
    model: nn.Module,
    config: FineTuningConfig
) -> nn.Module:
    """Apply LoRA to target modules in the model"""
    
    for name, module in model.named_modules():
        # Check if this module should have LoRA
        if any(target in name for target in config.lora_target_modules):
            if isinstance(module, nn.Linear):
                # Get dimensions
                in_features = module.in_features
                out_features = module.out_features
                
                # Create LoRA layer
                lora = LoRALayer(
                    in_features,
                    out_features,
                    rank=config.lora_rank,
                    alpha=config.lora_alpha,
                    dropout=config.lora_dropout,
                )
                
                # Store original forward
                original_forward = module.forward
                
                # Create new forward with LoRA
                def forward_with_lora(self, x):
                    base_output = original_forward(x)
                    return lora(x, base_output)
                
                # Replace forward method
                module.forward = forward_with_lora.__get__(module, type(module))
                
                # Store LoRA layer
                module.lora = lora
                
                logger.info(f"Applied LoRA to {name}")
                
    return model


def apply_adapters_to_model(
    model: nn.Module,
    config: FineTuningConfig
) -> nn.Module:
    """Apply adapters to specified layers"""
    
    adapter_count = 0
    
    for name, module in model.named_modules():
        # Check if this layer should have an adapter
        if config.adapter_layers is None or any(
            f"layer.{i}" in name for i in config.adapter_layers
        ):
            if "layer_norm" in name or "ln" in name:
                continue
                
            if hasattr(module, "hidden_size"):
                # Add adapter after this module
                adapter = AdapterLayer(
                    hidden_size=module.hidden_size,
                    adapter_size=config.adapter_size,
                    dropout=config.lora_dropout,
                )
                
                # Store adapter
                module.adapter = adapter
                
                # Modify forward to include adapter
                original_forward = module.forward
                
                def forward_with_adapter(self, *args, **kwargs):
                    output = original_forward(*args, **kwargs)
                    if isinstance(output, tuple):
                        # Handle multi-output modules
                        hidden_states = output[0]
                        hidden_states = self.adapter(hidden_states)
                        return (hidden_states,) + output[1:]
                    else:
                        return self.adapter(output)
                
                module.forward = forward_with_adapter.__get__(module, type(module))
                
                adapter_count += 1
                logger.info(f"Applied adapter to {name}")
                
    logger.info(f"Total adapters applied: {adapter_count}")
    return model


def freeze_model_weights(
    model: nn.Module,
    config: FineTuningConfig,
    epoch: int = 0
) -> nn.Module:
    """Freeze/unfreeze model weights based on strategy"""
    
    # Check if we should unfreeze
    if config.unfreeze_after_epoch > 0 and epoch >= config.unfreeze_after_epoch:
        logger.info("Unfreezing all weights")
        for param in model.parameters():
            param.requires_grad = True
        return model
        
    if config.strategy == "freeze_encoder":
        # Freeze encoder layers
        for name, param in model.named_parameters():
            if any(module in name for module in ["encoder", "embedding", "bert"]):
                param.requires_grad = False
                logger.debug(f"Froze {name}")
            else:
                param.requires_grad = True
                
    elif config.strategy == "lora":
        # Freeze everything except LoRA parameters
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
                logger.debug(f"Unfroze {name}")
            else:
                param.requires_grad = False
                
    elif config.strategy == "adapter":
        # Freeze everything except adapters
        for name, param in model.named_parameters():
            if "adapter" in name:
                param.requires_grad = True
                logger.debug(f"Unfroze {name}")
            else:
                param.requires_grad = False
                
    elif config.freeze_modules:
        # Custom freezing
        for name, param in model.named_parameters():
            if any(module in name for module in config.freeze_modules):
                param.requires_grad = False
                logger.debug(f"Froze {name}")
            else:
                param.requires_grad = True
                
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    
    return model


def expand_speaker_embeddings(
    model: nn.Module,
    config: FineTuningConfig,
    base_n_speakers: int
) -> nn.Module:
    """Expand speaker embeddings for new speakers"""
    
    if config.new_speakers is None:
        return model
        
    n_new_speakers = len(config.new_speakers)
    total_speakers = base_n_speakers + n_new_speakers
    
    # Find speaker embedding layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and "speaker" in name:
            old_embedding = module
            old_weight = old_embedding.weight.data
            
            # Create new embedding layer
            new_embedding = nn.Embedding(
                total_speakers,
                old_embedding.embedding_dim,
                padding_idx=old_embedding.padding_idx
            )
            
            # Copy old weights
            new_embedding.weight.data[:base_n_speakers] = old_weight
            
            # Initialize new speaker embeddings
            nn.init.normal_(
                new_embedding.weight.data[base_n_speakers:],
                mean=0.0,
                std=0.02
            )
            
            # Replace in model
            parent_name = name.rsplit('.', 1)[0]
            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            setattr(parent, name.split('.')[-1], new_embedding)
            
            logger.info(
                f"Expanded speaker embeddings from {base_n_speakers} to {total_speakers}"
            )
            
    return model


def load_checkpoint_for_finetuning(
    model: nn.Module,
    checkpoint_path: str,
    config: FineTuningConfig,
    strict: bool = False
) -> Tuple[nn.Module, Dict]:
    """Load checkpoint and prepare model for fine-tuning"""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Get state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
        
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=strict
    )
    
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")
        
    # Apply fine-tuning strategy
    if config.strategy == "lora":
        model = apply_lora_to_model(model, config)
    elif config.strategy == "adapter":
        model = apply_adapters_to_model(model, config)
        
    # Expand speaker embeddings if needed
    if "n_speakers" in checkpoint:
        model = expand_speaker_embeddings(
            model,
            config,
            checkpoint["n_speakers"]
        )
        
    # Freeze weights
    model = freeze_model_weights(model, config)
    
    return model, checkpoint


def save_finetuned_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: FineTuningConfig,
    metrics: Dict,
    output_path: str
):
    """Save fine-tuned model checkpoint"""
    
    # Prepare checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.__dict__,
        "metrics": metrics,
        "finetuning_strategy": config.strategy,
    }
    
    # Save LoRA weights separately if using LoRA
    if config.strategy == "lora":
        lora_weights = {}
        for name, module in model.named_modules():
            if hasattr(module, "lora"):
                lora_weights[f"{name}.lora_A"] = module.lora.lora_A.weight
                lora_weights[f"{name}.lora_B"] = module.lora.lora_B.weight
        checkpoint["lora_weights"] = lora_weights
        
    # Save adapter weights separately if using adapters
    elif config.strategy == "adapter":
        adapter_weights = {}
        for name, module in model.named_modules():
            if hasattr(module, "adapter"):
                adapter_weights[f"{name}.adapter"] = module.adapter.state_dict()
        checkpoint["adapter_weights"] = adapter_weights
        
    # Save checkpoint
    torch.save(checkpoint, output_path)
    logger.info(f"Saved fine-tuned model to {output_path}")


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into base model for inference"""
    
    for name, module in model.named_modules():
        if hasattr(module, "lora") and isinstance(module, nn.Linear):
            # Get LoRA weights
            lora_A = module.lora.lora_A.weight
            lora_B = module.lora.lora_B.weight
            scaling = module.lora.scaling
            
            # Compute LoRA weight update
            lora_weight = (lora_B @ lora_A) * scaling
            
            # Merge into base weights
            module.weight.data += lora_weight
            
            # Remove LoRA
            delattr(module, "lora")
            
            logger.info(f"Merged LoRA weights in {name}")
            
    return model