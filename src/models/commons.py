"""Common utilities for VITS and other models"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def sequence_mask(lengths: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    """Generate sequence mask"""
    if max_length is None:
        max_length = lengths.max()
        
    x = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)
    return x.unsqueeze(0) < lengths.unsqueeze(1)


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Generate monotonic alignment path from duration"""
    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, dim=1)
    
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype, device=mask.device)
    
    for b_idx in range(b):
        cum_dur = cum_duration[b_idx]
        for t_idx in range(t_x):
            if t_idx == 0:
                start = 0
            else:
                start = int(cum_dur[t_idx-1].item())
            end = int(cum_dur[t_idx].item())
            
            if end > start and start < t_y and end <= t_y:
                path[b_idx, t_idx, start:end] = 1
                
    return path * mask


def rand_slice_segments(
    x: torch.Tensor,
    x_lengths: torch.Tensor,
    segment_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly slice segments from batch"""
    b, d, t = x.size()
    ids_str_max = x_lengths - segment_size + 1
    ids_str_max = ids_str_max.clamp(min=0)
    
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).long()
    ret = torch.zeros(b, d, segment_size, device=x.device)
    
    for i in range(b):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
        
    return ret, ids_str


def convert_pad_shape(pad_shape: list) -> list:
    """Convert padding shape for compatibility"""
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def slice_segments(
    x: torch.Tensor,
    ids_str: torch.Tensor,
    segment_size: int
) -> torch.Tensor:
    """Slice segments from batch with given start indices"""
    b, d, t = x.size()
    ret = torch.zeros(b, d, segment_size, device=x.device)
    
    for i in range(b):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
        
    return ret


def init_weights(m, mean=0.0, std=0.01):
    """Initialize weights"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding for 'same' convolution"""
    return int((kernel_size * dilation - dilation) / 2)


def kl_divergence(
    m_p: torch.Tensor,
    logs_p: torch.Tensor, 
    m_q: torch.Tensor,
    logs_q: torch.Tensor,
    z_mask: torch.Tensor
) -> torch.Tensor:
    """Compute KL divergence between two normal distributions"""
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((m_p - m_q) ** 2) * torch.exp(-2.0 * logs_p)
    kl += 0.5 * (torch.exp(2.0 * logs_q) - 1.0) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    kl = kl / torch.sum(z_mask)
    return kl


# Monotonic alignment search
class MonotonicAlign:
    """Monotonic alignment search for duration extraction
    
    Pure PyTorch implementation that's slower than Cython but functional.
    """
    
    @staticmethod
    def maximum_path(neg_cent: torch.Tensor, x_mask: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
        """Find maximum path through cost matrix using dynamic programming
        
        Args:
            neg_cent: Negative log-likelihood matrix [B, T_y, T_x]
            x_mask: Text mask [B, T_x]
            y_mask: Mel mask [B, T_y]
            
        Returns:
            Path matrix [B, T_y, T_x]
        """
        b, t_y, t_x = neg_cent.shape
        path = torch.zeros(b, t_y, t_x, dtype=torch.float32, device=neg_cent.device)
        
        for b_idx in range(b):
            # Get valid lengths
            x_len = int(x_mask[b_idx].sum().item())
            y_len = int(y_mask[b_idx].sum().item())
            
            if x_len > 0 and y_len > 0:
                # Extract valid region
                cost = neg_cent[b_idx, :y_len, :x_len]
                
                # Dynamic programming to find maximum path
                # Initialize DP table
                value = torch.zeros(y_len, x_len, device=neg_cent.device)
                value[0, 0] = cost[0, 0]
                
                # Forward pass
                for y in range(1, y_len):
                    value[y, 0] = value[y-1, 0] + cost[y, 0]
                    
                for x in range(1, x_len):
                    value[0, x] = value[0, x-1] + cost[0, x]
                    
                for y in range(1, y_len):
                    for x in range(1, x_len):
                        value[y, x] = torch.max(
                            value[y-1, x] + cost[y, x],  # vertical
                            value[y, x-1] + cost[y, x]   # horizontal
                        )
                
                # Backward pass to reconstruct path
                x_idx = x_len - 1
                for y_idx in range(y_len - 1, -1, -1):
                    path[b_idx, y_idx, x_idx] = 1.0
                    
                    if x_idx > 0 and y_idx > 0:
                        # Check which direction we came from
                        if value[y_idx-1, x_idx] > value[y_idx, x_idx-1]:
                            y_idx = y_idx - 1
                        else:
                            x_idx = x_idx - 1
                    elif y_idx > 0:
                        y_idx = y_idx - 1
                    elif x_idx > 0:
                        x_idx = x_idx - 1
                        
        return path


monotonic_align = MonotonicAlign()


def f0_to_coarse(f0: torch.Tensor, f0_bin: int = 256, f0_min: float = 50.0, f0_max: float = 1100.0) -> torch.Tensor:
    """Convert F0 to coarse F0"""
    f0_mel_min = 1127 * torch.log(1 + torch.tensor(f0_min) / 700)
    f0_mel_max = 1127 * torch.log(1 + torch.tensor(f0_max) / 700)
    
    f0_mel = 1127 * torch.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    
    f0_coarse = torch.round(f0_mel).long()
    
    return f0_coarse