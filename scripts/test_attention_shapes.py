#!/usr/bin/env python3
"""アテンション機構の形状問題をテストするスクリプト"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

from src.models.modules import MultiHeadAttention


def test_attention_shapes():
    """MultiHeadAttentionの形状をテスト"""
    
    # パラメータ
    batch_size = 16
    channels = 192
    n_heads = 2
    seq_len = 50
    
    print("Testing MultiHeadAttention with:")
    print(f"  Batch size: {batch_size}")
    print(f"  Channels: {channels}")
    print(f"  Heads: {n_heads}")
    print(f"  Sequence length: {seq_len}")
    
    # MultiHeadAttentionモジュールを作成
    mha = MultiHeadAttention(
        channels=channels,
        out_channels=channels,
        n_heads=n_heads,
        p_dropout=0.0
    )
    
    # ダミー入力を作成
    x = torch.randn(batch_size, channels, seq_len)
    c = torch.randn(batch_size, channels, seq_len)
    
    # マスクを作成（様々な形状をテスト）
    print("\n=== Testing different mask shapes ===")
    
    # ケース1: マスクなし
    print("\n1. No mask:")
    try:
        output = mha(x, c, attn_mask=None)
        print(f"  Success! Output shape: {output.shape}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # ケース2: 2Dマスク [B, T]
    print("\n2. 2D mask [B, T]:")
    mask_2d = torch.ones(batch_size, seq_len)
    try:
        output = mha(x, c, attn_mask=mask_2d)
        print(f"  Success! Output shape: {output.shape}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # ケース3: 3Dマスク [B, 1, T]
    print("\n3. 3D mask [B, 1, T]:")
    mask_3d = torch.ones(batch_size, 1, seq_len)
    try:
        output = mha(x, c, attn_mask=mask_3d)
        print(f"  Success! Output shape: {output.shape}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # ケース4: 4Dマスク [B, 1, 1, T]
    print("\n4. 4D mask [B, 1, 1, T]:")
    mask_4d = torch.ones(batch_size, 1, 1, seq_len)
    try:
        output = mha(x, c, attn_mask=mask_4d)
        print(f"  Success! Output shape: {output.shape}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # ケース5: 誤った形状のマスク [B, n_heads, T]
    print("\n5. Incorrect mask [B, n_heads, T]:")
    mask_wrong = torch.ones(batch_size, n_heads, seq_len)
    try:
        output = mha(x, c, attn_mask=mask_wrong)
        print(f"  Success! Output shape: {output.shape}")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    test_attention_shapes()
