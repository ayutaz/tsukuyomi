#!/usr/bin/env python3
"""
Test script to verify CUDA 12.1+ optimizations in Tsukuyomi TTS.

This script checks:
1. CUDA version requirements
2. Flash Attention 2 availability
3. BF16 support
4. CUDA graphs functionality
5. Triton kernel compilation
6. Model optimization features
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ultimate_acoustic_model import UltimateAcousticModel, UltimateAcousticConfig
from src.models.bigvgan_v2 import BigVGANv2Generator, BigVGANv2Config
from src.models.f0_bert import F0BERT, F0BERTConfig
from src.models.xphonebert_japanese import XPhoneBERTJapanese, XPhoneBERTJapaneseConfig


def check_cuda_version():
    """Check if CUDA 12.1+ is available."""
    print("=== CUDA Version Check ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return False
    
    cuda_version = torch.version.cuda
    print(f"CUDA version: {cuda_version}")
    
    if cuda_version:
        major, minor = map(int, cuda_version.split('.')[:2])
        if major > 12 or (major == 12 and minor >= 1):
            print("✅ CUDA 12.1+ detected")
            return True
        else:
            print(f"❌ CUDA 12.1+ required, found {cuda_version}")
            return False
    
    return False


def check_flash_attention():
    """Check if Flash Attention 2 is available."""
    print("\n=== Flash Attention 2 Check ===")
    
    try:
        import flash_attn
        print("✅ Flash Attention 2 is available")
        print(f"Version: {flash_attn.__version__}")
        
        # Test Flash Attention
        batch_size, seq_len, n_heads, head_dim = 2, 128, 8, 64
        q = torch.randn(batch_size, seq_len, n_heads, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(batch_size, seq_len, n_heads, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(batch_size, seq_len, n_heads, head_dim, device='cuda', dtype=torch.float16)
        
        from flash_attn import flash_attn_func
        output = flash_attn_func(q, k, v)
        print(f"✅ Flash Attention test passed - output shape: {output.shape}")
        return True
        
    except ImportError:
        print("❌ Flash Attention 2 not available")
        return False
    except Exception as e:
        print(f"❌ Flash Attention test failed: {e}")
        return False


def check_bf16_support():
    """Check BF16 support."""
    print("\n=== BF16 Support Check ===")
    
    if torch.cuda.is_bf16_supported():
        print("✅ BF16 is supported")
        
        # Test BF16 operations
        x = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')
        y = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')
        z = torch.matmul(x, y)
        print(f"✅ BF16 matmul test passed - output shape: {z.shape}")
        return True
    else:
        print("❌ BF16 not supported")
        return False


def check_cuda_graphs():
    """Check CUDA graphs functionality."""
    print("\n=== CUDA Graphs Check ===")
    
    if not hasattr(torch.cuda, 'graph'):
        print("❌ CUDA graphs not available")
        return False
    
    try:
        # Simple model for testing
        model = torch.nn.Linear(128, 128).cuda()
        model.eval()
        
        # Static input
        static_input = torch.randn(32, 128, device='cuda')
        
        # Warmup
        with torch.no_grad():
            _ = model(static_input)
        
        # Capture graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_output = model(static_input)
        
        # Replay
        g.replay()
        
        print("✅ CUDA graphs are working")
        return True
        
    except Exception as e:
        print(f"❌ CUDA graphs test failed: {e}")
        return False


def check_triton():
    """Check Triton availability."""
    print("\n=== Triton Check ===")
    
    try:
        import triton
        print("✅ Triton is available")
        print(f"Version: {triton.__version__}")
        return True
    except ImportError:
        print("❌ Triton not available")
        return False


def test_model_optimizations():
    """Test model-specific CUDA 12.1+ optimizations."""
    print("\n=== Model Optimization Tests ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test Ultimate Acoustic Model
    print("\n1. Testing Ultimate Acoustic Model...")
    try:
        config = UltimateAcousticConfig(use_bf16=True, gradient_checkpointing=False)
        model = UltimateAcousticModel(config).to(device)
        model.eval()
        
        # Test forward pass
        batch_size = 2
        phoneme_embeddings = torch.randn(batch_size, 512, 100, device=device)
        phoneme_lengths = torch.tensor([100, 90], device=device)
        
        with torch.no_grad():
            output = model.inference(phoneme_embeddings, phoneme_lengths)
        
        print(f"✅ Acoustic model inference passed - output shape: {output.shape}")
        
        # Test compilation
        example_input = {
            'phoneme_embeddings': phoneme_embeddings,
            'phoneme_lengths': phoneme_lengths
        }
        model.compile_for_inference(example_input)
        print("✅ Acoustic model compilation completed")
        
    except Exception as e:
        print(f"❌ Acoustic model test failed: {e}")
    
    # Test BigVGAN-v2
    print("\n2. Testing BigVGAN-v2...")
    try:
        config = BigVGANv2Config()
        vocoder = BigVGANv2Generator(config).to(device)
        vocoder.eval()
        
        # Test forward pass
        mel = torch.randn(2, 128, 100, device=device)
        with torch.no_grad():
            waveform = vocoder(mel)
        
        print(f"✅ BigVGAN-v2 inference passed - output shape: {waveform.shape}")
        
        # Test compilation
        vocoder.compile_for_inference(mel)
        print("✅ BigVGAN-v2 compilation completed")
        
    except Exception as e:
        print(f"❌ BigVGAN-v2 test failed: {e}")
    
    # Test F0-BERT
    print("\n3. Testing F0-BERT...")
    try:
        config = F0BERTConfig()
        f0_model = F0BERT(config).to(device)
        f0_model.eval()
        
        # Test optimization
        f0_model.optimize_for_inference()
        print("✅ F0-BERT optimization completed")
        
    except Exception as e:
        print(f"❌ F0-BERT test failed: {e}")
    
    # Test XPhoneBERT-Japanese
    print("\n4. Testing XPhoneBERT-Japanese...")
    try:
        model = XPhoneBERTJapanese().to(device)
        model.eval()
        
        # Test optimization
        model.optimize_for_inference()
        print("✅ XPhoneBERT-Japanese optimization completed")
        
    except Exception as e:
        print(f"❌ XPhoneBERT-Japanese test failed: {e}")


def main():
    """Run all checks."""
    print("🚀 Tsukuyomi CUDA 12.1+ Optimization Test\n")
    
    # Run checks
    cuda_ok = check_cuda_version()
    flash_ok = check_flash_attention()
    bf16_ok = check_bf16_support()
    graphs_ok = check_cuda_graphs()
    triton_ok = check_triton()
    
    # Run model tests only if CUDA is available
    if cuda_ok:
        test_model_optimizations()
    
    # Summary
    print("\n=== Summary ===")
    print(f"CUDA 12.1+: {'✅' if cuda_ok else '❌'}")
    print(f"Flash Attention 2: {'✅' if flash_ok else '❌'}")
    print(f"BF16 Support: {'✅' if bf16_ok else '❌'}")
    print(f"CUDA Graphs: {'✅' if graphs_ok else '❌'}")
    print(f"Triton: {'✅' if triton_ok else '❌'}")
    
    if cuda_ok and flash_ok and bf16_ok:
        print("\n✅ All critical CUDA 12.1+ features are available!")
        print("Your system is ready for optimal Tsukuyomi TTS performance.")
    else:
        print("\n⚠️ Some CUDA 12.1+ features are missing.")
        print("Please ensure you have CUDA 12.1+ installed with all required packages.")


if __name__ == "__main__":
    main()