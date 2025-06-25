#!/usr/bin/env python3
"""Test installation and dependency compatibility"""

import sys
import importlib
import subprocess
from pathlib import Path

def test_import(module_name: str, package_name: str = None) -> bool:
    """Test if a module can be imported"""
    if package_name is None:
        package_name = module_name
    
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: Import failed - {e}")
        return False

def test_cuda():
    """Test CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  CUDA: Not available (CPU mode)")
    except Exception as e:
        print(f"❌ CUDA: Error - {e}")

def test_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor == 11:
        print(f"✅ Python: {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"⚠️  Python: {version.major}.{version.minor}.{version.micro} (3.11 recommended)")

def main():
    print("=== Tsukuyomi TTS Installation Test ===\n")
    
    # Test Python version
    test_python_version()
    
    # Core dependencies
    print("\n--- Core Dependencies ---")
    core_modules = [
        ("torch", "PyTorch"),
        ("torchaudio", "TorchAudio"),
        ("transformers", "Transformers"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("librosa", "librosa"),
    ]
    
    for module, name in core_modules:
        test_import(module, name)
    
    # Test CUDA
    print("\n--- CUDA Support ---")
    test_cuda()
    
    # Training dependencies
    print("\n--- Training Dependencies ---")
    training_modules = [
        ("omegaconf", "OmegaConf"),
        ("hydra", "Hydra"),
        ("accelerate", "Accelerate"),
        ("wandb", "Weights & Biases"),
        ("tensorboard", "TensorBoard"),
        ("einops", "Einops"),
    ]
    
    for module, name in training_modules:
        test_import(module, name)
    
    # Evaluation dependencies
    print("\n--- Evaluation Dependencies ---")
    eval_modules = [
        ("pesq", "PESQ"),
        ("pystoi", "PySTOI"),
        ("jiwer", "JiWER"),
        ("seaborn", "Seaborn"),
    ]
    
    for module, name in eval_modules:
        test_import(module, name)
    
    # Japanese text processing
    print("\n--- Japanese Text Processing ---")
    japanese_modules = [
        ("pyopenjtalk", "pyOpenJTalk Plus"),
        ("jaconv", "jaconv"),
        ("pykakasi", "PyKakasi"),
        ("unidic_lite", "UniDic Lite"),
    ]
    
    for module, name in japanese_modules:
        test_import(module, name)
    
    # Project modules
    print("\n--- Tsukuyomi Modules ---")
    project_modules = [
        ("src.models.xphonebert", "XPhoneBERT"),
        ("src.models.f0_bert", "F0-BERT"),
        ("src.models.vits", "VITS"),
        ("src.models.matcha_tts", "Matcha-TTS"),
        ("src.models.bigvgan_v2", "BigVGAN-v2"),
    ]
    
    all_success = True
    for module, name in project_modules:
        if not test_import(module, name):
            all_success = False
    
    # Summary
    print("\n=== Summary ===")
    if all_success:
        print("✅ All tests passed! Tsukuyomi TTS is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())