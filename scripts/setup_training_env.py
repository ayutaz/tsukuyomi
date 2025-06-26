#!/usr/bin/env python3
"""
Setup script for Tsukuyomi TTS training environment.
Checks system requirements and prepares the environment for training.
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil
import torch


# Colors for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

def print_colored(message, color=Colors.WHITE):
    """Print colored message."""
    print(f"{color}{message}{Colors.RESET}")

def check_python_version():
    """Check if Python version is 3.11+."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print_colored(f"✓ Python {version.major}.{version.minor}.{version.micro} detected", Colors.GREEN)
        return True
    else:
        print_colored(f"✗ Python {version.major}.{version.minor} detected. Python 3.11+ required", Colors.RED)
        return False

def check_cuda():
    """Check CUDA availability and version."""
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        gpu_count = torch.cuda.device_count()
        print_colored(f"✓ CUDA {cuda_version} available with {gpu_count} GPU(s)", Colors.GREEN)
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print_colored(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)", Colors.CYAN)
        
        # Check for CUDA 12.1+
        if cuda_version and float(cuda_version.split('.')[0]) >= 12.1:
            print_colored("  CUDA 12.1+ detected - Flash Attention 2 available", Colors.GREEN)
            return True
        else:
            print_colored("  CUDA 12.1+ recommended for optimal performance", Colors.YELLOW)
            return True
    else:
        print_colored("✗ CUDA not available. Training will be slow on CPU", Colors.YELLOW)
        return False

def check_memory():
    """Check system memory."""
    mem = psutil.virtual_memory()
    total_gb = mem.total / 1024**3
    available_gb = mem.available / 1024**3
    
    if total_gb >= 32:
        print_colored(f"✓ System memory: {total_gb:.1f} GB total, {available_gb:.1f} GB available", Colors.GREEN)
        return True
    else:
        print_colored(f"⚠ System memory: {total_gb:.1f} GB total. 32+ GB recommended", Colors.YELLOW)
        return True

def check_disk_space():
    """Check available disk space."""
    stat = shutil.disk_usage(os.getcwd())
    free_gb = stat.free / 1024**3
    
    if free_gb >= 100:
        print_colored(f"✓ Disk space: {free_gb:.1f} GB available", Colors.GREEN)
        return True
    else:
        print_colored(f"⚠ Disk space: {free_gb:.1f} GB available. 100+ GB recommended", Colors.YELLOW)
        return True

def check_docker():
    """Check if Docker is installed."""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print_colored(f"✓ Docker installed: {result.stdout.strip()}", Colors.GREEN)
            
            # Check for NVIDIA Docker
            result = subprocess.run(['docker', 'info'], capture_output=True, text=True, check=False)
            if 'nvidia' in result.stdout.lower():
                print_colored("  NVIDIA Docker runtime available", Colors.GREEN)
            else:
                print_colored("  NVIDIA Docker runtime not found (GPU support disabled in Docker)", Colors.YELLOW)
            return True
    except FileNotFoundError:
        print_colored("✗ Docker not installed", Colors.YELLOW)
        return False

def check_dependencies():
    """Check required Python packages."""
    required_packages = [
        'torch', 'numpy', 'scipy', 'librosa', 'soundfile',
        'transformers', 'accelerate', 'tensorboard', 'wandb'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if not missing:
        print_colored("✓ All required Python packages installed", Colors.GREEN)
        return True
    else:
        print_colored(f"✗ Missing packages: {', '.join(missing)}", Colors.RED)
        print_colored("  Run: uv pip install -e .", Colors.YELLOW)
        return False

def setup_directories():
    """Create necessary directories."""
    dirs = [
        'data/raw',
        'data/processed',
        'data/features',
        'models/checkpoints',
        'models/pretrained',
        'outputs/audio',
        'outputs/evaluation',
        'logs/tensorboard',
        'logs/training',
        'configs',
        'cache'
    ]
    
    print_colored("\nCreating directories:", Colors.BLUE)
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print_colored(f"  ✓ {dir_path}", Colors.GREEN)

def create_default_config():
    """Create default training configuration."""
    config = {
        "model": {
            "name": "tsukuyomi_100h",
            "type": "ultimate_acoustic_model",
            "params": {
                "hidden_size": 512,
                "num_layers": 12,
                "num_heads": 8,
                "dropout": 0.1
            }
        },
        "data": {
            "train_path": "data/processed/train",
            "val_path": "data/processed/val",
            "batch_size": 32,
            "num_workers": 4,
            "max_duration": 15.0,
            "min_duration": 0.5
        },
        "training": {
            "epochs": 100,
            "learning_rate": 0.0001,
            "warmup_steps": 4000,
            "gradient_accumulation_steps": 4,
            "mixed_precision": "bf16",
            "save_steps": 1000,
            "eval_steps": 500,
            "logging_steps": 100
        },
        "optimization": {
            "use_flash_attention": True,
            "use_gradient_checkpointing": False,
            "compile_model": True
        }
    }
    
    config_path = Path("configs/train_100h_default.yaml")
    if not config_path.exists():
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print_colored(f"\n✓ Created default config: {config_path}", Colors.GREEN)

def check_windows_specific():
    """Check Windows-specific requirements."""
    if platform.system() == 'Windows':
        print_colored("\nWindows-specific checks:", Colors.BLUE)
        
        # Check for Visual C++ Build Tools
        vs_path = Path("C:/Program Files (x86)/Microsoft Visual Studio")
        if vs_path.exists():
            print_colored("✓ Visual Studio detected", Colors.GREEN)
        else:
            print_colored("⚠ Visual Studio not found. Required for some packages", Colors.YELLOW)
        
        # Check for WSL2
        try:
            result = subprocess.run(['wsl', '--status'], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                print_colored("✓ WSL2 available", Colors.GREEN)
            else:
                print_colored("⚠ WSL2 not available. Consider installing for better compatibility", Colors.YELLOW)
        except:
            print_colored("⚠ WSL2 not available", Colors.YELLOW)

def main():
    """Main setup function."""
    print_colored("\n🌙 Tsukuyomi TTS Training Environment Setup", Colors.MAGENTA)
    print_colored("=" * 50, Colors.MAGENTA)
    
    # System checks
    print_colored("\nSystem Requirements Check:", Colors.BLUE)
    checks = [
        ("Python Version", check_python_version()),
        ("CUDA Support", check_cuda()),
        ("System Memory", check_memory()),
        ("Disk Space", check_disk_space()),
        ("Docker", check_docker()),
        ("Dependencies", check_dependencies())
    ]
    
    # Windows-specific checks
    if platform.system() == 'Windows':
        check_windows_specific()
    
    # Summary
    all_passed = all(check[1] for check in checks)
    
    print_colored("\n" + "=" * 50, Colors.MAGENTA)
    if all_passed:
        print_colored("✓ All checks passed!", Colors.GREEN)
    else:
        print_colored("⚠ Some checks failed or have warnings", Colors.YELLOW)
    
    # Setup directories
    setup_directories()
    
    # Create default config
    create_default_config()
    
    # Training recommendations
    print_colored("\nTraining Recommendations:", Colors.BLUE)
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if vram >= 80:  # H100
            print_colored("  • H100 detected: Use batch size 64-128", Colors.GREEN)
            print_colored("  • Enable Flash Attention 2", Colors.GREEN)
            print_colored("  • Use BF16 mixed precision", Colors.GREEN)
        elif vram >= 40:  # A100
            print_colored("  • A100 detected: Use batch size 32-64", Colors.GREEN)
            print_colored("  • Enable gradient checkpointing if needed", Colors.GREEN)
        else:
            print_colored("  • Use smaller batch size (8-16)", Colors.YELLOW)
            print_colored("  • Enable gradient accumulation", Colors.YELLOW)
    
    print_colored("\nSetup complete! 🚀", Colors.GREEN)
    print_colored("\nNext steps:", Colors.CYAN)
    print_colored("1. Prepare your dataset in data/raw/", Colors.WHITE)
    print_colored("2. Run preprocessing: python scripts/preprocess.py", Colors.WHITE)
    print_colored("3. Start training: python scripts/train.py --config configs/train_100h.yaml", Colors.WHITE)
    print_colored("\nFor Docker training:", Colors.CYAN)
    if platform.system() == 'Windows':
        print_colored("  .\\scripts\\docker_run.bat train", Colors.WHITE)
    else:
        print_colored("  ./scripts/docker_run.sh train", Colors.WHITE)

if __name__ == "__main__":
    main()
