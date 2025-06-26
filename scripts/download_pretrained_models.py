#!/usr/bin/env python3
"""
Download and setup pre-trained TTS models for immediate use
"""

import json
import logging
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Model configurations
MODELS = {
    "style-bert-vits2-jp": {
        "url": "https://huggingface.co/litagin/style_bert_vits2_jp/resolve/main/model.zip",
        "size": "1.2GB",
        "md5": None,  # Will be updated when official
        "description": "Style-BERT-VITS2 Japanese base model",
        "license": "AGPL-3.0",
    },
    "vits2-jp": {
        "url": "https://huggingface.co/spaces/skytnt/moe-tts/resolve/main/pretrained/vits2_jp.pth",
        "size": "145MB",
        "md5": None,
        "description": "VITS2 Japanese model",
        "license": "MIT",
    },
    "bert-vits2-jp": {
        "url": "https://huggingface.co/Plachta/BERT-VITS2-JP/resolve/main/model.pth",
        "size": "380MB",
        "md5": None,
        "description": "Bert-VITS2 Japanese model",
        "license": "AGPL-3.0",
    },
}


def download_file(url: str, dest_path: Path, expected_size: str = None) -> bool:
    """Download file with progress indicator."""
    try:
        logger.info(f"Downloading from {url}")
        logger.info(f"Expected size: {expected_size}")

        # Create parent directory
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            mb_downloaded = downloaded / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            sys.stdout.write(
                f"\rDownloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)"
            )
            sys.stdout.flush()

        urllib.request.urlretrieve(url, dest_path, reporthook=download_progress)
        print()  # New line after progress

        logger.info(f"Downloaded to {dest_path}")
        return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """Extract zip or tar archive."""
    try:
        extract_to.mkdir(parents=True, exist_ok=True)

        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in [".tar", ".gz", ".tgz"]:
            with tarfile.open(archive_path, "r:*") as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            logger.error(f"Unknown archive format: {archive_path.suffix}")
            return False

        logger.info(f"Extracted to {extract_to}")
        return True

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def setup_model(model_name: str, models_dir: Path = Path("models")) -> bool:
    """Download and setup a pre-trained model."""
    if model_name not in MODELS:
        logger.error(f"Unknown model: {model_name}")
        logger.info(f"Available models: {list(MODELS.keys())}")
        return False

    model_info = MODELS[model_name]
    model_dir = models_dir / model_name

    # Check if already exists
    if model_dir.exists() and any(model_dir.iterdir()):
        logger.info(f"Model {model_name} already exists at {model_dir}")
        return True

    # Show model info
    logger.info(f"\nModel: {model_name}")
    logger.info(f"Description: {model_info['description']}")
    logger.info(f"License: {model_info['license']}")
    logger.info(f"Size: {model_info['size']}")

    # Confirm download
    response = input("\nDo you want to download this model? (y/n): ")
    if response.lower() != "y":
        logger.info("Download cancelled")
        return False

    # Download
    temp_file = models_dir / f"{model_name}_temp.download"
    if not download_file(model_info["url"], temp_file, model_info["size"]):
        return False

    # Extract if archive
    if temp_file.suffix in [".zip", ".tar", ".gz", ".tgz"]:
        if not extract_archive(temp_file, model_dir):
            return False
        temp_file.unlink()  # Remove archive
    else:
        # Move single file
        model_dir.mkdir(parents=True, exist_ok=True)
        temp_file.rename(model_dir / temp_file.name.replace("_temp.download", ""))

    # Create metadata
    metadata = {
        "model_name": model_name,
        "source": model_info["url"],
        "license": model_info["license"],
        "description": model_info["description"],
    }

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✅ Model {model_name} setup complete!")
    return True


def setup_all_models():
    """Interactive setup for all models."""
    print("=== Tsukuyomi Pre-trained Model Setup ===\n")
    print("This script will help you download pre-trained TTS models.")
    print("Note: Some models have restrictive licenses. Check before commercial use.\n")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Show available models
    print("Available models:")
    for name, info in MODELS.items():
        print(f"  - {name}: {info['description']} ({info['size']}, {info['license']})")

    print("\nOptions:")
    print("1. Download all models")
    print("2. Download specific model")
    print("3. Exit")

    choice = input("\nEnter your choice (1-3): ")

    if choice == "1":
        for model_name in MODELS:
            print(f"\n{'='*60}")
            setup_model(model_name, models_dir)
    elif choice == "2":
        model_name = input("Enter model name: ")
        setup_model(model_name, models_dir)
    else:
        print("Exiting...")


def create_model_config():
    """Create configuration for downloaded models."""
    config = {
        "default_model": "vits2-jp",  # MIT license, good default
        "models": {},
        "inference": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "use_fp16": True,
            "batch_size": 1,
        },
    }

    models_dir = Path("models")
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir() and (model_dir / "metadata.json").exists():
            with open(model_dir / "metadata.json") as f:
                metadata = json.load(f)

            config["models"][model_dir.name] = {
                "path": str(model_dir),
                "type": model_dir.name.split("-")[0],  # Extract model type
                "license": metadata.get("license", "Unknown"),
                "enabled": True,
            }

    config_path = Path("config/pretrained_models.json")
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Created model configuration at {config_path}")


if __name__ == "__main__":
    import torch  # Check CUDA availability

    setup_all_models()

    # Create config if any models were downloaded
    if any(Path("models").iterdir()):
        create_model_config()

        print("\n✅ Setup complete!")
        print("\nNext steps:")
        print("1. Test models: python scripts/test_pretrained_models.py")
        print("2. Fine-tune on your data: python scripts/finetune_model.py")
        print("3. Export for Unity: python scripts/export_onnx.py")
