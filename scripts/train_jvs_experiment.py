#!/usr/bin/env python3
"""JVS実験用学習スクリプト

RTX 4070 Ti Super (16GB VRAM) に最適化
"""

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.cuda
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.training.trainer import TTSTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_gpu():
    """GPU情報を確認"""
    if not torch.cuda.is_available():
        logger.error("CUDAが利用できません")
        return False

    gpu_count = torch.cuda.device_count()
    logger.info(f"検出されたGPU数: {gpu_count}")

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name}")
        logger.info(f"  - メモリ: {props.total_memory / 1024**3:.1f} GB")
        logger.info(f"  - Compute Capability: {props.major}.{props.minor}")

        # 4070 Ti Superの確認
        if "4070 Ti" in props.name:
            logger.info("  ✓ RTX 4070 Ti Super を検出しました")

    # メモリ情報
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    logger.info(f"現在のGPUメモリ使用量: {allocated:.2f} GB / {reserved:.2f} GB")

    return True


def optimize_for_4070ti():
    """RTX 4070 Ti Super向けの最適化設定"""
    # PyTorch設定
    torch.backends.cudnn.benchmark = True  # 自動最適化
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32有効化（4070 Ti Superで高速）
    torch.backends.cudnn.allow_tf32 = True

    # メモリ管理
    torch.cuda.empty_cache()
    gc.collect()

    # 環境変数
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:128"  # メモリフラグメンテーション対策
    )

    logger.info("RTX 4070 Ti Super向けの最適化を適用しました")


def monitor_memory(step: int, phase: str):
    """メモリ使用量を監視"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3

        logger.info(
            f"[{phase}] Step {step} - GPU メモリ: "
            f"使用 {allocated:.2f}GB / 予約 {reserved:.2f}GB / 最大 {max_allocated:.2f}GB"
        )

        # メモリ不足の警告
        if reserved > 14.0:  # 16GBの87.5%
            logger.warning(
                "GPUメモリ使用量が高くなっています。バッチサイズの削減を検討してください。"
            )

        # 定期的なキャッシュクリア
        if step % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()


def create_jvs_trainer(config_path: Path, data_dir: Path):
    """JVS用のトレーナーを作成"""
    # 設定の読み込み
    config = OmegaConf.load(config_path)

    # データディレクトリの更新
    if data_dir:
        # LJSpeech形式の場合
        if (data_dir / "metadata.csv").exists():
            config.data.train_dir = str(data_dir)
            config.data.val_dir = str(data_dir)
            config.data.dataset_format = "ljspeech"
        else:
            # JVS形式の場合
            config.data.train_dir = str(data_dir)
            config.data.val_dir = str(data_dir)

    # メモリ最適化設定の確認
    if hasattr(config.training, "memory_optimization"):
        logger.info("メモリ最適化設定:")
        logger.info(
            f"  - Gradient Checkpointing: {config.training.memory_optimization.gradient_checkpointing}"
        )
        logger.info(
            f"  - Clear Cache Interval: {config.training.memory_optimization.clear_cache_interval}"
        )

    # トレーナーの作成
    trainer = TTSTrainer(config)

    # メモリ監視コールバックの追加
    original_train_epoch = trainer.train_epoch

    def train_epoch_with_monitor(*args, **kwargs):
        epoch = args[0] if args else kwargs.get("epoch", 0)
        monitor_memory(epoch, "EPOCH_START")
        result = original_train_epoch(*args, **kwargs)
        monitor_memory(epoch, "EPOCH_END")
        return result

    trainer.train_epoch = train_epoch_with_monitor

    return trainer


def run_experiment(config_path: Path, data_dir: Path, num_epochs: int = 10):
    """実験の実行"""
    logger.info("=" * 50)
    logger.info("JVS実験を開始します")
    logger.info("=" * 50)

    # GPU確認
    if not check_gpu():
        return

    # 4070 Ti Super最適化
    optimize_for_4070ti()

    # トレーナー作成
    trainer = create_jvs_trainer(config_path, data_dir)

    # 実験用の短期学習
    if num_epochs:
        trainer.config.training.num_epochs = num_epochs

    logger.info("\n実験設定:")
    logger.info(f"  - エポック数: {trainer.config.training.num_epochs}")
    logger.info(f"  - バッチサイズ: {trainer.config.training.batch_size}")
    logger.info(f"  - 勾配累積: {trainer.config.training.gradient_accumulation_steps}")
    logger.info(f"  - 混合精度: {trainer.config.training.mixed_precision}")
    logger.info(f"  - 話者数: {len(trainer.config.data.use_speakers)}")

    # 学習開始
    start_time = time.time()

    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("GPUメモリ不足エラーが発生しました")
            logger.error("以下の対策を試してください:")
            logger.error(
                "1. バッチサイズを小さくする (現在: %d)",
                trainer.config.training.batch_size,
            )
            logger.error("2. gradient_accumulation_steps を増やす")
            logger.error("3. モデルサイズを小さくする")
            logger.error("4. gradient_checkpointing を有効にする")
            raise
        else:
            raise

    elapsed_time = time.time() - start_time
    logger.info(f"\n実験完了！ 総時間: {elapsed_time / 60:.1f}分")

    # 最終的なメモリ使用量
    monitor_memory(-1, "FINAL")


def main():
    parser = argparse.ArgumentParser(description="JVS実験用学習")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_jvs_4070ti.yaml",
        help="設定ファイル",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="JVSデータディレクトリ（サブセット）",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="実験用エポック数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="バッチサイズ（設定を上書き）",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="プロファイリングを有効化",
    )

    args = parser.parse_args()

    # 設定の調整
    if args.batch_size:
        config = OmegaConf.load(args.config)
        config.training.batch_size = args.batch_size
        temp_config = Path("configs/temp_experiment.yaml")
        OmegaConf.save(config, temp_config)
        args.config = str(temp_config)

    # プロファイリング
    if args.profile:
        from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=tensorboard_trace_handler("./logs/profiler"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            run_experiment(
                Path(args.config),
                Path(args.data_dir) if args.data_dir else None,
                args.num_epochs,
            )

        logger.info("プロファイリング結果を ./logs/profiler に保存しました")
        logger.info("表示: tensorboard --logdir=./logs/profiler")
    else:
        run_experiment(
            Path(args.config),
            Path(args.data_dir) if args.data_dir else None,
            args.num_epochs,
        )


if __name__ == "__main__":
    main()
