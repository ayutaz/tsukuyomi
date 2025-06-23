#!/usr/bin/env python3
"""LJSpeech形式のデータセットで学習するスクリプト"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.ljspeech_dataset import create_ljspeech_datasets
from src.models.vits import VITS
from src.models.matcha_tts import MatchaTTS
from src.models.bigvgan import BigVGAN
from src.models.xphonebert import XPhoneBERT
from src.models.f0_bert import F0BERT
from src.training.trainer import TTSTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LJSpeechTrainer(TTSTrainer):
    """LJSpeech形式のデータセット用トレーナー"""
    
    def setup_data_loaders(self):
        """データローダーのセットアップ"""
        logger.info("Setting up LJSpeech data loaders...")
        
        # LJSpeech形式のデータローダー作成
        self.train_loader, self.val_loader = create_ljspeech_datasets(
            data_dir=self.config.data.data_dir,
            sample_rate=self.config.data.sample_rate,
            n_mels=self.config.data.n_mels,
            n_fft=self.config.data.n_fft,
            hop_length=self.config.data.hop_length,
            win_length=self.config.data.win_length,
            f_min=self.config.data.f_min,
            f_max=self.config.data.f_max,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.data.num_workers,
            max_audio_len=self.config.data.get('max_audio_len', None),
            min_audio_len=self.config.data.get('min_audio_len', 1024),
        )
        
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
        
    def train_step(self, batch):
        """学習ステップ"""
        # データの取得
        audio = batch['audio'].to(self.device)
        mel = batch['mel'].to(self.device)
        text = batch['text'].to(self.device)
        text_lengths = batch['text_lengths'].to(self.device)
        mel_lengths = batch['mel_lengths'].to(self.device)
        speaker_ids = batch['speaker_id'].to(self.device)
        
        # モデルによって処理を分岐
        if self.config.models.acoustic_model == "vits":
            # VITS用の処理
            outputs = self.model(
                text=text,
                text_lengths=text_lengths,
                mel=mel,
                mel_lengths=mel_lengths,
                speaker_ids=speaker_ids,
            )
            
            losses = outputs['losses']
            total_loss = sum(losses.values())
            
        elif self.config.models.acoustic_model == "matcha_tts":
            # Matcha-TTS用の処理
            outputs = self.model(
                text=text,
                text_lengths=text_lengths,
                mel=mel,
                mel_lengths=mel_lengths,
                speaker_ids=speaker_ids,
            )
            
            losses = outputs['losses']
            total_loss = sum(losses.values())
            
        else:
            raise ValueError(f"Unknown acoustic model: {self.config.models.acoustic_model}")
            
        # バックプロパゲーション
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 勾配クリッピング
        if self.config.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.gradient_clip
            )
            
        self.optimizer.step()
        
        # ロギング
        metrics = {
            'loss/total': total_loss.item(),
        }
        for name, loss in losses.items():
            metrics[f'loss/{name}'] = loss.item()
            
        return metrics
        
    @torch.no_grad()
    def validation_step(self, batch):
        """検証ステップ"""
        # データの取得
        audio = batch['audio'].to(self.device)
        mel = batch['mel'].to(self.device)
        text = batch['text'].to(self.device)
        text_lengths = batch['text_lengths'].to(self.device)
        mel_lengths = batch['mel_lengths'].to(self.device)
        speaker_ids = batch['speaker_id'].to(self.device)
        
        # 推論
        if self.config.models.acoustic_model == "vits":
            outputs = self.model(
                text=text,
                text_lengths=text_lengths,
                mel=mel,
                mel_lengths=mel_lengths,
                speaker_ids=speaker_ids,
            )
        else:
            outputs = self.model(
                text=text,
                text_lengths=text_lengths,
                mel=mel,
                mel_lengths=mel_lengths,
                speaker_ids=speaker_ids,
            )
            
        losses = outputs['losses']
        total_loss = sum(losses.values())
        
        metrics = {
            'val_loss/total': total_loss.item(),
        }
        for name, loss in losses.items():
            metrics[f'val_loss/{name}'] = loss.item()
            
        return metrics


def setup_models(config):
    """モデルのセットアップ"""
    models = {}
    
    # XPhoneBERTの初期化（必要な場合）
    if config.models.xphonebert.enabled:
        models['xphonebert'] = XPhoneBERT(
            model_name=config.models.xphonebert.model_name,
            hidden_size=config.models.xphonebert.hidden_size,
            num_layers=config.models.xphonebert.num_layers,
            num_heads=config.models.xphonebert.num_heads,
        )
        
    # F0-BERTの初期化（必要な場合）
    if config.models.f0_bert.enabled:
        models['f0_bert'] = F0BERT(
            hidden_size=config.models.f0_bert.hidden_size,
            num_layers=config.models.f0_bert.num_layers,
            num_heads=config.models.f0_bert.num_heads,
            pitch_bins=config.models.f0_bert.pitch_bins,
        )
        
    # 音響モデルの初期化
    if config.models.acoustic_model == "vits":
        models['acoustic'] = VITS(
            n_vocab=config.models.vits.n_vocab,
            n_speakers=config.models.vits.n_speakers,
            hidden_channels=config.models.vits.hidden_channels,
            filter_channels=config.models.vits.filter_channels,
            n_heads=config.models.vits.n_heads,
            n_layers=config.models.vits.n_layers,
            kernel_size=config.models.vits.kernel_size,
            p_dropout=config.models.vits.p_dropout,
        )
    elif config.models.acoustic_model == "matcha_tts":
        models['acoustic'] = MatchaTTS(
            n_vocab=config.models.matcha.n_vocab,
            n_speakers=config.models.matcha.n_speakers,
            hidden_channels=config.models.matcha.hidden_channels,
            filter_channels=config.models.matcha.filter_channels,
            n_heads=config.models.matcha.n_heads,
            n_layers=config.models.matcha.n_layers,
        )
        
    # ボコーダーの初期化
    models['vocoder'] = BigVGAN(
        num_mels=config.models.bigvgan.num_mels,
        upsample_initial_channel=config.models.bigvgan.upsample_initial_channel,
        resblock_kernel_sizes=config.models.bigvgan.resblock_kernel_sizes,
        resblock_dilation_sizes=config.models.bigvgan.resblock_dilation_sizes,
        upsample_rates=config.models.bigvgan.upsample_rates,
        upsample_kernel_sizes=config.models.bigvgan.upsample_kernel_sizes,
    )
    
    return models


def main():
    parser = argparse.ArgumentParser(description="LJSpeech形式での学習")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="設定ファイルのパス",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="LJSpeech形式のデータディレクトリ",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="再開用のチェックポイント",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/ljspeech",
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="エポック数（設定を上書き）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="バッチサイズ（設定を上書き）",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="学習率（設定を上書き）",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="分散学習を有効化",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="分散学習のローカルランク",
    )
    
    args = parser.parse_args()
    
    # 設定の読み込み
    config = OmegaConf.load(args.config)
    
    # コマンドライン引数で設定を上書き
    config.data.data_dir = args.data_dir
    config.training.checkpoint_dir = args.output_dir
    
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.optimizers.default.lr = args.learning_rate
        
    # 分散学習の設定
    if args.distributed:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        
    # モデルのセットアップ
    models = setup_models(config)
    
    # 統合モデルの作成（簡易版）
    # 実際にはすべてのコンポーネントを統合する必要があります
    model = models['acoustic']
    
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank])
        
    # トレーナーの作成
    trainer = LJSpeechTrainer(config)
    trainer.model = model
    trainer.setup_data_loaders()
    trainer.setup_optimizers()
    
    # チェックポイントからの再開
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        
    # 学習の実行
    logger.info("Starting training...")
    trainer.train()
    
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()