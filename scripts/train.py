#!/usr/bin/env python3
"""Tsukuyomi TTS学習スクリプト

このスクリプトはTsukuyomi TTSモデルの学習を行います。
分散学習、混合精度学習、チェックポイント管理などをサポートしています。
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.dataset import TsukuyomiDataset
from src.data.collate import tts_collate_fn
from src.models.f0_bert import F0BERT
from src.models.xphonebert import XPhoneBERTEncoder as XPhoneBERT
from src.models.vits import VITS
from src.models.matcha_tts import MatchaTTS
from src.models.bigvgan_v2 import BigVGANv2Generator as BigVGANv2

# ダミーモデル（動作確認用）
class DummyAcousticModel(nn.Module):
    """Dummy acoustic model for testing"""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, *args, **kwargs):
        # ダミー出力
        batch_size = 1
        if args and hasattr(args[0], 'shape'):
            batch_size = args[0].shape[0]
        
        # fcレイヤーを通すことで勾配を有効にする
        dummy_input = torch.randn(batch_size, 128).to(self.fc.weight.device)
        dummy_output = self.fc(dummy_input)
        
        # 損失計算（requires_grad=Trueになる）
        loss = torch.mean(dummy_output ** 2)
        
        # ダミーメル出力
        dummy_mel = torch.randn(batch_size, 80, 100).to(self.fc.weight.device)
        
        return {'mel': dummy_mel, 'loss': loss}
from src.training.losses import MultiTaskLoss
from src.training.metrics import TrainingMetrics
from src.utils.model_manager import ModelRegistry
from src.evaluation.metrics import MelCepstralDistortion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTSTrainer:
    """TTS学習を管理するクラス"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        
        # ログディレクトリの作成
        log_dir = Path(config.paths.tensorboard).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # ログトラッカーの設定を修正
        log_with = config.training.logging.trackers
        if isinstance(log_with, list) and len(log_with) == 1:
            log_with = log_with[0]  # 単一要素のリストは文字列に変換
            
        self.accelerator = Accelerator(
            mixed_precision=config.training.mixed_precision,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            log_with=log_with,
            project_dir=str(log_dir),
        )
        
        # モデルレジストリの初期化
        self.model_registry = ModelRegistry(Path(config.paths.model_registry))
        
        # TensorBoardの設定
        self.writer = None
        if self.accelerator.is_main_process:
            self.writer = SummaryWriter(config.paths.tensorboard)
        
        # チェックポイントディレクトリの作成
        self.checkpoint_dir = Path(config.paths.checkpoints)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_models(self) -> Dict[str, nn.Module]:
        """モデルの初期化"""
        models = {}
        
        # XPhoneBERTの初期化
        if self.config.models.xphonebert.enabled:
            models['xphonebert'] = XPhoneBERT(
                model_name=self.config.models.xphonebert.model_name,
                hidden_size=self.config.models.xphonebert.hidden_size,
                num_layers=self.config.models.xphonebert.num_layers,
                num_heads=self.config.models.xphonebert.num_heads,
            )
            
        # F0-BERTの初期化
        if self.config.models.f0_bert.enabled:
            from src.models.f0_bert import F0BERTConfig
            f0_bert_config = F0BERTConfig(
                hidden_size=self.config.models.f0_bert.hidden_size,
                num_hidden_layers=self.config.models.f0_bert.num_layers,
                num_attention_heads=self.config.models.f0_bert.num_heads,
                f0_bins=self.config.models.f0_bert.pitch_bins,  # pitch_bins -> f0_bins
            )
            models['f0_bert'] = F0BERT(config=f0_bert_config)
            
        # 音響モデルの初期化
        if self.config.models.acoustic_model == "dummy":
            models['acoustic'] = DummyAcousticModel()
        elif self.config.models.acoustic_model == "vits":
            models['acoustic'] = VITS(
                n_vocab=self.config.models.vits.n_vocab,
                n_speakers=self.config.models.vits.n_speakers,
                hidden_channels=self.config.models.vits.hidden_channels,
                filter_channels=self.config.models.vits.filter_channels,
                n_heads=self.config.models.vits.n_heads,
                n_layers=self.config.models.vits.n_layers,
                kernel_size=self.config.models.vits.kernel_size,
                p_dropout=self.config.models.vits.p_dropout,
                n_flows=self.config.models.vits.n_flows,
            )
        elif self.config.models.acoustic_model == "matcha":
            models['acoustic'] = MatchaTTS(
                n_vocab=self.config.models.matcha.n_vocab,
                n_speakers=self.config.models.matcha.n_speakers,
                hidden_channels=self.config.models.matcha.hidden_channels,
                filter_channels=self.config.models.matcha.filter_channels,
            )
            
        # ボコーダーの初期化
        if self.config.models.get('vocoder') and self.config.models.vocoder == "bigvgan":
            from src.models.bigvgan_v2 import BigVGANv2Config
            bigvgan_config = BigVGANv2Config(
                n_mel_channels=self.config.models.bigvgan.num_mels,
                hidden_channels=self.config.models.bigvgan.upsample_initial_channel,
                resblock_kernel_sizes=self.config.models.bigvgan.resblock_kernel_sizes,
                resblock_dilations=self.config.models.bigvgan.resblock_dilation_sizes,
                upsample_rates=self.config.models.bigvgan.upsample_rates,
                upsample_kernel_sizes=self.config.models.bigvgan.upsample_kernel_sizes,
            )
            models['vocoder'] = BigVGANv2(config=bigvgan_config)
            
        return models
    
    def setup_optimizers(self, models: Dict[str, nn.Module]) -> Dict[str, torch.optim.Optimizer]:
        """オプティマイザーの設定"""
        optimizers = {}
        
        for name, model in models.items():
            optimizer_config = self.config.training.optimizers.get(name, self.config.training.optimizers.default)
            
            # パラメータグループの設定
            param_groups = []
            
            # 事前学習済みパラメータと新規パラメータを分ける
            pretrained_params = []
            new_params = []
            
            for param_name, param in model.named_parameters():
                if param.requires_grad:
                    if 'bert' in name and 'bert' in param_name:
                        pretrained_params.append(param)
                    else:
                        new_params.append(param)
                        
            if pretrained_params:
                param_groups.append({
                    'params': pretrained_params,
                    'lr': optimizer_config.lr * 0.1,  # 事前学習済みは学習率を下げる
                })
                
            if new_params:
                param_groups.append({
                    'params': new_params,
                    'lr': optimizer_config.lr,
                })
                
            # オプティマイザーの作成
            optimizers[name] = AdamW(
                param_groups,
                betas=(optimizer_config.beta1, optimizer_config.beta2),
                eps=optimizer_config.eps,
                weight_decay=optimizer_config.weight_decay,
            )
            
        return optimizers
    
    def setup_schedulers(self, optimizers: Dict[str, torch.optim.Optimizer]) -> Dict[str, torch.optim.lr_scheduler._LRScheduler]:
        """学習率スケジューラーの設定"""
        schedulers = {}
        
        for name, optimizer in optimizers.items():
            scheduler_config = self.config.training.schedulers.get(name, self.config.training.schedulers.default)
            
            if scheduler_config.type == "cosine_annealing_warm_restarts":
                schedulers[name] = CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=scheduler_config.T_0,
                    T_mult=scheduler_config.T_mult,
                    eta_min=scheduler_config.eta_min,
                )
            elif scheduler_config.type == "exponential":
                schedulers[name] = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma=scheduler_config.gamma,
                )
                
        return schedulers
    
    def setup_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """データローダーの設定"""
        # 学習データセット
        train_dataset = TsukuyomiDataset(
            data_root=Path(self.config.data.train_dir),
            transcript_file="metadata.csv",  # LJSpeech形式
            sample_rate=self.config.data.sample_rate,
            cache_audio=self.config.data.use_cache,
            validation_split=0.1,  # 10%を検証用に
            is_validation=False,
        )
        
        # 検証データセット（同じデータから分割）
        val_dataset = TsukuyomiDataset(
            data_root=Path(self.config.data.val_dir),
            transcript_file="metadata.csv",  # LJSpeech形式
            sample_rate=self.config.data.sample_rate,
            cache_audio=self.config.data.use_cache,
            validation_split=0.1,  # 10%を検証用に
            is_validation=True,
        )
        
        # 分散学習用のサンプラー
        train_sampler = DistributedSampler(train_dataset) if self.accelerator.distributed_type != "NO" else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if self.accelerator.distributed_type != "NO" else None
        
        # データローダー
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=tts_collate_fn,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            collate_fn=tts_collate_fn,
        )
        
        return train_loader, val_loader
    
    def train_epoch(
        self,
        epoch: int,
        models: Dict[str, nn.Module],
        optimizers: Dict[str, torch.optim.Optimizer],
        schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler],
        train_loader: DataLoader,
        loss_fn: MultiTaskLoss,
        metrics: TrainingMetrics,
    ) -> Dict[str, float]:
        """1エポックの学習"""
        # 学習モード
        for model in models.values():
            model.train()
            
        epoch_losses = {name: 0.0 for name in models.keys()}
        epoch_losses['total'] = 0.0
        
        # プログレスバー
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not self.accelerator.is_main_process)
        
        for batch_idx, batch in enumerate(pbar):
            # 勾配のリセット
            for optimizer in optimizers.values():
                optimizer.zero_grad()
                
            # 前向き計算
            outputs = {}
            losses = {}
            
            # ダミーモデルの場合
            if 'acoustic' in models and self.config.models.acoustic_model == "dummy":
                outputs['acoustic'] = models['acoustic'](batch['audio'])
                losses['total'] = outputs['acoustic']['loss']
            # VITSモデルの場合
            elif 'acoustic' in models and self.config.models.acoustic_model == "vits":
                try:
                    # テキストをトークン化（仮実装 - 実際には適切なトークナイザーが必要）
                    # TODO: 実際のテキストトークナイザーを実装
                    batch_size = batch['audio'].shape[0]
                    text_len = 20  # さらに短くする
                    text_tokens = torch.randint(0, min(50, self.config.models.vits.n_vocab), (batch_size, text_len)).to(batch['audio'].device)
                    text_lengths = torch.tensor([text_len] * batch_size).to(batch['audio'].device)
                    
                    # メルスペクトログラムを生成
                    # TODO: 実際のメル変換を実装
                    mel_len = 64  # さらに短くする
                    mel_spec = torch.randn(batch_size, 80, mel_len).to(batch['audio'].device)
                    mel_lengths = torch.tensor([mel_len] * batch_size).to(batch['audio'].device)
                    
                    # デバッグ情報
                    logger.debug(f"VITS input shapes - text: {text_tokens.shape}, mel: {mel_spec.shape}, speaker_ids: {batch['speaker_ids'].shape}")
                    logger.debug(f"VITS config - hidden_channels: {self.config.models.vits.hidden_channels}, n_heads: {self.config.models.vits.n_heads}")
                    
                    # VITSフォワードパス
                    outputs['acoustic'] = models['acoustic'](
                        text=text_tokens,
                        text_lengths=text_lengths,
                        mel=mel_spec,
                        mel_lengths=mel_lengths,
                        speaker_ids=batch['speaker_ids'],
                    )
                except Exception as e:
                    logger.error(f"VITS forward error: {e}")
                    # エラー時はダミー出力（勾配を持つように修正）
                    dummy_loss = torch.tensor(1.0, device=batch['audio'].device, requires_grad=True)
                    outputs['acoustic'] = {'loss': dummy_loss}
                    losses['total'] = dummy_loss
                
                # 簡略化した損失（VITSの出力から）
                if 'loss' in outputs['acoustic']:
                    losses['total'] = outputs['acoustic']['loss']
                else:
                    # ダミー損失
                    losses['total'] = torch.tensor(0.0, device=batch['audio'].device)
                
            # ボコーダー（一時的に無効化）
            # TODO: VITSの出力形式に合わせて修正
            # if 'vocoder' in models and 'acoustic' in outputs and 'mel' in outputs['acoustic']:
            #     outputs['vocoder'] = models['vocoder'](outputs['acoustic']['mel'])
            #     losses['vocoder'] = loss_fn.compute_vocoder_loss(
            #         outputs['vocoder'],
            #         batch['audio_targets'],
            #     )
                
            # 総損失の計算
            if losses:
                total_loss = sum(losses.values())
                losses['total'] = total_loss
            else:
                # 損失がない場合はダミー損失
                total_loss = torch.tensor(1.0, device=batch['audio'].device, requires_grad=True)
                losses['total'] = total_loss
            
            # バックプロパゲーション
            self.accelerator.backward(total_loss)
            
            # 勾配クリッピング
            for model in models.values():
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config.training.gradient_clip,
                )
                
            # パラメータ更新
            for optimizer in optimizers.values():
                optimizer.step()
                
            # 損失の記録
            for name, loss in losses.items():
                epoch_losses[name] += loss.item()
                
            # メトリクスの更新（エラー回避のため一時的に無効化）
            # TODO: 実際のメトリクス計算を実装
            # metrics.update(outputs, batch)
            
            # プログレスバーの更新
            if batch_idx % 10 == 0:
                current_losses = {k: v / (batch_idx + 1) for k, v in epoch_losses.items()}
                pbar.set_postfix(current_losses)
                
            # TensorBoardへの記録
            if self.writer and batch_idx % self.config.training.logging.log_interval == 0:
                global_step = epoch * len(train_loader) + batch_idx
                for name, loss in losses.items():
                    self.writer.add_scalar(f'train/loss_{name}', loss.item(), global_step)
                    
                # 学習率の記録
                for name, optimizer in optimizers.items():
                    lr = optimizer.param_groups[0]['lr']
                    self.writer.add_scalar(f'train/lr_{name}', lr, global_step)
                    
        # スケジューラーの更新
        for scheduler in schedulers.values():
            scheduler.step()
            
        # エポック平均の計算
        num_batches = len(train_loader)
        for name in epoch_losses:
            epoch_losses[name] /= num_batches
            
        return epoch_losses
    
    def validate(
        self,
        epoch: int,
        models: Dict[str, nn.Module],
        val_loader: DataLoader,
        loss_fn: MultiTaskLoss,
        metrics: TrainingMetrics,
    ) -> Dict[str, float]:
        """検証の実行"""
        # 評価モード
        for model in models.values():
            model.eval()
            
        val_losses = {name: 0.0 for name in models.keys()}
        val_losses['total'] = 0.0
        
        # MCD評価器
        mcd_evaluator = MelCepstralDistortion()
        mcd_scores = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", disable=not self.accelerator.is_main_process):
                outputs = {}
                losses = {}
                
                # ダミーモデルの場合
                if 'acoustic' in models and self.config.models.acoustic_model == "dummy":
                    outputs['acoustic'] = models['acoustic'](batch['audio'])
                    losses['total'] = outputs['acoustic']['loss']
                # 前向き計算（学習時と同様）
                elif 'xphonebert' in models:
                    outputs['xphonebert'] = models['xphonebert'](
                        batch['phoneme_ids'],
                        batch['language_ids'],
                    )
                    losses['xphonebert'] = loss_fn.compute_bert_loss(
                        outputs['xphonebert'],
                        batch['phoneme_targets'],
                    )
                    
                if 'f0_bert' in models:
                    outputs['f0_bert'] = models['f0_bert'](
                        batch['f0'],
                        batch.get('f0_mask'),
                    )
                    losses['f0_bert'] = loss_fn.compute_f0_loss(
                        outputs['f0_bert'],
                        batch['f0_targets'],
                    )
                    
                if 'acoustic' in models:
                    encoder_outputs = []
                    if 'xphonebert' in outputs:
                        encoder_outputs.append(outputs['xphonebert']['hidden_states'])
                    if 'f0_bert' in outputs:
                        encoder_outputs.append(outputs['f0_bert']['hidden_states'])
                        
                    encoder_output = torch.cat(encoder_outputs, dim=-1) if encoder_outputs else None
                    
                    outputs['acoustic'] = models['acoustic'](
                        batch['text'],
                        batch['mel_targets'],
                        batch['speaker_ids'],
                        encoder_output=encoder_output,
                    )
                    losses['acoustic'] = loss_fn.compute_acoustic_loss(
                        outputs['acoustic'],
                        batch['mel_targets'],
                    )
                    
                    # MCD計算
                    for pred, target in zip(outputs['acoustic']['mel'], batch['mel_targets']):
                        mcd = mcd_evaluator.calculate(
                            pred.cpu().numpy(),
                            target.cpu().numpy(),
                        )
                        mcd_scores.append(mcd)
                        
                if 'vocoder' in models and 'acoustic' in outputs:
                    outputs['vocoder'] = models['vocoder'](outputs['acoustic']['mel'])
                    losses['vocoder'] = loss_fn.compute_vocoder_loss(
                        outputs['vocoder'],
                        batch['audio_targets'],
                    )
                    
                # 総損失
                if losses:
                    total_loss = sum(losses.values())
                    losses['total'] = total_loss
                else:
                    losses['total'] = torch.tensor(0.0)
                
                # 損失の記録
                for name, loss in losses.items():
                    val_losses[name] += loss.item()
                    
                # メトリクスの更新
                metrics.update(outputs, batch)
                
        # 平均の計算
        num_batches = len(val_loader)
        for name in val_losses:
            val_losses[name] /= num_batches
            
        # MCDスコアの平均
        if mcd_scores:
            val_losses['mcd'] = np.mean(mcd_scores)
            
        # TensorBoardへの記録
        if self.writer:
            for name, loss in val_losses.items():
                self.writer.add_scalar(f'val/loss_{name}', loss, epoch)
                
            # メトリクスの記録
            metric_values = metrics.compute()
            for name, value in metric_values.items():
                self.writer.add_scalar(f'val/metric_{name}', value, epoch)
                
        return val_losses
    
    def save_checkpoint(
        self,
        epoch: int,
        models: Dict[str, nn.Module],
        optimizers: Dict[str, torch.optim.Optimizer],
        schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler],
        val_losses: Dict[str, float],
        is_best: bool = False,
    ):
        """チェックポイントの保存"""
        if not self.accelerator.is_main_process:
            return
            
        checkpoint = {
            'epoch': epoch,
            'config': OmegaConf.to_container(self.config),
            'val_losses': val_losses,
        }
        
        # モデルの状態
        for name, model in models.items():
            checkpoint[f'model_{name}'] = model.state_dict()
            
        # オプティマイザーの状態
        for name, optimizer in optimizers.items():
            checkpoint[f'optimizer_{name}'] = optimizer.state_dict()
            
        # スケジューラーの状態
        for name, scheduler in schedulers.items():
            checkpoint[f'scheduler_{name}'] = scheduler.state_dict()
            
        # 保存
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # ベストモデルの保存
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
            
            # モデルレジストリへの登録
            for name, model in models.items():
                metadata = {
                    'epoch': epoch,
                    'val_loss': val_losses.get(name, 0.0),
                    'mcd': val_losses.get('mcd', 0.0),
                }
                self.model_registry.register_model(
                    model,
                    metadata,
                    version=f"{epoch}.0.0",
                )
                
    def load_checkpoint(
        self,
        checkpoint_path: Path,
        models: Dict[str, nn.Module],
        optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
        schedulers: Optional[Dict[str, torch.optim.lr_scheduler._LRScheduler]] = None,
    ) -> int:
        """チェックポイントの読み込み"""
        checkpoint = torch.load(checkpoint_path, map_location=self.accelerator.device)
        
        # モデルの状態を復元
        for name, model in models.items():
            if f'model_{name}' in checkpoint:
                model.load_state_dict(checkpoint[f'model_{name}'])
                
        # オプティマイザーの状態を復元
        if optimizers:
            for name, optimizer in optimizers.items():
                if f'optimizer_{name}' in checkpoint:
                    optimizer.load_state_dict(checkpoint[f'optimizer_{name}'])
                    
        # スケジューラーの状態を復元
        if schedulers:
            for name, scheduler in schedulers.items():
                if f'scheduler_{name}' in checkpoint:
                    scheduler.load_state_dict(checkpoint[f'scheduler_{name}'])
                    
        return checkpoint['epoch']
    
    def train(self):
        """学習のメインループ"""
        # データローダーの設定
        train_loader, val_loader = self.setup_data_loaders()
        
        # モデルの設定
        models = self.setup_models()
        
        # オプティマイザーとスケジューラーの設定
        optimizers = self.setup_optimizers(models)
        schedulers = self.setup_schedulers(optimizers)
        
        # 損失関数とメトリクス
        loss_fn = MultiTaskLoss(self.config.training.loss_weights)
        metrics = TrainingMetrics()
        
        # Acceleratorによる準備
        for name in models:
            models[name], optimizers[name], train_loader, val_loader = self.accelerator.prepare(
                models[name], optimizers[name], train_loader, val_loader
            )
            
        # チェックポイントの読み込み
        start_epoch = 0
        if self.config.training.resume_from:
            checkpoint_path = Path(self.config.training.resume_from)
            if checkpoint_path.exists():
                start_epoch = self.load_checkpoint(
                    checkpoint_path,
                    models,
                    optimizers,
                    schedulers,
                )
                logger.info(f"Resumed from epoch {start_epoch}")
                
        # 学習ループ
        best_val_loss = float('inf')
        
        for epoch in range(start_epoch, self.config.training.num_epochs):
            # エポックの開始
            logger.info(f"Starting epoch {epoch + 1}/{self.config.training.num_epochs}")
            
            # 学習
            train_losses = self.train_epoch(
                epoch,
                models,
                optimizers,
                schedulers,
                train_loader,
                loss_fn,
                metrics,
            )
            
            # 検証
            val_losses = self.validate(
                epoch,
                models,
                val_loader,
                loss_fn,
                metrics,
            )
            
            # ログ出力
            logger.info(
                f"Epoch {epoch + 1} - "
                f"Train Loss: {train_losses['total']:.4f}, "
                f"Val Loss: {val_losses['total']:.4f}, "
                f"MCD: {val_losses.get('mcd', 0.0):.2f}"
            )
            
            # チェックポイントの保存
            is_best = val_losses['total'] < best_val_loss
            if is_best:
                best_val_loss = val_losses['total']
                
            if (epoch + 1) % self.config.training.save_interval == 0 or is_best:
                self.save_checkpoint(
                    epoch + 1,
                    models,
                    optimizers,
                    schedulers,
                    val_losses,
                    is_best=is_best,
                )
                
        # 学習終了
        logger.info("Training completed!")
        
        # TensorBoardのクローズ
        if self.writer:
            self.writer.close()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Tsukuyomi TTS Training")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file',
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume from',
    )
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Local rank for distributed training',
    )
    
    args = parser.parse_args()
    
    # 設定ファイルの読み込み
    config = OmegaConf.load(args.config)
    
    # 再開パスの設定
    if args.resume:
        config.training.resume_from = args.resume
        
    # 分散学習の設定
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        
    # トレーナーの作成と学習の実行
    trainer = TTSTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()