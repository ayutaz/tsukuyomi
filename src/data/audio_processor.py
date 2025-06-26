"""
Audio processing utilities for TTS
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

logger = logging.getLogger(__name__)


class AudioProcessor:
    """音声処理ユーティリティ
    
    音声データの前処理、メルスペクトログラム変換、正規化などを行う
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        power: float = 1.0,
        norm: str = "slaney",
        mel_scale: str = "htk",
        ref_level_db: float = 20.0,
        min_level_db: float = -100.0,
        symmetric_norm: bool = True,
        max_abs_value: float = 4.0,
        clip_norm: bool = True,
    ):
        """
        Args:
            sample_rate: サンプリングレート
            n_fft: FFTサイズ
            win_length: 窓長
            hop_length: ホップ長
            n_mels: メルフィルタバンクの数
            f_min: 最小周波数
            f_max: 最大周波数（Noneの場合はsample_rate/2）
            center: 中央パディングを使用するか
            pad_mode: パディングモード
            power: スペクトログラムのパワー（1.0: magnitude, 2.0: power）
            norm: メルスケールの正規化方法
            mel_scale: メルスケールの種類
            ref_level_db: リファレンスレベル（dB）
            min_level_db: 最小レベル（dB）
            symmetric_norm: 対称正規化を使用するか
            max_abs_value: 正規化後の最大絶対値
            clip_norm: クリッピング正規化を使用するか
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2
        self.center = center
        self.pad_mode = pad_mode
        self.power = power
        self.ref_level_db = ref_level_db
        self.min_level_db = min_level_db
        self.symmetric_norm = symmetric_norm
        self.max_abs_value = max_abs_value
        self.clip_norm = clip_norm
        
        # メルスペクトログラム変換
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=self.f_max,
            n_mels=n_mels,
            center=center,
            pad_mode=pad_mode,
            power=power,
            norm=norm,
            mel_scale=mel_scale,
        )
        
        logger.info(f"Initialized AudioProcessor: sr={sample_rate}, "
                   f"n_mels={n_mels}, hop_length={hop_length}")
        
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """音声ファイルを読み込み
        
        Args:
            audio_path: 音声ファイルのパス
            
        Returns:
            音声波形 [1, T] or [T]
        """
        waveform, sr = torchaudio.load(audio_path)
        
        # リサンプリング
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # モノラルに変換
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        return waveform
    
    def wav_to_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """音声波形をメルスペクトログラムに変換
        
        Args:
            wav: 音声波形 [B, T] or [T]
            
        Returns:
            メルスペクトログラム [B, n_mels, T_mel] or [n_mels, T_mel]
        """
        # 入力の次元を調整
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        # メルスペクトログラムに変換
        mel = self.mel_transform(wav)
        
        # dBスケールに変換
        mel = self.amplitude_to_db(mel)
        
        # 正規化
        mel = self.normalize(mel)
        
        if squeeze_output:
            mel = mel.squeeze(0)
            
        return mel
    
    def amplitude_to_db(self, mel: torch.Tensor) -> torch.Tensor:
        """振幅スペクトログラムをdBスケールに変換"""
        min_value = 1e-5
        return 20 * torch.log10(torch.clamp(mel, min=min_value))
    
    def db_to_amplitude(self, mel_db: torch.Tensor) -> torch.Tensor:
        """dBスケールを振幅スペクトログラムに変換"""
        return torch.pow(10.0, mel_db / 20.0)
    
    def normalize(self, mel_db: torch.Tensor) -> torch.Tensor:
        """メルスペクトログラムを正規化"""
        # リファレンスレベルで正規化
        mel_norm = mel_db - self.ref_level_db
        
        # 最小値でクリップ
        mel_norm = torch.clamp(mel_norm, min=self.min_level_db)
        
        if self.symmetric_norm:
            # [-1, 1]の範囲に正規化
            mel_norm = 2 * self.max_abs_value * (
                (mel_norm - self.min_level_db) / (-self.min_level_db)
            ) - self.max_abs_value
            
            if self.clip_norm:
                mel_norm = torch.clamp(
                    mel_norm, min=-self.max_abs_value, max=self.max_abs_value
                )
        else:
            # [0, 1]の範囲に正規化
            mel_norm = (mel_norm - self.min_level_db) / (-self.min_level_db)
            
            if self.clip_norm:
                mel_norm = torch.clamp(mel_norm, min=0.0, max=1.0)
                
        return mel_norm
    
    def denormalize(self, mel_norm: torch.Tensor) -> torch.Tensor:
        """正規化されたメルスペクトログラムを元に戻す"""
        if self.symmetric_norm:
            mel_db = (mel_norm + self.max_abs_value) / (2 * self.max_abs_value)
            mel_db = mel_db * (-self.min_level_db) + self.min_level_db
        else:
            mel_db = mel_norm * (-self.min_level_db) + self.min_level_db
            
        mel_db = mel_db + self.ref_level_db
        return mel_db
    
    def mel_to_wav(self, mel: torch.Tensor, vocoder: Optional[torch.nn.Module] = None) -> torch.Tensor:
        """メルスペクトログラムを音声波形に変換
        
        Args:
            mel: メルスペクトログラム [B, n_mels, T_mel] or [n_mels, T_mel]
            vocoder: ボコーダーモデル（Noneの場合はGriffin-Lim）
            
        Returns:
            音声波形 [B, T] or [T]
        """
        if vocoder is not None:
            # ニューラルボコーダーを使用
            return vocoder(mel)
        else:
            # Griffin-Limアルゴリズムを使用（簡易実装）
            # 注意: 実際のGriffin-Lim実装にはlibrosaが必要
            logger.warning("Griffin-Lim not implemented. Returning zeros.")
            if mel.dim() == 3:
                batch_size, _, mel_len = mel.shape
                wav_len = mel_len * self.hop_length
                return torch.zeros(batch_size, wav_len)
            else:
                mel_len = mel.shape[-1]
                wav_len = mel_len * self.hop_length
                return torch.zeros(wav_len)
    
    def get_mel_lengths(self, wav_lengths: torch.Tensor) -> torch.Tensor:
        """音声長からメルスペクトログラム長を計算"""
        return torch.div(wav_lengths - self.win_length, self.hop_length, rounding_mode='floor') + 1
