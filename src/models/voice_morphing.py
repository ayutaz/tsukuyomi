"""音声モーフィングモジュール

複数の話者やスタイルを滑らかに混合・変換
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal


class FeatureInterpolator(nn.Module):
    """特徴補間モジュール

    異なる話者の特徴を滑らかに補間
    """

    def __init__(
        self,
        feature_dim: int = 256,
        interpolation_type: str = "spherical",  # linear, spherical, learned
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.interpolation_type = interpolation_type

        if interpolation_type == "learned":
            # 学習可能な補間
            self.interpolation_net = nn.Sequential()
            for i in range(num_layers):
                self.interpolation_net.add_module(
                    f"layer_{i}",
                    nn.Sequential(
                        nn.Linear(
                            feature_dim * 2 + 1 if i == 0 else hidden_dim, hidden_dim
                        ),
                        nn.LayerNorm(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ),
                )
            self.interpolation_net.add_module(
                "output", nn.Linear(hidden_dim, feature_dim)
            )

    def forward(
        self,
        features: List[torch.Tensor],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: 特徴のリスト [(batch_size, feature_dim), ...]
            weights: 補間重み (batch_size, num_features)

        Returns:
            interpolated: 補間された特徴 (batch_size, feature_dim)
        """
        if len(features) == 1:
            return features[0]

        if self.interpolation_type == "linear":
            # 線形補間
            stacked = torch.stack(features, dim=1)  # (batch, num_features, dim)
            weights_expanded = weights.unsqueeze(-1)  # (batch, num_features, 1)
            interpolated = (stacked * weights_expanded).sum(dim=1)

        elif self.interpolation_type == "spherical":
            # 球面線形補間（SLERP）
            if len(features) == 2:
                interpolated = self._slerp(features[0], features[1], weights[:, 1:2])
            else:
                # 多点の場合は段階的にSLERP
                interpolated = features[0]
                cumulative_weight = weights[:, 0:1]

                for i in range(1, len(features)):
                    current_weight = weights[:, i : i + 1]
                    total_weight = cumulative_weight + current_weight
                    alpha = current_weight / (total_weight + 1e-8)
                    interpolated = self._slerp(interpolated, features[i], alpha)
                    cumulative_weight = total_weight

        elif self.interpolation_type == "learned":
            # 学習された補間
            if len(features) == 2:
                concat_features = torch.cat(
                    [features[0], features[1], weights[:, 1:2]], dim=-1
                )
                interpolated = self.interpolation_net(concat_features)
            else:
                # 多点の場合はペアワイズ
                interpolated = features[0]
                for i in range(1, len(features)):
                    alpha = weights[:, i : i + 1] / weights[:, : i + 1].sum(
                        dim=1, keepdim=True
                    )
                    concat_features = torch.cat(
                        [interpolated, features[i], alpha], dim=-1
                    )
                    interpolated = self.interpolation_net(concat_features)

        return interpolated

    def _slerp(
        self,
        v1: torch.Tensor,
        v2: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """球面線形補間"""
        # 正規化
        v1_norm = F.normalize(v1, p=2, dim=-1)
        v2_norm = F.normalize(v2, p=2, dim=-1)

        # 内積
        dot = (v1_norm * v2_norm).sum(dim=-1, keepdim=True)
        dot = torch.clamp(dot, -1.0, 1.0)

        # 角度
        theta = torch.acos(dot)

        # SLERP
        sin_theta = torch.sin(theta)

        # 角度が小さい場合は線形補間
        small_angle = sin_theta.abs() < 1e-5

        w1 = torch.where(small_angle, 1 - t, torch.sin((1 - t) * theta) / sin_theta)
        w2 = torch.where(small_angle, t, torch.sin(t * theta) / sin_theta)

        return w1 * v1 + w2 * v2


class VoiceMorphingEncoder(nn.Module):
    """音声モーフィング用エンコーダー

    話者非依存の内容表現と話者表現を分離
    """

    def __init__(
        self,
        input_dim: int = 80,  # メルスペクトログラム
        content_dim: int = 256,
        speaker_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 共有エンコーダー
        self.shared_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers // 2,
        )

        # 入力投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # コンテンツエンコーダー
        self.content_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers // 2,
        )

        # 話者エンコーダー
        self.speaker_encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, speaker_dim, kernel_size=1),
        )

        # 出力投影
        self.content_proj = nn.Linear(hidden_dim, content_dim)
        self.speaker_pool = nn.AdaptiveAvgPool1d(1)

        # 情報ボトルネック（話者情報を除去）
        self.content_bottleneck = nn.Sequential(
            nn.Linear(content_dim, content_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(content_dim // 2, content_dim),
        )

    def forward(
        self,
        mel: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            mel: メルスペクトログラム (batch_size, time, n_mels)
            mask: マスク (batch_size, time)

        Returns:
            outputs: コンテンツと話者表現
        """
        # 入力投影
        x = self.input_proj(mel)

        # 共有エンコーディング
        shared = self.shared_encoder(x, src_key_padding_mask=mask)

        # コンテンツ抽出
        content = self.content_encoder(shared, src_key_padding_mask=mask)
        content = self.content_proj(content)
        content = self.content_bottleneck(content)

        # 話者抽出
        speaker_input = shared.transpose(1, 2)  # (batch, hidden, time)
        speaker = self.speaker_encoder(speaker_input)
        speaker = self.speaker_pool(speaker).squeeze(-1)

        return {
            "content": content,
            "speaker": speaker,
            "shared": shared,
        }


class MorphingDecoder(nn.Module):
    """モーフィングデコーダー

    コンテンツと話者表現から音声を再構成
    """

    def __init__(
        self,
        content_dim: int = 256,
        speaker_dim: int = 256,
        output_dim: int = 80,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 話者埋め込みの投影
        self.speaker_proj = nn.Sequential(
            nn.Linear(speaker_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # コンテンツと話者の結合
        self.combine = nn.Sequential(
            nn.Linear(content_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # デコーダー
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # 出力投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        content: torch.Tensor,
        speaker: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            content: コンテンツ表現 (batch_size, time, content_dim)
            speaker: 話者表現 (batch_size, speaker_dim)
            mask: マスク

        Returns:
            output: 再構成されたメルスペクトログラム
        """
        batch_size, seq_len = content.shape[:2]

        # 話者埋め込みを時間軸に拡張
        speaker_proj = self.speaker_proj(speaker)
        speaker_expanded = speaker_proj.unsqueeze(1).expand(-1, seq_len, -1)

        # 結合
        combined = torch.cat([content, speaker_expanded], dim=-1)
        combined = self.combine(combined)

        # デコード
        decoded = self.decoder(
            combined,
            combined,
            tgt_key_padding_mask=mask,
            memory_key_padding_mask=mask,
        )

        # 出力
        output = self.output_proj(decoded)

        return output


class VoiceMorphing(nn.Module):
    """統合音声モーフィングモジュール"""

    def __init__(
        self,
        input_dim: int = 80,
        content_dim: int = 256,
        speaker_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        interpolation_type: str = "spherical",
        enable_pitch_shift: bool = True,
        enable_time_stretch: bool = True,
        **kwargs,
    ):
        super().__init__()

        # エンコーダー
        self.encoder = VoiceMorphingEncoder(
            input_dim=input_dim,
            content_dim=content_dim,
            speaker_dim=speaker_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **kwargs,
        )

        # デコーダー
        self.decoder = MorphingDecoder(
            content_dim=content_dim,
            speaker_dim=speaker_dim,
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **kwargs,
        )

        # 特徴補間
        self.feature_interpolator = FeatureInterpolator(
            feature_dim=speaker_dim,
            interpolation_type=interpolation_type,
            **kwargs,
        )

        # ピッチシフト（オプション）
        self.enable_pitch_shift = enable_pitch_shift
        if enable_pitch_shift:
            self.pitch_shifter = PitchShifter()

        # タイムストレッチ（オプション）
        self.enable_time_stretch = enable_time_stretch
        if enable_time_stretch:
            self.time_stretcher = TimeStretcher()

    def extract_features(
        self,
        mels: List[torch.Tensor],
        masks: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """複数の音声から特徴を抽出"""
        contents = []
        speakers = []

        for i, mel in enumerate(mels):
            mask = masks[i] if masks else None
            features = self.encoder(mel, mask)
            contents.append(features["content"])
            speakers.append(features["speaker"])

        return contents, speakers

    def morph(
        self,
        source_mels: List[torch.Tensor],
        target_speakers: Optional[List[torch.Tensor]] = None,
        weights: Optional[torch.Tensor] = None,
        pitch_shift: Optional[float] = None,
        time_stretch: Optional[float] = None,
        masks: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """音声モーフィング

        Args:
            source_mels: ソース音声のメルスペクトログラムリスト
            target_speakers: ターゲット話者埋め込み（Noneの場合はソースから抽出）
            weights: 補間重み
            pitch_shift: ピッチシフト量（半音単位）
            time_stretch: 時間伸縮率
            masks: マスクのリスト

        Returns:
            morphed_mel: モーフィングされたメルスペクトログラム
        """
        # 特徴抽出
        contents, speakers = self.extract_features(source_mels, masks)

        # デフォルトの重み
        if weights is None:
            num_sources = len(source_mels)
            weights = torch.ones(1, num_sources) / num_sources
            weights = weights.to(source_mels[0].device)

        # コンテンツの選択または混合
        if len(contents) == 1:
            content = contents[0]
        else:
            # 重み付き平均（コンテンツは通常混合しない）
            content = contents[0]  # 最初のソースのコンテンツを使用

        # 話者の補間
        if target_speakers is None:
            target_speakers = speakers

        interpolated_speaker = self.feature_interpolator(target_speakers, weights)

        # デコード
        morphed_mel = self.decoder(
            content, interpolated_speaker, masks[0] if masks else None
        )

        # ピッチシフト
        if pitch_shift is not None and self.enable_pitch_shift:
            morphed_mel = self.pitch_shifter(morphed_mel, pitch_shift)

        # タイムストレッチ
        if time_stretch is not None and self.enable_time_stretch:
            morphed_mel = self.time_stretcher(morphed_mel, time_stretch)

        return morphed_mel

    def interpolate_speakers(
        self,
        speakers: List[torch.Tensor],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """話者埋め込みの補間"""
        return self.feature_interpolator(speakers, weights)

    def voice_conversion(
        self,
        source_mel: torch.Tensor,
        target_mel: torch.Tensor,
        content_from_source: bool = True,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """声質変換

        Args:
            source_mel: ソース音声
            target_mel: ターゲット音声
            content_from_source: コンテンツをソースから取るか
            mask: マスク

        Returns:
            converted_mel: 変換された音声
        """
        # 特徴抽出
        source_features = self.encoder(source_mel, mask)
        target_features = self.encoder(target_mel, mask)

        # コンテンツと話者の組み合わせ
        if content_from_source:
            content = source_features["content"]
            speaker = target_features["speaker"]
        else:
            content = target_features["content"]
            speaker = source_features["speaker"]

        # デコード
        converted_mel = self.decoder(content, speaker, mask)

        return converted_mel


class PitchShifter(nn.Module):
    """ピッチシフトモジュール"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        mel: torch.Tensor,
        shift_semitones: float,
    ) -> torch.Tensor:
        """
        Args:
            mel: メルスペクトログラム (batch, time, n_mels)
            shift_semitones: シフト量（半音単位）

        Returns:
            shifted_mel: ピッチシフトされたメル
        """
        # 周波数軸でのシフト
        batch_size, time_steps, n_mels = mel.shape

        # シフト量の計算
        shift_factor = 2 ** (shift_semitones / 12)

        # 周波数ビンのリマッピング
        freq_bins = torch.arange(n_mels, device=mel.device, dtype=torch.float)
        shifted_bins = freq_bins / shift_factor

        # 線形補間
        shifted_mel = torch.zeros_like(mel)
        for b in range(batch_size):
            for t in range(time_steps):
                shifted_mel[b, t] = self._interp1d(freq_bins, mel[b, t], shifted_bins)

        return shifted_mel

    def _interp1d(self, x, y, xi):
        """1次元線形補間"""
        # PyTorchでの簡易実装
        indices = torch.searchsorted(x, xi)
        indices = torch.clamp(indices, 1, len(x) - 1)

        x0 = x[indices - 1]
        x1 = x[indices]
        y0 = y[indices - 1]
        y1 = y[indices]

        alpha = (xi - x0) / (x1 - x0)
        yi = y0 * (1 - alpha) + y1 * alpha

        return yi


class TimeStretcher(nn.Module):
    """タイムストレッチモジュール"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        mel: torch.Tensor,
        stretch_factor: float,
    ) -> torch.Tensor:
        """
        Args:
            mel: メルスペクトログラム (batch, time, n_mels)
            stretch_factor: 伸縮率（>1で伸張、<1で圧縮）

        Returns:
            stretched_mel: 時間伸縮されたメル
        """
        batch_size, time_steps, n_mels = mel.shape

        # 新しい時間長
        new_time_steps = int(time_steps * stretch_factor)

        # 時間軸での補間
        mel_transposed = mel.transpose(1, 2)  # (batch, n_mels, time)
        stretched_mel = F.interpolate(
            mel_transposed,
            size=new_time_steps,
            mode="linear",
            align_corners=False,
        )
        stretched_mel = stretched_mel.transpose(1, 2)  # (batch, time, n_mels)

        return stretched_mel


class ContinuousMorphing(nn.Module):
    """連続的な音声モーフィング

    時間軸に沿って滑らかに話者を変化させる
    """

    def __init__(
        self,
        morphing_module: VoiceMorphing,
        window_size: int = 32,
        hop_size: int = 16,
    ):
        super().__init__()
        self.morphing_module = morphing_module
        self.window_size = window_size
        self.hop_size = hop_size

    def forward(
        self,
        source_mel: torch.Tensor,
        target_speakers: List[torch.Tensor],
        transition_points: List[int],
        transition_lengths: List[int],
    ) -> torch.Tensor:
        """
        Args:
            source_mel: ソース音声 (batch, time, n_mels)
            target_speakers: 各セクションのターゲット話者
            transition_points: 遷移開始位置
            transition_lengths: 遷移の長さ

        Returns:
            morphed_mel: 連続的にモーフィングされた音声
        """
        batch_size, time_steps, n_mels = source_mel.shape
        morphed_mel = torch.zeros_like(source_mel)

        # ウィンドウごとに処理
        for start in range(0, time_steps - self.window_size, self.hop_size):
            end = start + self.window_size
            window = source_mel[:, start:end, :]

            # 現在の位置での話者を決定
            current_speaker_idx = 0
            for i, point in enumerate(transition_points):
                if start >= point:
                    current_speaker_idx = i + 1

            # 遷移中かチェック
            in_transition = False
            transition_alpha = 0.0

            for i, (point, length) in enumerate(
                zip(transition_points, transition_lengths)
            ):
                if point <= start < point + length:
                    in_transition = True
                    transition_alpha = (start - point) / length
                    current_speaker_idx = i
                    break

            # モーフィング
            if in_transition and current_speaker_idx < len(target_speakers) - 1:
                # 2話者間の補間
                weights = torch.tensor(
                    [1 - transition_alpha, transition_alpha]
                ).unsqueeze(0)
                speakers = [
                    target_speakers[current_speaker_idx],
                    target_speakers[current_speaker_idx + 1],
                ]
                morphed_window = self.morphing_module.morph([window], speakers, weights)
            else:
                # 単一話者
                morphed_window = self.morphing_module.morph(
                    [window],
                    [
                        target_speakers[
                            min(current_speaker_idx, len(target_speakers) - 1)
                        ]
                    ],
                )

            # オーバーラップ加算
            if start == 0:
                morphed_mel[:, start:end, :] = morphed_window
            else:
                overlap = self.hop_size
                fade_in = torch.linspace(0, 1, overlap, device=source_mel.device)
                fade_out = 1 - fade_in

                morphed_mel[:, start : start + overlap, :] = morphed_mel[
                    :, start : start + overlap, :
                ] * fade_out.unsqueeze(-1) + morphed_window[
                    :, :overlap, :
                ] * fade_in.unsqueeze(
                    -1
                )
                morphed_mel[:, start + overlap : end, :] = morphed_window[
                    :, overlap:, :
                ]

        return morphed_mel
