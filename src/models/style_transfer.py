"""スタイル転送モジュール

話者の声質やスタイルを転送・混合するためのコンポーネント
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class StyleEncoder(nn.Module):
    """スタイルエンコーダー
    
    音声からスタイル埋め込みを抽出
    """
    
    def __init__(
        self,
        input_dim: int = 80,  # メルスペクトログラムの次元
        style_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.1,
        use_instance_norm: bool = True,
    ):
        super().__init__()
        self.style_dim = style_dim
        
        # 畳み込みエンコーダー
        self.conv_layers = nn.ModuleList()
        in_channels = input_dim
        
        for i in range(num_layers):
            out_channels = hidden_dim // (2 ** (num_layers - i - 1))
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                    nn.InstanceNorm1d(out_channels) if use_instance_norm else nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            in_channels = out_channels
            
        # Temporal Average Pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # スタイル埋め込み層
        self.style_embedding = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, style_dim * 2),  # mean + std
        )
        
        # スタイルトークン（学習可能な参照スタイル）
        self.num_style_tokens = 10
        self.style_tokens = nn.Parameter(torch.randn(self.num_style_tokens, style_dim))
        
    def forward(
        self,
        mel: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            mel: メルスペクトログラム (batch_size, n_mels, time)
            mask: マスク (batch_size, time)
            
        Returns:
            style_outputs: スタイル情報
        """
        # 畳み込み処理
        x = mel
        for conv in self.conv_layers:
            x = conv(x)
            
        # マスク適用
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).float()
            x = x * mask_expanded
            
        # 時間方向のプーリング
        x = self.temporal_pool(x).squeeze(-1)
        
        # スタイル埋め込み
        style_params = self.style_embedding(x)
        style_mean, style_logstd = style_params.chunk(2, dim=-1)
        style_std = F.softplus(style_logstd) + 1e-5
        
        # サンプリング（再パラメータ化トリック）
        eps = torch.randn_like(style_mean)
        style_embedding = style_mean + eps * style_std
        
        return {
            'style_embedding': style_embedding,
            'style_mean': style_mean,
            'style_std': style_std,
            'style_tokens': self.style_tokens,
        }
        
    def get_style_token(self, token_id: int) -> torch.Tensor:
        """学習済みスタイルトークンを取得"""
        return self.style_tokens[token_id].unsqueeze(0)


class StyleAdapter(nn.Module):
    """スタイルアダプター
    
    Feature-wise Linear Modulation (FiLM) を使用したスタイル適応
    """
    
    def __init__(
        self,
        feature_dim: int,
        style_dim: int,
        num_layers: int = 4,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or feature_dim
        
        # スタイル変換ネットワーク
        self.style_transforms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.style_transforms.append(
                nn.Sequential(
                    nn.Linear(style_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, feature_dim * 2),  # scale + shift
                )
            )
            
        # レイヤー正規化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(feature_dim) for _ in range(num_layers)
        ])
        
    def forward(
        self,
        features: torch.Tensor,
        style_embedding: torch.Tensor,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            features: 入力特徴 (batch_size, seq_len, feature_dim)
            style_embedding: スタイル埋め込み (batch_size, style_dim)
            layer_idx: 適用するレイヤーのインデックス
            
        Returns:
            adapted_features: スタイル適応後の特徴
        """
        # スタイルパラメータの計算
        style_params = self.style_transforms[layer_idx](style_embedding)
        scale, shift = style_params.chunk(2, dim=-1)
        
        # FiLM変調
        scale = scale.unsqueeze(1)  # (batch_size, 1, feature_dim)
        shift = shift.unsqueeze(1)
        
        # アフィン変換
        features = self.layer_norms[layer_idx](features)
        adapted_features = features * (1 + scale) + shift
        
        return adapted_features


class StyleMixer(nn.Module):
    """スタイルミキサー
    
    複数のスタイルを混合
    """
    
    def __init__(
        self,
        style_dim: int = 256,
        num_mix_layers: int = 2,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # アテンションベースの混合
        self.mix_attention = nn.MultiheadAttention(
            embed_dim=style_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        
        # 混合重み予測
        self.weight_predictor = nn.Sequential(
            nn.Linear(style_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # 非線形混合
        self.mix_transform = nn.Sequential(
            nn.Linear(style_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, style_dim),
        )
        
    def forward(
        self,
        style_embeddings: List[torch.Tensor],
        mix_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            style_embeddings: スタイル埋め込みのリスト
            mix_weights: 混合重み (num_styles,)
            
        Returns:
            mixed_style: 混合されたスタイル埋め込み
        """
        if len(style_embeddings) == 1:
            return style_embeddings[0]
            
        # スタック
        styles = torch.stack(style_embeddings, dim=1)  # (batch_size, num_styles, style_dim)
        batch_size, num_styles, style_dim = styles.shape
        
        if mix_weights is None:
            # アテンションベースの重み計算
            mixed, attention_weights = self.mix_attention(
                styles.mean(dim=1, keepdim=True),  # クエリ
                styles,  # キー
                styles,  # バリュー
            )
            mixed = mixed.squeeze(1)
        else:
            # 指定された重みで混合
            mix_weights = mix_weights.view(1, num_styles, 1)
            mixed = (styles * mix_weights).sum(dim=1)
            
        # 非線形変換
        if num_styles == 2:
            # 2つのスタイルの場合は特別な処理
            concat_styles = torch.cat([styles[:, 0], styles[:, 1]], dim=-1)
            weight = self.weight_predictor(concat_styles)
            mixed = styles[:, 0] * weight + styles[:, 1] * (1 - weight)
            mixed = self.mix_transform(concat_styles) + mixed
            
        return mixed


class StyleTransferModule(nn.Module):
    """統合スタイル転送モジュール"""
    
    def __init__(
        self,
        feature_dim: int = 768,
        style_dim: int = 256,
        num_adapter_layers: int = 4,
        enable_mixing: bool = True,
        **kwargs,
    ):
        super().__init__()
        
        # スタイルエンコーダー
        self.style_encoder = StyleEncoder(
            style_dim=style_dim,
            **kwargs,
        )
        
        # スタイルアダプター
        self.style_adapter = StyleAdapter(
            feature_dim=feature_dim,
            style_dim=style_dim,
            num_layers=num_adapter_layers,
            **kwargs,
        )
        
        # スタイルミキサー
        self.enable_mixing = enable_mixing
        if enable_mixing:
            self.style_mixer = StyleMixer(
                style_dim=style_dim,
                **kwargs,
            )
            
        # スタイルバンク（事前定義されたスタイル）
        self.register_buffer('style_bank', torch.randn(50, style_dim))
        self.style_names = {}  # スタイル名のマッピング
        
    def extract_style(
        self,
        mel: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """音声からスタイルを抽出"""
        return self.style_encoder(mel, mask)
        
    def apply_style(
        self,
        features: torch.Tensor,
        style_embedding: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """特徴にスタイルを適用"""
        if layer_indices is None:
            layer_indices = range(self.style_adapter.num_layers)
            
        adapted_features = features
        for idx in layer_indices:
            adapted_features = self.style_adapter(
                adapted_features, style_embedding, idx
            )
            
        return adapted_features
        
    def mix_styles(
        self,
        source_styles: List[torch.Tensor],
        mix_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """複数のスタイルを混合"""
        if not self.enable_mixing:
            return source_styles[0]
            
        return self.style_mixer(source_styles, mix_weights)
        
    def forward(
        self,
        features: torch.Tensor,
        reference_mel: Optional[torch.Tensor] = None,
        style_embedding: Optional[torch.Tensor] = None,
        style_id: Optional[int] = None,
        mix_styles: Optional[List[torch.Tensor]] = None,
        mix_weights: Optional[torch.Tensor] = None,
        layer_indices: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: 入力特徴 (batch_size, seq_len, feature_dim)
            reference_mel: 参照音声のメルスペクトログラム
            style_embedding: 直接指定されたスタイル埋め込み
            style_id: スタイルバンクからのID
            mix_styles: 混合するスタイルのリスト
            mix_weights: 混合重み
            layer_indices: スタイルを適用するレイヤー
            
        Returns:
            outputs: 処理結果
        """
        outputs = {}
        
        # スタイル埋め込みの取得
        if style_embedding is None:
            if reference_mel is not None:
                # 参照音声からスタイル抽出
                style_outputs = self.extract_style(reference_mel)
                style_embedding = style_outputs['style_embedding']
                outputs.update(style_outputs)
            elif style_id is not None:
                # スタイルバンクから取得
                style_embedding = self.style_bank[style_id].unsqueeze(0)
            elif mix_styles is not None:
                # スタイル混合
                style_embedding = self.mix_styles(mix_styles, mix_weights)
            else:
                # デフォルトスタイル（ゼロベクトル）
                batch_size = features.shape[0]
                device = features.device
                style_embedding = torch.zeros(batch_size, self.style_encoder.style_dim, device=device)
                
        outputs['style_embedding'] = style_embedding
        
        # スタイル適用
        styled_features = self.apply_style(features, style_embedding, layer_indices)
        outputs['features'] = styled_features
        
        return outputs
        
    def register_style(self, name: str, style_embedding: torch.Tensor, idx: int):
        """名前付きスタイルを登録"""
        self.style_bank[idx] = style_embedding.detach()
        self.style_names[name] = idx
        
    def get_style_by_name(self, name: str) -> Optional[torch.Tensor]:
        """名前でスタイルを取得"""
        if name in self.style_names:
            idx = self.style_names[name]
            return self.style_bank[idx]
        return None


class AdaptiveStyleTransfer(nn.Module):
    """適応的スタイル転送
    
    コンテンツに応じてスタイル転送の強度を調整
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        style_dim: int = 256,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # コンテンツ分析
        self.content_analyzer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=2,
        )
        
        # スタイル強度予測
        self.strength_predictor = nn.Sequential(
            nn.Linear(feature_dim + style_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # 適応的FiLM
        self.adaptive_film = nn.ModuleList([
            nn.Sequential(
                nn.Linear(style_dim + 1, hidden_dim),  # +1 for strength
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim * 2),
            )
            for _ in range(3)
        ])
        
    def forward(
        self,
        features: torch.Tensor,
        style_embedding: torch.Tensor,
        content_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: 入力特徴 (batch_size, seq_len, feature_dim)
            style_embedding: スタイル埋め込み (batch_size, style_dim)
            content_mask: コンテンツマスク
            
        Returns:
            adapted_features: 適応的にスタイル転送された特徴
        """
        # コンテンツ分析
        content_features = self.content_analyzer(features, src_key_padding_mask=content_mask)
        
        # グローバルコンテンツ表現
        if content_mask is not None:
            mask_expanded = content_mask.unsqueeze(-1).float()
            content_features = content_features * (1 - mask_expanded)
            lengths = (1 - mask_expanded).sum(dim=1)
            content_global = content_features.sum(dim=1) / lengths.clamp(min=1)
        else:
            content_global = content_features.mean(dim=1)
            
        # スタイル強度の予測
        strength_input = torch.cat([content_global, style_embedding], dim=-1)
        style_strength = self.strength_predictor(strength_input)
        
        # 適応的スタイル適用
        adapted_features = features
        style_with_strength = torch.cat([style_embedding, style_strength], dim=-1)
        
        for film_layer in self.adaptive_film:
            params = film_layer(style_with_strength)
            scale, shift = params.chunk(2, dim=-1)
            scale = scale.unsqueeze(1) * style_strength.unsqueeze(1)
            shift = shift.unsqueeze(1) * style_strength.unsqueeze(1)
            adapted_features = adapted_features * (1 + scale) + shift
            
        return adapted_features