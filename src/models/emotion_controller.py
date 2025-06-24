"""感情制御モジュール

音声合成に感情表現を追加するためのコンポーネント
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class EmotionEncoder(nn.Module):
    """感情エンコーダー

    感情ラベルまたは連続値から感情埋め込みを生成
    """

    def __init__(
        self,
        num_emotions: int = 10,
        emotion_embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_continuous: bool = True,
    ):
        super().__init__()
        self.num_emotions = num_emotions
        self.emotion_embedding_dim = emotion_embedding_dim
        self.use_continuous = use_continuous

        # カテゴリカル感情埋め込み
        self.emotion_embeddings = nn.Embedding(num_emotions, emotion_embedding_dim)

        # 連続感情値エンコーダー（VAE的アプローチ）
        if use_continuous:
            # Valence-Arousal-Dominance (VAD) モデル
            self.vad_encoder = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, emotion_embedding_dim * 2),  # mean + logvar
            )

        # 感情変換ネットワーク
        self.emotion_transform = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(emotion_embedding_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, emotion_embedding_dim),
                )
                for _ in range(num_layers)
            ]
        )

        # 感情強度制御
        self.intensity_scale = nn.Parameter(torch.ones(1))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAEの再パラメータ化トリック"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        emotion_id: Optional[torch.Tensor] = None,
        emotion_vad: Optional[torch.Tensor] = None,
        intensity: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            emotion_id: カテゴリカル感情ID (batch_size,)
            emotion_vad: 連続感情値 [Valence, Arousal, Dominance] (batch_size, 3)
            intensity: 感情の強度 (0.0-2.0)

        Returns:
            emotion_embedding: 感情埋め込み (batch_size, emotion_embedding_dim)
            kl_loss: KLダイバージェンス損失（連続値の場合）
        """
        kl_loss = None

        if emotion_id is not None:
            # カテゴリカル感情
            emotion_embedding = self.emotion_embeddings(emotion_id)
        elif emotion_vad is not None and self.use_continuous:
            # 連続感情値（VAD）
            vad_output = self.vad_encoder(emotion_vad)
            mu, logvar = vad_output.chunk(2, dim=-1)
            emotion_embedding = self.reparameterize(mu, logvar)

            # KL損失の計算
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / emotion_vad.shape[0]  # バッチ平均
        else:
            # ニュートラル感情（ゼロベクトル）
            batch_size = 1 if emotion_id is None else emotion_id.shape[0]
            device = next(self.parameters()).device
            emotion_embedding = torch.zeros(
                batch_size, self.emotion_embedding_dim, device=device
            )

        # 感情変換レイヤー
        for transform in self.emotion_transform:
            emotion_embedding = emotion_embedding + transform(emotion_embedding)

        # 強度スケーリング
        emotion_embedding = emotion_embedding * intensity * self.intensity_scale

        return emotion_embedding, kl_loss


class EmotionFusion(nn.Module):
    """感情情報の融合モジュール

    テキスト/音響特徴と感情埋め込みを融合
    """

    def __init__(
        self,
        feature_dim: int,
        emotion_dim: int,
        fusion_type: str = "concat",  # concat, add, gate, cross_attention
        hidden_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.emotion_dim = emotion_dim
        self.fusion_type = fusion_type

        if fusion_type == "concat":
            self.output_dim = feature_dim + emotion_dim
            self.fusion = nn.Sequential(
                nn.Linear(self.output_dim, hidden_dim or self.output_dim),
                nn.LayerNorm(hidden_dim or self.output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        elif fusion_type == "add":
            assert feature_dim == emotion_dim, "Dimensions must match for addition"
            self.output_dim = feature_dim
            self.fusion = nn.Identity()
        elif fusion_type == "gate":
            self.output_dim = feature_dim
            # ゲート機構
            self.gate = nn.Sequential(
                nn.Linear(feature_dim + emotion_dim, feature_dim),
                nn.Sigmoid(),
            )
            self.transform = nn.Linear(emotion_dim, feature_dim)
        elif fusion_type == "cross_attention":
            self.output_dim = feature_dim
            # クロスアテンション
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.emotion_proj = nn.Linear(emotion_dim, feature_dim)
            self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        features: torch.Tensor,
        emotion_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: 入力特徴 (batch_size, seq_len, feature_dim)
            emotion_embedding: 感情埋め込み (batch_size, emotion_dim)
            mask: アテンションマスク (batch_size, seq_len)

        Returns:
            fused_features: 融合後の特徴 (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len = features.shape[:2]

        if self.fusion_type == "concat":
            # 感情埋め込みを時間軸に拡張
            emotion_expanded = emotion_embedding.unsqueeze(1).expand(-1, seq_len, -1)
            fused = torch.cat([features, emotion_expanded], dim=-1)
            return self.fusion(fused)

        elif self.fusion_type == "add":
            emotion_expanded = emotion_embedding.unsqueeze(1).expand(-1, seq_len, -1)
            return features + emotion_expanded

        elif self.fusion_type == "gate":
            emotion_expanded = emotion_embedding.unsqueeze(1).expand(-1, seq_len, -1)
            gate_input = torch.cat([features, emotion_expanded], dim=-1)
            gate = self.gate(gate_input)
            emotion_transformed = self.transform(emotion_expanded)
            return features + gate * emotion_transformed

        elif self.fusion_type == "cross_attention":
            # 感情埋め込みをクエリとして使用
            emotion_query = self.emotion_proj(emotion_embedding).unsqueeze(1)
            attended, _ = self.cross_attention(
                emotion_query,
                features,
                features,
                key_padding_mask=mask,
            )
            # 残差接続
            return self.norm(features + attended.expand(-1, seq_len, -1))


class EmotionPredictor(nn.Module):
    """テキストから感情を予測するモジュール"""

    def __init__(
        self,
        input_dim: int,
        num_emotions: int = 10,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # テキストエンコーダー（Transformerベース）
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # 感情分類ヘッド
        self.emotion_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_emotions),
        )

        # VAD回帰ヘッド
        self.vad_regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # Valence, Arousal, Dominance
            nn.Tanh(),  # -1 to 1の範囲に正規化
        )

    def forward(
        self,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_features: テキスト特徴 (batch_size, seq_len, input_dim)
            text_mask: パディングマスク (batch_size, seq_len)

        Returns:
            predictions: 感情予測結果
        """
        # テキストエンコーディング
        encoded = self.text_encoder(text_features, src_key_padding_mask=text_mask)

        # グローバルプーリング
        if text_mask is not None:
            mask_expanded = text_mask.unsqueeze(-1).float()
            encoded = encoded * (1 - mask_expanded)
            lengths = (1 - mask_expanded).sum(dim=1)
            pooled = encoded.sum(dim=1) / lengths.clamp(min=1)
        else:
            pooled = encoded.mean(dim=1)

        # 感情予測
        emotion_logits = self.emotion_classifier(pooled)
        emotion_probs = F.softmax(emotion_logits, dim=-1)

        # VAD値予測
        vad_values = self.vad_regressor(pooled)

        return {
            "emotion_logits": emotion_logits,
            "emotion_probs": emotion_probs,
            "vad_values": vad_values,
        }


class EmotionController(nn.Module):
    """統合感情制御モジュール"""

    def __init__(
        self,
        feature_dim: int = 768,
        num_emotions: int = 10,
        emotion_embedding_dim: int = 256,
        fusion_type: str = "cross_attention",
        predict_emotion: bool = True,
        **kwargs,
    ):
        super().__init__()

        # 感情エンコーダー
        self.emotion_encoder = EmotionEncoder(
            num_emotions=num_emotions,
            emotion_embedding_dim=emotion_embedding_dim,
            **kwargs,
        )

        # 感情融合
        self.emotion_fusion = EmotionFusion(
            feature_dim=feature_dim,
            emotion_dim=emotion_embedding_dim,
            fusion_type=fusion_type,
            **kwargs,
        )

        # 感情予測（オプション）
        self.predict_emotion = predict_emotion
        if predict_emotion:
            self.emotion_predictor = EmotionPredictor(
                input_dim=feature_dim,
                num_emotions=num_emotions,
                **kwargs,
            )

        # 感情ラベルの定義
        self.emotion_labels = [
            "neutral",  # ニュートラル
            "happy",  # 喜び
            "sad",  # 悲しみ
            "angry",  # 怒り
            "fearful",  # 恐怖
            "surprised",  # 驚き
            "disgusted",  # 嫌悪
            "excited",  # 興奮
            "calm",  # 穏やか
            "confident",  # 自信
        ]

    def forward(
        self,
        features: torch.Tensor,
        emotion_id: Optional[torch.Tensor] = None,
        emotion_vad: Optional[torch.Tensor] = None,
        emotion_intensity: float = 1.0,
        predict_from_text: bool = False,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: 入力特徴 (batch_size, seq_len, feature_dim)
            emotion_id: 感情ID (batch_size,)
            emotion_vad: VAD値 (batch_size, 3)
            emotion_intensity: 感情強度
            predict_from_text: テキストから感情を予測
            mask: マスク (batch_size, seq_len)

        Returns:
            outputs: 処理結果
        """
        outputs = {}

        # テキストから感情予測
        if predict_from_text and self.predict_emotion and emotion_id is None:
            predictions = self.emotion_predictor(features, mask)
            outputs.update(predictions)

            # 予測された感情を使用
            emotion_id = predictions["emotion_logits"].argmax(dim=-1)
            emotion_vad = predictions["vad_values"]

        # 感情エンコーディング
        emotion_embedding, kl_loss = self.emotion_encoder(
            emotion_id, emotion_vad, emotion_intensity
        )
        outputs["emotion_embedding"] = emotion_embedding
        if kl_loss is not None:
            outputs["emotion_kl_loss"] = kl_loss

        # 特徴との融合
        fused_features = self.emotion_fusion(features, emotion_embedding, mask)
        outputs["features"] = fused_features

        return outputs

    def get_emotion_embedding(
        self,
        emotion_name: Optional[str] = None,
        emotion_id: Optional[int] = None,
        emotion_vad: Optional[List[float]] = None,
        intensity: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """感情埋め込みを取得する便利メソッド"""
        if device is None:
            device = next(self.parameters()).device

        if emotion_name is not None:
            emotion_id = self.emotion_labels.index(emotion_name)
            emotion_id = torch.tensor([emotion_id], device=device)
        elif emotion_id is not None:
            emotion_id = torch.tensor([emotion_id], device=device)
        elif emotion_vad is not None:
            emotion_vad = torch.tensor([emotion_vad], device=device, dtype=torch.float)

        embedding, _ = self.emotion_encoder(emotion_id, emotion_vad, intensity)
        return embedding
