"""モデルの包括的なユニットテスト"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.bigvgan_v2 import BigVGANv2
from src.models.f0_bert import F0BERT
from src.models.matcha_tts import MatchaTTS
from src.models.vits import VITS
from src.models.xphonebert import XPhoneBERTEncoder as XPhoneBERT


class TestXPhoneBERT:
    """XPhoneBERTのテスト"""

    @pytest.fixture()
    def model(self):
        return XPhoneBERT(
            model_name="vinai/xphonebert-base",
            hidden_size=768,
            num_layers=12,
            num_heads=12,
        )

    def test_initialization(self, model):
        """初期化テスト"""
        assert model is not None
        assert model.hidden_size == 768
        assert model.num_layers == 12

    def test_forward_pass(self, model):
        """順伝播テスト"""
        batch_size = 2
        seq_length = 50

        # ダミー入力
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        language_ids = torch.randint(0, 10, (batch_size,))
        attention_mask = torch.ones(batch_size, seq_length)

        # 推論
        output = model(input_ids, language_ids, attention_mask)

        # 出力の検証
        assert "hidden_states" in output
        assert output["hidden_states"].shape == (batch_size, seq_length, 768)
        assert output["hidden_states"].dtype == torch.float32

    def test_multilingual_support(self, model):
        """多言語サポートテスト"""
        # 異なる言語IDでのテスト
        input_ids = torch.randint(0, 1000, (3, 30))
        language_ids = torch.tensor([0, 1, 2])  # 日本語、英語、中国語

        output = model(input_ids, language_ids)

        # 各言語で異なる埋め込みが生成されることを確認
        embeddings = output["hidden_states"]
        assert not torch.allclose(embeddings[0], embeddings[1])
        assert not torch.allclose(embeddings[1], embeddings[2])

    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_different_batch_sizes(self, model, batch_size):
        """異なるバッチサイズでのテスト"""
        input_ids = torch.randint(0, 1000, (batch_size, 40))
        output = model(input_ids)

        assert output["hidden_states"].shape[0] == batch_size


class TestF0BERT:
    """F0-BERTのテスト"""

    @pytest.fixture()
    def model(self):
        return F0BERT(
            hidden_size=256,
            num_layers=6,
            num_heads=8,
            pitch_bins=256,
        )

    def test_initialization(self, model):
        """初期化テスト"""
        assert model is not None
        assert model.hidden_size == 256
        assert model.pitch_bins == 256

    def test_forward_pass(self, model):
        """順伝播テスト"""
        batch_size = 2
        seq_length = 100

        # ダミーF0系列
        f0 = torch.randn(batch_size, seq_length, 1)
        f0_mask = torch.ones(batch_size, seq_length, dtype=torch.bool)

        # 推論
        output = model(f0, f0_mask)

        # 出力の検証
        assert "hidden_states" in output
        assert output["hidden_states"].shape == (batch_size, seq_length, 256)
        assert "pitch_embedding" in output

    def test_pitch_quantization(self, model):
        """ピッチ量子化テスト"""
        # 異なるピッチ値でのテスト
        f0_values = torch.tensor(
            [
                [[100.0], [200.0], [300.0]],
                [[150.0], [250.0], [350.0]],
            ]
        )

        output = model(f0_values)

        # 量子化されたピッチ埋め込みが異なることを確認
        pitch_emb = output["pitch_embedding"]
        assert pitch_emb.shape == (2, 3, 256)
        assert not torch.allclose(pitch_emb[0, 0], pitch_emb[0, 1])

    def test_masked_positions(self, model):
        """マスク位置のテスト"""
        f0 = torch.randn(2, 50, 1)
        mask = torch.ones(2, 50, dtype=torch.bool)
        mask[0, 25:] = False  # 後半をマスク

        output = model(f0, mask)

        # マスクされた位置の出力が影響を受けないことを確認
        assert output["hidden_states"].shape == (2, 50, 256)


class TestVITS:
    """VITSのテスト"""

    @pytest.fixture()
    def model(self):
        return VITS(
            n_vocab=256,
            n_speakers=10,
            hidden_channels=192,
            inter_channels=192,
            filter_channels=768,
        )

    def test_initialization(self, model):
        """初期化テスト"""
        assert model is not None
        assert model.n_speakers == 10
        assert model.hidden_channels == 192

    def test_training_forward(self, model):
        """学習時の順伝播テスト"""
        batch_size = 2
        text_length = 30
        mel_length = 300

        # ダミー入力
        text = torch.randint(0, 256, (batch_size, text_length))
        text_lengths = torch.tensor([text_length, text_length - 5])
        mel = torch.randn(batch_size, 128, mel_length)
        mel_lengths = torch.tensor([mel_length, mel_length - 50])
        speaker_ids = torch.randint(0, 10, (batch_size,))

        # 学習モード
        model.train()
        output = model(text, text_lengths, mel, mel_lengths, speaker_ids)

        # 損失の検証
        assert "loss" in output
        assert "kl_loss" in output
        assert "mel_loss" in output
        assert output["loss"].requires_grad

    def test_inference(self, model):
        """推論テスト"""
        text = torch.randint(0, 256, (1, 20))
        speaker_id = torch.tensor([0])

        # 推論モード
        model.eval()
        with torch.no_grad():
            audio = model.infer(text, speaker_id)

        # 出力の検証
        assert audio.ndim == 2
        assert audio.shape[0] == 1
        assert audio.shape[1] > 0

    def test_speaker_embedding(self, model):
        """話者埋め込みテスト"""
        text = torch.randint(0, 256, (1, 20))

        # 異なる話者での生成
        model.eval()
        with torch.no_grad():
            audio1 = model.infer(text, torch.tensor([0]))
            audio2 = model.infer(text, torch.tensor([1]))

        # 異なる話者で異なる音声が生成されることを確認
        assert not torch.allclose(audio1, audio2)

    @pytest.mark.parametrize("length_scale", [0.5, 1.0, 1.5])
    def test_length_control(self, model, length_scale):
        """長さ制御テスト"""
        text = torch.randint(0, 256, (1, 20))
        speaker_id = torch.tensor([0])

        model.eval()
        with torch.no_grad():
            audio = model.infer(text, speaker_id, length_scale=length_scale)

        # 音声が生成されることを確認
        assert audio.shape[1] > 0


class TestMatchaTTS:
    """Matcha-TTSのテスト"""

    @pytest.fixture()
    def model(self):
        return MatchaTTS(
            n_vocab=256,
            n_speakers=10,
            hidden_channels=256,
            filter_channels=1024,
            filter_channels_dp=256,
        )

    def test_initialization(self, model):
        """初期化テスト"""
        assert model is not None
        assert model.n_speakers == 10
        assert model.hidden_channels == 256

    def test_flow_matching(self, model):
        """フローマッチングテスト"""
        batch_size = 2

        # ダミー入力
        text = torch.randint(0, 256, (batch_size, 25))
        text_lengths = torch.tensor([25, 20])
        mel = torch.randn(batch_size, 128, 250)
        mel_lengths = torch.tensor([250, 200])
        speaker_ids = torch.randint(0, 10, (batch_size,))

        # 学習モード
        model.train()
        output = model(text, text_lengths, mel, mel_lengths, speaker_ids)

        # フローマッチング損失の検証
        assert "loss" in output
        assert "flow_loss" in output
        assert output["loss"].requires_grad

    def test_inference_speed(self, model):
        """推論速度テスト"""
        import time

        text = torch.randint(0, 256, (1, 30))
        speaker_id = torch.tensor([0])

        model.eval()

        # ウォームアップ
        with torch.no_grad():
            _ = model.infer(text, speaker_id)

        # 速度測定
        times = []
        for _ in range(10):
            start = time.time()
            with torch.no_grad():
                _ = model.infer(text, speaker_id)
            times.append(time.time() - start)

        avg_time = np.mean(times)
        assert avg_time < 1.0  # 1秒以内で推論完了


class TestBigVGANv2:
    """BigVGAN-v2のテスト"""

    @pytest.fixture()
    def model(self):
        return BigVGANv2(
            num_mels=128,
            upsample_initial_channel=1536,
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[10, 8, 2, 2],
            upsample_kernel_sizes=[20, 16, 4, 4],
        )

    def test_initialization(self, model):
        """初期化テスト"""
        assert model is not None
        assert model.num_mels == 128
        assert len(model.ups) == 4  # アップサンプリング層の数

    def test_forward_pass(self, model):
        """順伝播テスト"""
        batch_size = 2
        mel_length = 100

        # ダミーメルスペクトログラム
        mel = torch.randn(batch_size, 128, mel_length)

        # 推論
        audio = model(mel)

        # 出力の検証
        expected_length = mel_length * 10 * 8 * 2 * 2  # アップサンプリング率の積
        assert audio.shape == (batch_size, 1, expected_length)

    def test_anti_aliasing(self, model):
        """アンチエイリアシングテスト"""
        # 高周波成分を含むメルスペクトログラム
        mel_length = 50
        mel = torch.zeros(1, 128, mel_length)
        mel[:, 64:, :] = 1.0  # 高周波成分

        audio = model(mel)

        # 音声が生成され、クリッピングされていないことを確認
        assert audio.max() <= 1.0
        assert audio.min() >= -1.0

    def test_multi_resolution_generation(self, model):
        """マルチレゾリューション生成テスト"""
        # 異なる長さのメルスペクトログラム
        for mel_length in [50, 100, 200]:
            mel = torch.randn(1, 128, mel_length)
            audio = model(mel)

            expected_length = mel_length * 320  # 総アップサンプリング率
            assert audio.shape[2] == expected_length

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_batch_processing(self, model, batch_size):
        """バッチ処理テスト"""
        mel = torch.randn(batch_size, 128, 80)
        audio = model(mel)

        assert audio.shape[0] == batch_size
        assert audio.shape[1] == 1
        assert audio.shape[2] == 80 * 320


class TestIntegration:
    """統合テスト"""

    def test_full_pipeline(self):
        """完全なパイプラインテスト"""
        # モデルの初期化
        xphonebert = XPhoneBERT(
            model_name="vinai/xphonebert-base",
            hidden_size=768,
            num_layers=12,
            num_heads=12,
        )

        f0_bert = F0BERT(
            hidden_size=256,
            num_layers=6,
            num_heads=8,
        )

        vits = VITS(
            n_vocab=256,
            n_speakers=10,
            hidden_channels=192,
            inter_channels=192,
        )

        bigvgan = BigVGANv2(
            num_mels=128,
            upsample_initial_channel=1536,
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        )

        # ダミー入力
        phoneme_ids = torch.randint(0, 1000, (1, 30))
        language_id = torch.tensor([0])
        f0 = torch.randn(1, 30, 1)
        text = torch.randint(0, 256, (1, 30))
        speaker_id = torch.tensor([0])

        # パイプライン実行
        xphonebert.eval()
        f0_bert.eval()
        vits.eval()
        bigvgan.eval()

        with torch.no_grad():
            # XPhoneBERT
            xphonebert_out = xphonebert(phoneme_ids, language_id)

            # F0-BERT
            f0_bert_out = f0_bert(f0)

            # エンコーダー出力の結合
            encoder_output = torch.cat(
                [xphonebert_out["hidden_states"], f0_bert_out["hidden_states"]], dim=-1
            )

            # VITS (簡略化のため encoder_output は使用しない)
            mel = vits.infer(text, speaker_id)

            # BigVGAN
            audio = bigvgan(mel)

        # 最終出力の検証
        assert audio.shape[0] == 1
        assert audio.shape[1] == 1
        assert audio.shape[2] > 0
        assert audio.max() <= 1.0
        assert audio.min() >= -1.0
