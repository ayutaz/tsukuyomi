"""推論とサービングの包括的なユニットテスト"""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.inference.optimized_inference import (
    InferenceConfig,
    LRUCache,
    OptimizedInferenceEngine,
    StreamingInference,
)
from src.serving.triton_server import (
    TritonModelExporter,
    TritonTTSClient,
)


class TestLRUCache:
    """LRUキャッシュのテスト"""

    def test_basic_operations(self):
        """基本操作のテスト"""
        cache = LRUCache(capacity=3)

        # 追加
        cache.put("key1", torch.tensor([1, 2, 3]))
        cache.put("key2", torch.tensor([4, 5, 6]))

        # 取得
        value = cache.get("key1")
        assert value is not None
        torch.testing.assert_close(value, torch.tensor([1, 2, 3]))

        # 存在しないキー
        assert cache.get("key3") is None

    def test_lru_eviction(self):
        """LRU削除のテスト"""
        cache = LRUCache(capacity=2)

        # キャパシティを超える追加
        cache.put("key1", torch.tensor([1]))
        cache.put("key2", torch.tensor([2]))
        cache.put("key3", torch.tensor([3]))  # key1が削除される

        # 最も古いkey1が削除されていることを確認
        assert cache.get("key1") is None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None

    def test_access_order_update(self):
        """アクセス順序更新のテスト"""
        cache = LRUCache(capacity=2)

        cache.put("key1", torch.tensor([1]))
        cache.put("key2", torch.tensor([2]))

        # key1をアクセスして最新に
        _ = cache.get("key1")

        # key3を追加（key2が削除される）
        cache.put("key3", torch.tensor([3]))

        assert cache.get("key1") is not None  # アクセスしたので残る
        assert cache.get("key2") is None  # 削除される
        assert cache.get("key3") is not None


class TestInferenceConfig:
    """推論設定のテスト"""

    def test_default_config(self):
        """デフォルト設定のテスト"""
        config = InferenceConfig()

        assert config.device == "cuda"
        assert config.dtype == torch.float16
        assert config.max_batch_size == 32
        assert config.cache_size == 1000

    def test_custom_config(self):
        """カスタム設定のテスト"""
        config = InferenceConfig(
            device="cpu",
            dtype=torch.float32,
            max_batch_size=16,
            use_compile=False,
        )

        assert config.device == "cpu"
        assert config.dtype == torch.float32
        assert config.max_batch_size == 16
        assert not config.use_compile


class TestOptimizedInferenceEngine:
    """最適化推論エンジンのテスト"""

    @pytest.fixture()
    def mock_model_path(self):
        """モックモデルパスの作成"""
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            # ダミーチェックポイント
            checkpoint = {
                "model_acoustic": {},
                "model_vocoder": {},
                "config": {},
            }
            torch.save(checkpoint, f.name)
            yield Path(f.name)

    @patch("src.inference.optimized_inference.VITS")
    @patch("src.inference.optimized_inference.BigVGANv2")
    def test_initialization(self, mock_bigvgan, mock_vits, mock_model_path):
        """初期化テスト"""
        # モックモデルの設定
        mock_vits_instance = MagicMock()
        mock_vits.return_value = mock_vits_instance

        mock_bigvgan_instance = MagicMock()
        mock_bigvgan.return_value = mock_bigvgan_instance

        # エンジンの初期化
        config = InferenceConfig(device="cpu")
        engine = OptimizedInferenceEngine(mock_model_path, config)

        assert engine.config.device == "cpu"
        assert "acoustic" in engine.models
        assert "vocoder" in engine.models

    def test_cache_key_computation(self):
        """キャッシュキー計算のテスト"""
        with (
            patch("src.inference.optimized_inference.VITS"),
            patch("src.inference.optimized_inference.BigVGANv2"),
        ):

            config = InferenceConfig(device="cpu")
            engine = OptimizedInferenceEngine.__new__(OptimizedInferenceEngine)
            engine.config = config

            # 同じ入力で同じキー
            key1 = engine._compute_cache_key("hello", 0, speed=1.0)
            key2 = engine._compute_cache_key("hello", 0, speed=1.0)
            assert key1 == key2

            # 異なる入力で異なるキー
            key3 = engine._compute_cache_key("world", 0, speed=1.0)
            assert key1 != key3

            # パラメータが異なれば異なるキー
            key4 = engine._compute_cache_key("hello", 0, speed=1.5)
            assert key1 != key4

    @patch("src.inference.optimized_inference.VITS")
    @patch("src.inference.optimized_inference.BigVGANv2")
    def test_batch_processing(self, mock_bigvgan, mock_vits, mock_model_path):
        """バッチ処理のテスト"""
        # モックの設定
        mock_vits_instance = MagicMock()
        mock_vits_instance.infer.return_value = torch.randn(1, 128, 100)
        mock_vits.return_value = mock_vits_instance

        mock_bigvgan_instance = MagicMock()
        mock_bigvgan_instance.return_value = torch.randn(1, 1, 32000)
        mock_bigvgan.return_value = mock_bigvgan_instance

        config = InferenceConfig(device="cpu", max_batch_size=2)
        engine = OptimizedInferenceEngine(mock_model_path, config)

        # バッチサイズ内
        texts = ["text1", "text2"]
        speaker_ids = [0, 1]

        audios = engine.synthesize_batch(texts, speaker_ids)

        assert len(audios) == 2
        assert all(isinstance(audio, np.ndarray) for audio in audios)

    def test_cache_effectiveness(self):
        """キャッシュ効果のテスト"""
        with (
            patch("src.inference.optimized_inference.VITS") as mock_vits,
            patch("src.inference.optimized_inference.BigVGANv2") as mock_bigvgan,
        ):

            # モックの設定
            mock_vits_instance = MagicMock()
            mock_vits_instance.infer.return_value = torch.randn(1, 128, 100)
            mock_vits.return_value = mock_vits_instance

            mock_bigvgan_instance = MagicMock()
            mock_bigvgan_instance.return_value = torch.randn(1, 1, 32000)
            mock_bigvgan.return_value = mock_bigvgan_instance

            config = InferenceConfig(device="cpu")
            engine = OptimizedInferenceEngine.__new__(OptimizedInferenceEngine)
            engine.config = config
            engine.models = {
                "acoustic": mock_vits_instance,
                "vocoder": mock_bigvgan_instance,
            }
            engine.mel_cache = LRUCache(100)
            engine.embedding_cache = LRUCache(100)

            # 初回呼び出し
            engine._vocoder_inference = MagicMock(return_value=np.zeros(32000))
            engine._acoustic_inference_batch = MagicMock(
                return_value=[torch.randn(1, 128, 100)]
            )

            audio1 = engine.synthesize_batch(["hello"], [0])

            # 2回目呼び出し（キャッシュヒット）
            audio2 = engine.synthesize_batch(["hello"], [0])

            # 音響モデルは1回しか呼ばれない
            engine._acoustic_inference_batch.assert_called_once()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_mixed_precision(self, dtype):
        """混合精度のテスト"""
        with (
            patch("src.inference.optimized_inference.VITS"),
            patch("src.inference.optimized_inference.BigVGANv2"),
        ):

            config = InferenceConfig(device="cpu", dtype=dtype)
            engine = OptimizedInferenceEngine.__new__(OptimizedInferenceEngine)
            engine.config = config

            # 精度設定の確認
            assert engine.config.dtype == dtype


class TestStreamingInference:
    """ストリーミング推論のテスト"""

    @pytest.fixture()
    def mock_engine(self):
        """モック推論エンジン"""
        engine = MagicMock()
        engine.synthesize.return_value = (np.random.randn(48000), 48000)
        return engine

    def test_text_splitting(self, mock_engine):
        """テキスト分割のテスト"""
        streaming = StreamingInference(mock_engine, chunk_size=1024)

        text = "これは最初の文です。これは2番目の文です。最後の文です。"
        sentences = streaming._split_text(text)

        assert len(sentences) == 3
        assert sentences[0] == "これは最初の文です"
        assert sentences[1] == "これは2番目の文です"
        assert sentences[2] == "最後の文です"

    def test_streaming_generation(self, mock_engine):
        """ストリーミング生成のテスト"""
        streaming = StreamingInference(mock_engine, chunk_size=1024)

        text = "これはテストです。"
        chunks = list(streaming.synthesize_streaming(text))

        # チャンクが生成されることを確認
        assert len(chunks) > 0
        assert all(isinstance(chunk, np.ndarray) for chunk in chunks)
        assert all(len(chunk) <= 1024 for chunk in chunks)

    def test_empty_text_handling(self, mock_engine):
        """空テキストの処理テスト"""
        streaming = StreamingInference(mock_engine)

        chunks = list(streaming.synthesize_streaming(""))

        # 空テキストでは何も生成されない
        assert len(chunks) == 0


class TestTritonModelExporter:
    """Tritonモデルエクスポーターのテスト"""

    @pytest.fixture()
    def temp_model_repo(self):
        """一時モデルリポジトリ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_initialization(self, temp_model_repo):
        """初期化テスト"""
        exporter = TritonModelExporter(temp_model_repo)

        assert exporter.model_repository == temp_model_repo
        assert temp_model_repo.exists()

    @patch("torch.jit.trace")
    def test_pytorch_export(self, mock_trace, temp_model_repo):
        """PyTorchモデルエクスポートのテスト"""
        exporter = TritonModelExporter(temp_model_repo)

        # モックモデル
        model = MagicMock(spec=torch.nn.Module)
        model.eval.return_value = model

        # トレースされたモデル
        traced_model = MagicMock()
        mock_trace.return_value = traced_model

        # エクスポート実行
        exporter._export_pytorch_model(
            model, temp_model_repo / "test_model/1", "test_model"
        )

        # save が呼ばれたことを確認
        traced_model.save.assert_called_once()

    def test_config_generation(self, temp_model_repo):
        """設定ファイル生成のテスト"""
        exporter = TritonModelExporter(temp_model_repo)

        model_name = "test_model"
        exporter._generate_config(
            model_name=model_name,
            backend="pytorch",
            version=1,
            input_shapes={"input": [1, 512]},
            output_shapes={"output": [1, 256]},
            batch_size=8,
        )

        config_path = temp_model_repo / model_name / "config.pbtxt"
        assert config_path.exists()

        # 設定内容の確認
        with open(config_path) as f:
            content = f.read()
            assert f'name: "{model_name}"' in content
            assert 'backend: "pytorch"' in content
            assert "max_batch_size: 8" in content

    def test_ensemble_export(self, temp_model_repo):
        """アンサンブルモデルエクスポートのテスト"""
        exporter = TritonModelExporter(temp_model_repo)

        models = [
            {"name": "encoder", "inputs": ["text"], "outputs": ["embeddings"]},
            {"name": "decoder", "inputs": ["embeddings"], "outputs": ["audio"]},
        ]

        exporter.export_ensemble("tts_ensemble", models)

        config_path = temp_model_repo / "tts_ensemble" / "config.pbtxt"
        assert config_path.exists()

        with open(config_path) as f:
            content = f.read()
            assert 'platform: "ensemble"' in content
            assert 'model_name: "encoder"' in content
            assert 'model_name: "decoder"' in content


class TestTritonTTSClient:
    """Triton TTSクライアントのテスト"""

    @patch("src.serving.triton_server.grpcclient")
    def test_initialization(self, mock_grpc):
        """初期化テスト"""
        client = TritonTTSClient(url="localhost:8001", protocol="grpc")

        assert client.url == "localhost:8001"
        assert client.protocol == "grpc"
        mock_grpc.InferenceServerClient.assert_called_once()

    @patch("src.serving.triton_server.grpcclient")
    def test_server_readiness(self, mock_grpc):
        """サーバー準備状態確認のテスト"""
        # モッククライアント
        mock_client = MagicMock()
        mock_client.is_server_ready.return_value = True
        mock_grpc.InferenceServerClient.return_value = mock_client

        client = TritonTTSClient()
        assert client.is_server_ready()

    @patch("src.serving.triton_server.grpcclient")
    def test_text_encoding(self, mock_grpc):
        """テキストエンコーディングのテスト"""
        client = TritonTTSClient()

        text = "hello"
        encoded = client._encode_text(text)

        assert isinstance(encoded, np.ndarray)
        assert encoded.dtype == np.int64
        assert len(encoded) == len(text)

    @patch("src.serving.triton_server.grpcclient")
    def test_batch_synthesis(self, mock_grpc):
        """バッチ合成のテスト"""
        # モック設定
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.as_numpy.return_value = np.random.randn(3, 48000)
        mock_client.infer.return_value = mock_response
        mock_grpc.InferenceServerClient.return_value = mock_client

        client = TritonTTSClient()

        texts = ["text1", "text2", "text3"]
        speaker_ids = [0, 1, 2]

        # モックInferInput
        mock_grpc.InferInput = MagicMock()
        mock_grpc.InferRequestedOutput = MagicMock()

        audios = client.synthesize_batch(texts, speaker_ids)

        assert len(audios) == 3
        assert all(isinstance(audio, np.ndarray) for audio in audios)


class TestPerformance:
    """パフォーマンステスト"""

    @pytest.mark.benchmark()
    def test_inference_speed(self):
        """推論速度のベンチマーク"""
        with (
            patch("src.inference.optimized_inference.VITS") as mock_vits,
            patch("src.inference.optimized_inference.BigVGANv2") as mock_bigvgan,
        ):

            # 軽量なモック
            mock_vits_instance = MagicMock()
            mock_vits_instance.infer.return_value = torch.zeros(1, 128, 100)
            mock_vits.return_value = mock_vits_instance

            mock_bigvgan_instance = MagicMock()
            mock_bigvgan_instance.return_value = torch.zeros(1, 1, 32000)
            mock_bigvgan.return_value = mock_bigvgan_instance

            config = InferenceConfig(device="cpu")
            engine = OptimizedInferenceEngine.__new__(OptimizedInferenceEngine)
            engine.config = config
            engine.models = {
                "acoustic": mock_vits_instance,
                "vocoder": mock_bigvgan_instance,
            }
            engine.mel_cache = LRUCache(100)
            engine.embedding_cache = LRUCache(100)
            engine.tokenizer = MagicMock()

            # 速度測定
            start = time.time()
            for _ in range(10):
                _ = engine.synthesize("テストテキスト", speaker_id=0)
            elapsed = time.time() - start

            # 10回の推論が1秒以内に完了
            assert elapsed < 1.0
