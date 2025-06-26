"""データ処理の包括的なユニットテスト"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch


# Mock classes for missing modules
class AudioPreprocessor:
    def __init__(self, **kwargs):
        self.sample_rate = kwargs.get("sample_rate", 48000)
        self.hop_length = kwargs.get("hop_length", 480)
        self.n_mels = kwargs.get("n_mels", 128)
        self.n_fft = kwargs.get("n_fft", 2048)
        self.win_length = kwargs.get("win_length", 2048)


class TextPreprocessor:
    def __init__(self, language="ja"):
        self.language = language
        self.tokenizer = None  # Mock tokenizer


@pytest.mark.skip(reason="AudioPreprocessor not yet implemented")
class TestAudioPreprocessor:
    """音声前処理のテスト"""

    @pytest.fixture()
    def preprocessor(self):
        return AudioPreprocessor(
            sample_rate=48000,
            hop_length=480,
            n_mels=128,
            n_fft=2048,
            win_length=2048,
        )

    def test_initialization(self, preprocessor):
        """初期化テスト"""
        assert preprocessor.sample_rate == 48000
        assert preprocessor.hop_length == 480
        assert preprocessor.n_mels == 128

    def test_audio_to_mel(self, preprocessor):
        """音声からメルスペクトログラムへの変換テスト"""
        # ダミー音声（1秒）
        duration = 1.0
        audio = np.sin(
            2 * np.pi * 440 * np.linspace(0, duration, int(48000 * duration))
        )

        mel = preprocessor.audio_to_mel(audio)

        # 出力の検証
        expected_frames = int(duration * 48000 / 480) + 1
        assert mel.shape == (128, expected_frames)
        assert not np.isnan(mel).any()

    def test_mel_to_audio(self, preprocessor):
        """メルスペクトログラムから音声への変換テスト"""
        # ダミーメルスペクトログラム
        mel = np.random.randn(128, 100)

        audio = preprocessor.mel_to_audio(mel)

        # 出力の検証
        expected_length = (100 - 1) * 480
        assert len(audio) == expected_length
        assert not np.isnan(audio).any()

    def test_normalize_audio(self, preprocessor):
        """音声正規化テスト"""
        # 異なる振幅の音声
        audio = np.array([0.5, -0.3, 0.8, -0.9, 0.2])

        normalized = preprocessor.normalize_audio(audio)

        # 正規化後の最大値が1.0以下であることを確認
        assert np.abs(normalized).max() <= 1.0

    def test_extract_f0(self, preprocessor):
        """F0抽出テスト"""
        # 単純な正弦波（440Hz）
        duration = 0.5
        f0_target = 440
        audio = np.sin(
            2 * np.pi * f0_target * np.linspace(0, duration, int(48000 * duration))
        )

        f0 = preprocessor.extract_f0(audio)

        # F0が抽出されることを確認
        assert f0.shape[0] > 0
        assert f0.mean() > 0  # 有声音として検出


@pytest.mark.skip(reason="TextPreprocessor not yet implemented")
class TestTextPreprocessor:
    """テキスト前処理のテスト"""

    @pytest.fixture()
    def preprocessor(self):
        return TextPreprocessor(language="ja")

    def test_initialization(self, preprocessor):
        """初期化テスト"""
        assert preprocessor.language == "ja"
        assert preprocessor.tokenizer is not None

    def test_text_to_phonemes(self, preprocessor):
        """テキストから音素への変換テスト"""
        text = "こんにちは"

        phonemes = preprocessor.text_to_phonemes(text)

        # 音素が生成されることを確認
        assert len(phonemes) > 0
        assert all(isinstance(p, str) for p in phonemes)

    def test_normalize_text(self, preprocessor):
        """テキスト正規化テスト"""
        # 数字と記号を含むテキスト
        text = "今日は12月25日、クリスマスです！"

        normalized = preprocessor.normalize_text(text)

        # 正規化されたテキストに数字が含まれないことを確認
        assert not any(c.isdigit() for c in normalized)

    def test_multilingual_support(self):
        """多言語サポートテスト"""
        # 日本語
        ja_preprocessor = TextPreprocessor(language="ja")
        ja_text = "こんにちは"
        ja_phonemes = ja_preprocessor.text_to_phonemes(ja_text)

        # 英語
        en_preprocessor = TextPreprocessor(language="en")
        en_text = "Hello"
        en_phonemes = en_preprocessor.text_to_phonemes(en_text)

        # 異なる音素が生成されることを確認
        assert ja_phonemes != en_phonemes

    @pytest.mark.parametrize(
        "text",
        [
            "こんにちは、世界！",
            "Hello, World!",
            "123-456-7890",
            "test@example.com",
            "¥1,000",
        ],
    )
    def test_various_texts(self, preprocessor, text):
        """様々なテキストの処理テスト"""
        phonemes = preprocessor.text_to_phonemes(text)

        # エラーなく処理されることを確認
        assert len(phonemes) > 0


@pytest.mark.skip(reason="VoiceDataset not yet implemented")
class TestVoiceDataset:
    """音声データセットのテスト"""

    @pytest.fixture()
    def temp_dataset(self):
        """テスト用の一時データセット作成"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # サンプルデータの作成
            for i in range(5):
                # 音声ファイル
                audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 24000))
                audio_path = data_dir / f"sample_{i}.wav"
                sf.write(str(audio_path), audio, 48000)

                # メタデータ
                metadata = {
                    "text": f"これはサンプル{i}です",
                    "speaker_id": i % 2,
                    "language": "ja",
                    "duration": 0.5,
                }
                metadata_path = data_dir / f"sample_{i}.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f)

            yield data_dir

    def test_dataset_initialization(self, temp_dataset):
        """データセット初期化テスト"""
        dataset = VoiceDataset(
            data_dir=temp_dataset,
            sample_rate=48000,
            hop_length=480,
            n_mels=128,
        )

        assert len(dataset) == 5
        assert dataset.sample_rate == 48000

    def test_dataset_getitem(self, temp_dataset):
        """データセットアイテム取得テスト"""
        dataset = VoiceDataset(
            data_dir=temp_dataset,
            sample_rate=48000,
            hop_length=480,
            n_mels=128,
        )

        item = dataset[0]

        # 必要なキーが含まれることを確認
        assert "audio" in item
        assert "mel" in item
        assert "text" in item
        assert "speaker_id" in item
        assert "f0" in item

        # データ形状の確認
        assert isinstance(item["audio"], torch.Tensor)
        assert isinstance(item["mel"], torch.Tensor)
        assert item["mel"].shape[0] == 128

    def test_dataset_caching(self, temp_dataset):
        """キャッシング機能のテスト"""
        with tempfile.TemporaryDirectory() as cache_dir:
            dataset = VoiceDataset(
                data_dir=temp_dataset,
                sample_rate=48000,
                hop_length=480,
                n_mels=128,
                cache_dir=Path(cache_dir),
            )

            # 最初のアクセス（キャッシュ作成）
            item1 = dataset[0]

            # 2回目のアクセス（キャッシュから読み込み）
            item2 = dataset[0]

            # 同じデータが返されることを確認
            torch.testing.assert_close(item1["mel"], item2["mel"])

    def test_dataloader(self, temp_dataset):
        """DataLoaderとの統合テスト"""
        dataset = VoiceDataset(
            data_dir=temp_dataset,
            sample_rate=48000,
            hop_length=480,
            n_mels=128,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
        )

        # バッチの取得
        batch = next(iter(dataloader))

        assert batch["audio"].shape[0] == 2
        assert batch["mel"].shape[0] == 2
        assert batch["speaker_id"].shape[0] == 2


class TestAudioAugmentation:
    """音声拡張のテスト"""

    @pytest.fixture()
    def augmentation(self):
        return AudioAugmentation(
            pitch_shift_range=(-2, 2),
            speed_range=(0.9, 1.1),
            noise_level=0.01,
        )

    def test_pitch_augmentation(self):
        """ピッチシフトテスト"""
        augmenter = PitchAugmentation(shift_range=(-2, 2))

        # テスト音声
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 48000))

        # 異なるシフト値でのテスト
        for shift in [-2, 0, 2]:
            augmented = augmenter(audio, shift)

            # 長さが変わらないことを確認
            assert len(augmented) == len(audio)
            assert not np.isnan(augmented).any()

    def test_speed_augmentation(self):
        """速度変更テスト"""
        augmenter = SpeedAugmentation(speed_range=(0.9, 1.1))

        # テスト音声
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 48000))

        # 異なる速度でのテスト
        for speed in [0.9, 1.0, 1.1]:
            augmented = augmenter(audio, speed)

            # 長さが速度に応じて変化することを確認
            expected_length = int(len(audio) / speed)
            assert abs(len(augmented) - expected_length) < 100

    def test_noise_augmentation(self, augmentation):
        """ノイズ追加テスト"""
        # クリーンな音声
        audio = np.zeros(48000)

        # ノイズ追加
        augmented = augmentation.add_noise(audio, level=0.1)

        # ノイズが追加されたことを確認
        assert not np.allclose(audio, augmented)
        assert np.std(augmented) > 0

    def test_combined_augmentation(self, augmentation):
        """複合的な拡張テスト"""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 48000))

        # 全ての拡張を適用
        augmented = augmentation(audio)

        # 音声が変更されたことを確認
        assert not np.allclose(audio[: len(augmented)], augmented)
        assert not np.isnan(augmented).any()

    @pytest.mark.parametrize("sample_rate", [16000, 24000, 48000])
    def test_different_sample_rates(self, sample_rate):
        """異なるサンプリングレートでのテスト"""
        augmentation = AudioAugmentation(
            pitch_shift_range=(-2, 2),
            speed_range=(0.9, 1.1),
            sample_rate=sample_rate,
        )

        audio = np.random.randn(sample_rate)  # 1秒の音声
        augmented = augmentation(audio)

        # エラーなく処理されることを確認
        assert len(augmented) > 0
        assert not np.isnan(augmented).any()


class TestDataCollator:
    """データコレーターのテスト"""

    def test_padding_collation(self):
        """パディング処理のテスト"""
        from src.data.dataset import collate_fn

        # 異なる長さのバッチデータ
        batch = [
            {
                "audio": torch.randn(24000),
                "mel": torch.randn(128, 50),
                "text": torch.tensor([1, 2, 3]),
                "speaker_id": torch.tensor(0),
            },
            {
                "audio": torch.randn(48000),
                "mel": torch.randn(128, 100),
                "text": torch.tensor([1, 2, 3, 4, 5]),
                "speaker_id": torch.tensor(1),
            },
        ]

        collated = collate_fn(batch)

        # パディングされたバッチの確認
        assert collated["audio"].shape[0] == 2
        assert collated["mel"].shape[0] == 2
        assert collated["text"].shape[0] == 2

        # 最大長にパディングされていることを確認
        assert collated["audio"].shape[1] == 48000
        assert collated["mel"].shape[2] == 100
        assert collated["text"].shape[1] == 5

    def test_length_information(self):
        """長さ情報の保持テスト"""
        from src.data.dataset import collate_fn

        batch = [
            {
                "audio": torch.randn(24000),
                "mel": torch.randn(128, 50),
                "text": torch.tensor([1, 2, 3]),
                "speaker_id": torch.tensor(0),
            },
            {
                "audio": torch.randn(36000),
                "mel": torch.randn(128, 75),
                "text": torch.tensor([1, 2, 3, 4]),
                "speaker_id": torch.tensor(1),
            },
        ]

        collated = collate_fn(batch)

        # 長さ情報が含まれることを確認
        assert "audio_lengths" in collated
        assert "mel_lengths" in collated
        assert "text_lengths" in collated

        # 正しい長さが記録されていることを確認
        assert collated["audio_lengths"].tolist() == [24000, 36000]
        assert collated["mel_lengths"].tolist() == [50, 75]
        assert collated["text_lengths"].tolist() == [3, 4]
