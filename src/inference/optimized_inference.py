"""最適化された推論エンジン

バッチ処理、キャッシング、量子化などの最適化技術を実装
"""

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from ..models.vits import VITS
from ..models.matcha_tts import MatchaTTS
from ..models.bigvgan import BigVGANv2
from ..models.xphonebert import XPhoneBERT
from ..models.f0_bert import F0BERT


@dataclass
class InferenceConfig:
    """推論設定"""
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    max_batch_size: int = 32
    cache_size: int = 1000
    use_compile: bool = True
    use_graph: bool = True
    num_threads: int = 4


class LRUCache:
    """LRUキャッシュの実装"""
    
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        
    def get(self, key: str) -> Optional[torch.Tensor]:
        if key not in self.cache:
            return None
        # 最近使用したものを最後に移動
        self.cache.move_to_end(key)
        return self.cache[key]
        
    def put(self, key: str, value: torch.Tensor):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # 最も古いものを削除
            self.cache.popitem(last=False)
            
    def clear(self):
        self.cache.clear()


class OptimizedInferenceEngine:
    """最適化された推論エンジン"""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        config: Optional[InferenceConfig] = None,
    ):
        self.config = config or InferenceConfig()
        self.device = torch.device(self.config.device)
        
        # キャッシュの初期化
        self.embedding_cache = LRUCache(self.config.cache_size)
        self.mel_cache = LRUCache(self.config.cache_size)
        
        # モデルの読み込み
        self._load_models(model_path)
        
        # 最適化の適用
        self._optimize_models()
        
        # トークナイザーの初期化
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/xphonebert-base")
        
    def _load_models(self, model_path: Union[str, Path]):
        """モデルの読み込み"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # モデルの初期化
        self.models = {}
        
        # XPhoneBERTの読み込み
        if 'model_xphonebert' in checkpoint:
            self.models['xphonebert'] = XPhoneBERT(
                model_name="vinai/xphonebert-base",
                hidden_size=768,
                num_layers=12,
                num_heads=12,
            )
            self.models['xphonebert'].load_state_dict(checkpoint['model_xphonebert'])
            
        # F0-BERTの読み込み
        if 'model_f0_bert' in checkpoint:
            self.models['f0_bert'] = F0BERT(
                hidden_size=256,
                num_layers=6,
                num_heads=8,
            )
            self.models['f0_bert'].load_state_dict(checkpoint['model_f0_bert'])
            
        # 音響モデルの読み込み
        if 'model_acoustic' in checkpoint:
            # VITSまたはMatcha-TTSの判定
            state_dict = checkpoint['model_acoustic']
            if 'text_encoder.embed' in state_dict:
                self.models['acoustic'] = VITS(
                    n_vocab=256,
                    n_speakers=100,
                    hidden_channels=192,
                    inter_channels=192,
                )
            else:
                self.models['acoustic'] = MatchaTTS(
                    n_vocab=256,
                    n_speakers=100,
                    hidden_channels=256,
                    filter_channels=1024,
                )
            self.models['acoustic'].load_state_dict(state_dict)
            
        # ボコーダーの読み込み
        if 'model_vocoder' in checkpoint:
            self.models['vocoder'] = BigVGANv2(
                num_mels=128,
                upsample_initial_channel=1536,
                resblock_kernel_sizes=[3, 7, 11],
                resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            )
            self.models['vocoder'].load_state_dict(checkpoint['model_vocoder'])
            
        # デバイスに移動
        for model in self.models.values():
            model.to(self.device)
            model.eval()
            
    def _optimize_models(self):
        """モデルの最適化"""
        # 推論モードの設定
        for model in self.models.values():
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
                
        # 混合精度の適用
        if self.config.dtype != torch.float32:
            for name, model in self.models.items():
                if name != 'xphonebert':  # BERTは精度が重要なのでfp32を維持
                    model.half() if self.config.dtype == torch.float16 else model.bfloat16()
                    
        # torch.compileの適用（PyTorch 2.0+）
        if self.config.use_compile and hasattr(torch, 'compile'):
            for name, model in self.models.items():
                self.models[name] = torch.compile(model, mode='reduce-overhead')
                
        # CUDAグラフの準備
        if self.config.use_graph and self.device.type == 'cuda':
            self._prepare_cuda_graphs()
            
    def _prepare_cuda_graphs(self):
        """CUDAグラフの準備"""
        self.graphs = {}
        self.graph_inputs = {}
        self.graph_outputs = {}
        
        # ダミー入力でグラフをキャプチャ
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            # 音響モデルのグラフ
            if 'acoustic' in self.models:
                dummy_text = torch.randint(0, 100, (1, 50)).to(self.device)
                dummy_speaker = torch.tensor([0]).to(self.device)
                dummy_mel = self.models['acoustic'].infer(dummy_text, dummy_speaker)
                
                self.graphs['acoustic'] = torch.cuda.current_graph()
                self.graph_inputs['acoustic'] = (dummy_text, dummy_speaker)
                self.graph_outputs['acoustic'] = dummy_mel
                
    def _compute_cache_key(self, text: str, speaker_id: int, **kwargs) -> str:
        """キャッシュキーの計算"""
        key_data = {
            'text': text,
            'speaker_id': speaker_id,
            **kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
        
    def preprocess_batch(
        self,
        texts: List[str],
        speaker_ids: List[int],
        language_ids: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """バッチ前処理"""
        batch = {}
        
        # テキストのトークン化
        if 'xphonebert' in self.models:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt',
            )
            batch['input_ids'] = encoded['input_ids'].to(self.device)
            batch['attention_mask'] = encoded['attention_mask'].to(self.device)
        else:
            # 簡易的なテキストエンコーディング
            text_tensors = []
            for text in texts:
                text_ids = [ord(c) % 256 for c in text]
                text_tensors.append(torch.tensor(text_ids))
            batch['text'] = pad_sequence(text_tensors, batch_first=True).to(self.device)
            
        # 話者IDとの言語ID
        batch['speaker_ids'] = torch.tensor(speaker_ids).to(self.device)
        if language_ids:
            batch['language_ids'] = torch.tensor(language_ids).to(self.device)
            
        return batch
        
    @torch.no_grad()
    def synthesize_batch(
        self,
        texts: List[str],
        speaker_ids: List[int],
        language_ids: Optional[List[int]] = None,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        energy: float = 1.0,
    ) -> List[np.ndarray]:
        """バッチ音声合成"""
        # バッチサイズの確認
        batch_size = len(texts)
        if batch_size > self.config.max_batch_size:
            # 大きすぎる場合は分割処理
            results = []
            for i in range(0, batch_size, self.config.max_batch_size):
                end_idx = min(i + self.config.max_batch_size, batch_size)
                sub_results = self.synthesize_batch(
                    texts[i:end_idx],
                    speaker_ids[i:end_idx],
                    language_ids[i:end_idx] if language_ids else None,
                    speed, pitch_shift, energy,
                )
                results.extend(sub_results)
            return results
            
        # キャッシュチェック
        audios = []
        uncached_indices = []
        
        for i, (text, speaker_id) in enumerate(zip(texts, speaker_ids)):
            cache_key = self._compute_cache_key(
                text, speaker_id,
                speed=speed, pitch_shift=pitch_shift, energy=energy,
            )
            cached_mel = self.mel_cache.get(cache_key)
            
            if cached_mel is not None:
                # キャッシュヒット
                audio = self._vocoder_inference(cached_mel)
                audios.append(audio)
            else:
                uncached_indices.append(i)
                audios.append(None)
                
        # キャッシュミスの処理
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            uncached_speakers = [speaker_ids[i] for i in uncached_indices]
            uncached_languages = [language_ids[i] for i in uncached_indices] if language_ids else None
            
            # バッチ推論
            uncached_mels = self._acoustic_inference_batch(
                uncached_texts,
                uncached_speakers,
                uncached_languages,
                speed, pitch_shift, energy,
            )
            
            # 結果の統合とキャッシュ
            for idx, mel in zip(uncached_indices, uncached_mels):
                cache_key = self._compute_cache_key(
                    texts[idx], speaker_ids[idx],
                    speed=speed, pitch_shift=pitch_shift, energy=energy,
                )
                self.mel_cache.put(cache_key, mel)
                
                audio = self._vocoder_inference(mel)
                audios[idx] = audio
                
        return audios
        
    def _acoustic_inference_batch(
        self,
        texts: List[str],
        speaker_ids: List[int],
        language_ids: Optional[List[int]],
        speed: float,
        pitch_shift: float,
        energy: float,
    ) -> List[torch.Tensor]:
        """音響モデルのバッチ推論"""
        # 前処理
        batch = self.preprocess_batch(texts, speaker_ids, language_ids)
        
        # エンコーダー処理
        encoder_outputs = []
        
        if 'xphonebert' in self.models:
            bert_output = self.models['xphonebert'](
                batch['input_ids'],
                batch.get('language_ids'),
                attention_mask=batch['attention_mask'],
            )
            encoder_outputs.append(bert_output['hidden_states'])
            
        # 音響モデル推論
        if 'acoustic' in self.models:
            encoder_output = torch.cat(encoder_outputs, dim=-1) if encoder_outputs else None
            
            # 速度、ピッチ、エネルギーの調整
            length_scale = 1.0 / speed if speed > 0 else 1.0
            
            if hasattr(self.models['acoustic'], 'infer_batch'):
                mels = self.models['acoustic'].infer_batch(
                    batch.get('text', batch.get('input_ids')),
                    batch['speaker_ids'],
                    encoder_output=encoder_output,
                    length_scale=length_scale,
                    pitch_shift=pitch_shift,
                    energy_scale=energy,
                )
            else:
                # バッチ非対応の場合は個別処理
                mels = []
                for i in range(len(texts)):
                    text_input = batch.get('text', batch.get('input_ids'))[i:i+1]
                    speaker_id = batch['speaker_ids'][i:i+1]
                    enc_out = encoder_output[i:i+1] if encoder_output is not None else None
                    
                    mel = self.models['acoustic'].infer(
                        text_input,
                        speaker_id,
                        encoder_output=enc_out,
                        length_scale=length_scale,
                    )
                    mels.append(mel)
                    
            return mels
        else:
            raise ValueError("No acoustic model loaded")
            
    def _vocoder_inference(self, mel: torch.Tensor) -> np.ndarray:
        """ボコーダー推論"""
        if 'vocoder' not in self.models:
            raise ValueError("No vocoder loaded")
            
        # メルスペクトログラムからオーディオ生成
        audio = self.models['vocoder'](mel.unsqueeze(0))
        
        # NumPy配列に変換
        audio_np = audio.squeeze().cpu().numpy()
        
        # クリッピング
        audio_np = np.clip(audio_np, -1.0, 1.0)
        
        return audio_np
        
    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        language_id: Optional[int] = None,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        energy: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """単一テキストの音声合成"""
        audio = self.synthesize_batch(
            [text],
            [speaker_id],
            [language_id] if language_id is not None else None,
            speed, pitch_shift, energy,
        )[0]
        
        return audio, 48000  # サンプリングレート
        
    def clear_cache(self):
        """キャッシュのクリア"""
        self.embedding_cache.clear()
        self.mel_cache.clear()
        
    def benchmark(self, num_samples: int = 100) -> Dict[str, float]:
        """パフォーマンスベンチマーク"""
        import random
        import string
        
        # テストデータの生成
        texts = []
        for _ in range(num_samples):
            length = random.randint(10, 100)
            text = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length))
            texts.append(text)
            
        speaker_ids = [random.randint(0, 10) for _ in range(num_samples)]
        
        # ウォームアップ
        _ = self.synthesize_batch(texts[:5], speaker_ids[:5])
        
        # ベンチマーク
        metrics = {}
        
        # シングル推論
        start_time = time.time()
        for text, speaker_id in zip(texts, speaker_ids):
            _ = self.synthesize(text, speaker_id)
        single_time = time.time() - start_time
        metrics['single_inference_time'] = single_time / num_samples
        
        # バッチ推論
        batch_sizes = [1, 4, 8, 16, 32]
        for batch_size in batch_sizes:
            if batch_size > num_samples:
                continue
                
            start_time = time.time()
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                _ = self.synthesize_batch(
                    texts[i:end_idx],
                    speaker_ids[i:end_idx],
                )
            batch_time = time.time() - start_time
            metrics[f'batch_{batch_size}_inference_time'] = batch_time / num_samples
            
        # キャッシュ効果
        self.clear_cache()
        
        # 初回推論
        start_time = time.time()
        _ = self.synthesize_batch(texts[:10], speaker_ids[:10])
        first_time = time.time() - start_time
        
        # キャッシュ済み推論
        start_time = time.time()
        _ = self.synthesize_batch(texts[:10], speaker_ids[:10])
        cached_time = time.time() - start_time
        
        metrics['cache_speedup'] = first_time / cached_time
        
        return metrics


class StreamingInference:
    """ストリーミング推論の実装"""
    
    def __init__(self, engine: OptimizedInferenceEngine, chunk_size: int = 1024):
        self.engine = engine
        self.chunk_size = chunk_size
        
    def synthesize_streaming(
        self,
        text: str,
        speaker_id: int = 0,
        language_id: Optional[int] = None,
    ):
        """ストリーミング音声合成
        
        Yields:
            np.ndarray: オーディオチャンク
        """
        # テキストをチャンクに分割
        sentences = self._split_text(text)
        
        for sentence in sentences:
            if sentence.strip():
                # 文ごとに合成
                audio, _ = self.engine.synthesize(
                    sentence,
                    speaker_id,
                    language_id,
                )
                
                # チャンクごとに出力
                for i in range(0, len(audio), self.chunk_size):
                    yield audio[i:i + self.chunk_size]
                    
    def _split_text(self, text: str) -> List[str]:
        """テキストを文に分割"""
        # 簡易的な文分割
        import re
        sentences = re.split(r'[。！？\.\!\?]+', text)
        return [s for s in sentences if s.strip()]