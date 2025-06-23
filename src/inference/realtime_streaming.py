"""リアルタイムストリーミング最適化

低レイテンシでのストリーミング音声合成
"""

import asyncio
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.streams import Stream

from ..models.vits import VITS
from ..models.bigvgan import BigVGANv2


@dataclass
class StreamingConfig:
    """ストリーミング設定"""
    chunk_size: int = 256  # 文字単位のチャンクサイズ
    lookahead_chunks: int = 2  # 先読みチャンク数
    mel_chunk_size: int = 64  # メルスペクトログラムのチャンクサイズ
    audio_chunk_size: int = 4096  # 音声サンプルのチャンクサイズ
    buffer_size: int = 10  # バッファサイズ
    max_latency_ms: float = 100.0  # 最大許容レイテンシ（ミリ秒）
    use_cuda_streams: bool = True  # CUDAストリームの使用
    num_cuda_streams: int = 3  # CUDAストリーム数
    enable_jit: bool = True  # JITコンパイルの使用


class ChunkedAttention(nn.Module):
    """チャンク化されたアテンション機構
    
    メモリ効率的なストリーミング用アテンション
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        chunk_size: int = 64,
        memory_size: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.memory_size = memory_size
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # メモリバッファ
        self.register_buffer('memory_buffer', torch.zeros(1, memory_size, embed_dim))
        self.memory_ptr = 0
        
    def forward(
        self,
        x: torch.Tensor,
        use_memory: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 入力 (batch_size, chunk_size, embed_dim)
            use_memory: メモリを使用するか
            
        Returns:
            output: 出力
            new_memory: 更新されたメモリ
        """
        batch_size = x.shape[0]
        
        if use_memory and self.memory_ptr > 0:
            # メモリと結合
            memory = self.memory_buffer[:, :self.memory_ptr].expand(batch_size, -1, -1)
            full_input = torch.cat([memory, x], dim=1)
        else:
            full_input = x
            
        # アテンション計算
        output, _ = self.attention(x, full_input, full_input)
        
        # メモリ更新
        if use_memory:
            # 新しいメモリを追加
            new_memory_size = min(x.shape[1], self.memory_size - self.memory_ptr)
            if new_memory_size > 0:
                self.memory_buffer[:, self.memory_ptr:self.memory_ptr + new_memory_size] = x[:1, -new_memory_size:]
                self.memory_ptr += new_memory_size
            else:
                # メモリが満杯の場合はFIFO
                self.memory_buffer = torch.cat([
                    self.memory_buffer[:, x.shape[1]:],
                    x[:1]
                ], dim=1)
                
        return output, self.memory_buffer
        
    def reset_memory(self):
        """メモリをリセット"""
        self.memory_buffer.zero_()
        self.memory_ptr = 0


class StreamingVITS(nn.Module):
    """ストリーミング対応VITS
    
    チャンク単位での生成に最適化
    """
    
    def __init__(
        self,
        base_vits: VITS,
        chunk_size: int = 64,
        overlap_size: int = 16,
    ):
        super().__init__()
        self.base_vits = base_vits
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        
        # チャンク化アテンションに置き換え
        self._replace_attention_layers()
        
        # 状態バッファ
        self.encoder_state = None
        self.decoder_state = None
        self.prev_overlap = None
        
    def _replace_attention_layers(self):
        """アテンション層をチャンク化バージョンに置き換え"""
        # TODO: 実際のVITSアーキテクチャに合わせて実装
        pass
        
    def encode_chunk(
        self,
        text_chunk: torch.Tensor,
        speaker_id: torch.Tensor,
    ) -> torch.Tensor:
        """テキストチャンクをエンコード"""
        # エンコーダーの順伝播（状態を保持）
        with torch.no_grad():
            # 簡略化された実装
            encoded = self.base_vits.text_encoder(text_chunk)
            if speaker_id is not None:
                spk_emb = self.base_vits.speaker_embedding(speaker_id)
                encoded = encoded + spk_emb.unsqueeze(1)
                
        return encoded
        
    def decode_chunk(
        self,
        encoded_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """エンコードされたチャンクをデコード"""
        with torch.no_grad():
            # 簡略化された実装
            mel_chunk = self.base_vits.decoder(encoded_chunk)
            
            # オーバーラップ処理
            if self.prev_overlap is not None:
                # クロスフェード
                fade_len = self.overlap_size
                fade_in = torch.linspace(0, 1, fade_len, device=mel_chunk.device)
                fade_out = 1 - fade_in
                
                mel_chunk[:, :, :fade_len] = (
                    mel_chunk[:, :, :fade_len] * fade_in +
                    self.prev_overlap * fade_out
                )
                
            # 次のチャンク用にオーバーラップを保存
            if self.overlap_size > 0:
                self.prev_overlap = mel_chunk[:, :, -self.overlap_size:].clone()
                
        return mel_chunk
        
    def reset_state(self):
        """内部状態をリセット"""
        self.encoder_state = None
        self.decoder_state = None
        self.prev_overlap = None


class StreamingPipeline:
    """ストリーミング音声合成パイプライン"""
    
    def __init__(
        self,
        acoustic_model: Union[VITS, StreamingVITS],
        vocoder: BigVGANv2,
        config: StreamingConfig = StreamingConfig(),
        device: str = "cuda",
    ):
        self.acoustic_model = acoustic_model
        self.vocoder = vocoder
        self.config = config
        self.device = torch.device(device)
        
        # モデルを評価モードに
        self.acoustic_model.eval()
        self.vocoder.eval()
        
        # JITコンパイル
        if config.enable_jit and hasattr(torch.jit, 'script'):
            try:
                self.vocoder = torch.jit.script(self.vocoder)
            except:
                pass  # JITコンパイルが失敗した場合は通常のモデルを使用
                
        # CUDAストリーム
        self.cuda_streams = []
        if config.use_cuda_streams and device == "cuda":
            for _ in range(config.num_cuda_streams):
                self.cuda_streams.append(Stream())
                
        # バッファ
        self.text_buffer = deque(maxlen=config.buffer_size)
        self.mel_buffer = deque(maxlen=config.buffer_size)
        self.audio_buffer = deque(maxlen=config.buffer_size)
        
        # 処理スレッド
        self.processing_thread = None
        self.stop_event = threading.Event()
        
    def preprocess_text_chunk(self, text: str) -> torch.Tensor:
        """テキストチャンクの前処理"""
        # 簡略化された実装
        text_ids = [ord(c) % 256 for c in text]
        return torch.tensor(text_ids).unsqueeze(0).to(self.device)
        
    def process_text_chunk(
        self,
        text_chunk: str,
        speaker_id: int = 0,
    ) -> np.ndarray:
        """テキストチャンクを処理して音声を生成"""
        # テキスト前処理
        text_tensor = self.preprocess_text_chunk(text_chunk)
        speaker_tensor = torch.tensor([speaker_id]).to(self.device)
        
        # 音響モデル処理
        if isinstance(self.acoustic_model, StreamingVITS):
            encoded = self.acoustic_model.encode_chunk(text_tensor, speaker_tensor)
            mel_chunk = self.acoustic_model.decode_chunk(encoded)
        else:
            # 通常のVITS（非ストリーミング）
            with torch.no_grad():
                mel_chunk = self.acoustic_model.infer(text_tensor, speaker_tensor)
                
        # ボコーダー処理
        with torch.no_grad():
            if self.config.use_cuda_streams and self.cuda_streams:
                # CUDAストリームを使用
                stream_idx = len(self.mel_buffer) % len(self.cuda_streams)
                with torch.cuda.stream(self.cuda_streams[stream_idx]):
                    audio_chunk = self.vocoder(mel_chunk)
                    audio_chunk = audio_chunk.squeeze().cpu().numpy()
            else:
                audio_chunk = self.vocoder(mel_chunk)
                audio_chunk = audio_chunk.squeeze().cpu().numpy()
                
        return audio_chunk
        
    async def stream_synthesis(
        self,
        text_stream: AsyncIterator[str],
        speaker_id: int = 0,
    ) -> AsyncIterator[np.ndarray]:
        """非同期ストリーミング音声合成"""
        buffer = ""
        
        async for text_chunk in text_stream:
            buffer += text_chunk
            
            # チャンクサイズに達したら処理
            while len(buffer) >= self.config.chunk_size:
                chunk_to_process = buffer[:self.config.chunk_size]
                buffer = buffer[self.config.chunk_size:]
                
                # 処理開始時刻
                start_time = time.time()
                
                # 音声生成
                audio = await asyncio.get_event_loop().run_in_executor(
                    None, self.process_text_chunk, chunk_to_process, speaker_id
                )
                
                # レイテンシチェック
                latency_ms = (time.time() - start_time) * 1000
                if latency_ms > self.config.max_latency_ms:
                    print(f"Warning: Latency {latency_ms:.1f}ms exceeds limit")
                    
                yield audio
                
        # 残りのバッファを処理
        if buffer:
            audio = await asyncio.get_event_loop().run_in_executor(
                None, self.process_text_chunk, buffer, speaker_id
            )
            yield audio
            
    def start_processing_thread(self):
        """処理スレッドを開始"""
        if self.processing_thread is None:
            self.stop_event.clear()
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.start()
            
    def stop_processing_thread(self):
        """処理スレッドを停止"""
        if self.processing_thread is not None:
            self.stop_event.set()
            self.processing_thread.join()
            self.processing_thread = None
            
    def _processing_loop(self):
        """バックグラウンド処理ループ"""
        while not self.stop_event.is_set():
            # テキストバッファから処理
            if self.text_buffer:
                text_chunk = self.text_buffer.popleft()
                mel_chunk = self._process_to_mel(text_chunk)
                self.mel_buffer.append(mel_chunk)
                
            # メルバッファから処理
            if self.mel_buffer:
                mel_chunk = self.mel_buffer.popleft()
                audio_chunk = self._process_to_audio(mel_chunk)
                self.audio_buffer.append(audio_chunk)
                
            # 短いスリープ
            time.sleep(0.001)
            
    def _process_to_mel(self, text_chunk: str) -> torch.Tensor:
        """テキストからメルスペクトログラムへ"""
        # 実装は process_text_chunk と同様
        pass
        
    def _process_to_audio(self, mel_chunk: torch.Tensor) -> np.ndarray:
        """メルスペクトログラムから音声へ"""
        # 実装は process_text_chunk と同様
        pass


class WebSocketStreaming:
    """WebSocket経由のストリーミング"""
    
    def __init__(
        self,
        pipeline: StreamingPipeline,
        sample_rate: int = 48000,
    ):
        self.pipeline = pipeline
        self.sample_rate = sample_rate
        
    async def handle_connection(self, websocket, path):
        """WebSocket接続のハンドリング"""
        try:
            # テキストストリームを非同期で受信
            async def text_generator():
                async for message in websocket:
                    data = json.loads(message)
                    if data['type'] == 'text':
                        yield data['content']
                    elif data['type'] == 'end':
                        break
                        
            # 音声ストリーミング
            speaker_id = 0  # デフォルト話者
            async for audio_chunk in self.pipeline.stream_synthesis(
                text_generator(), speaker_id
            ):
                # 音声データをBase64エンコードして送信
                audio_b64 = base64.b64encode(audio_chunk.tobytes()).decode()
                await websocket.send(json.dumps({
                    'type': 'audio',
                    'data': audio_b64,
                    'sample_rate': self.sample_rate,
                }))
                
            # 終了通知
            await websocket.send(json.dumps({'type': 'end'}))
            
        except Exception as e:
            print(f"WebSocket error: {e}")
            
    async def start_server(self, host: str = "localhost", port: int = 8765):
        """WebSocketサーバーを開始"""
        import websockets
        async with websockets.serve(self.handle_connection, host, port):
            print(f"Streaming server started at ws://{host}:{port}")
            await asyncio.Future()  # 永続的に実行


class OptimizedVocoder(nn.Module):
    """ストリーミング最適化されたボコーダー
    
    インクリメンタル生成をサポート
    """
    
    def __init__(
        self,
        base_vocoder: BigVGANv2,
        cache_size: int = 16,
    ):
        super().__init__()
        self.base_vocoder = base_vocoder
        self.cache_size = cache_size
        
        # 畳み込みキャッシュ
        self.conv_caches = {}
        self._initialize_caches()
        
    def _initialize_caches(self):
        """畳み込み層のキャッシュを初期化"""
        # 各畳み込み層用のキャッシュを作成
        for name, module in self.base_vocoder.named_modules():
            if isinstance(module, nn.Conv1d):
                kernel_size = module.kernel_size[0]
                in_channels = module.in_channels
                self.conv_caches[name] = torch.zeros(
                    1, in_channels, kernel_size - 1
                )
                
    def forward_incremental(
        self,
        mel_frame: torch.Tensor,
    ) -> torch.Tensor:
        """インクリメンタル推論
        
        Args:
            mel_frame: 単一のメルフレーム (batch_size, n_mels, 1)
            
        Returns:
            audio_frame: 生成された音声フレーム
        """
        # TODO: 実際のBigVGANアーキテクチャに合わせて実装
        # ここでは簡略化された実装
        with torch.no_grad():
            audio_frame = self.base_vocoder(mel_frame)
        return audio_frame
        
    def reset_cache(self):
        """キャッシュをリセット"""
        for cache in self.conv_caches.values():
            cache.zero_()


# ユーティリティ関数
def measure_latency(
    pipeline: StreamingPipeline,
    test_text: str = "これはレイテンシ測定用のテストテキストです。",
    num_iterations: int = 10,
) -> Dict[str, float]:
    """レイテンシを測定"""
    latencies = []
    
    for _ in range(num_iterations):
        start = time.time()
        _ = pipeline.process_text_chunk(test_text)
        latency = (time.time() - start) * 1000  # ミリ秒
        latencies.append(latency)
        
    return {
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
    }


# Base64とJSONのインポート
import base64
import json