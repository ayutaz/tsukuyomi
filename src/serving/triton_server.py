"""Triton Inference Server統合

高性能な本番環境向けモデルサービング
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException, triton_to_np_dtype

from ..models.vits import VITS
from ..models.matcha_tts import MatchaTTS
from ..models.bigvgan import BigVGANv2
from ..models.xphonebert import XPhoneBERT
from ..models.f0_bert import F0BERT


class TritonModelExporter:
    """Triton用モデルエクスポーター"""
    
    def __init__(self, model_repository: Path):
        self.model_repository = Path(model_repository)
        self.model_repository.mkdir(parents=True, exist_ok=True)
        
    def export_model(
        self,
        model: nn.Module,
        model_name: str,
        version: int = 1,
        batch_size: int = 8,
        input_shapes: Dict[str, List[int]] = None,
        output_shapes: Dict[str, List[int]] = None,
    ):
        """モデルをTriton形式でエクスポート"""
        model_dir = self.model_repository / model_name / str(version)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # モデルの保存形式を決定
        if isinstance(model, (VITS, MatchaTTS, BigVGANv2)):
            # PyTorchモデルとして保存
            self._export_pytorch_model(model, model_dir, model_name)
            backend = "pytorch"
        else:
            # ONNXとして保存
            self._export_onnx_model(
                model, model_dir, model_name,
                input_shapes, batch_size
            )
            backend = "onnxruntime"
            
        # config.pbtxtの生成
        self._generate_config(
            model_name, backend, version,
            input_shapes, output_shapes, batch_size
        )
        
    def _export_pytorch_model(self, model: nn.Module, model_dir: Path, model_name: str):
        """PyTorchモデルのエクスポート"""
        # TorchScriptに変換
        model.eval()
        
        # モデルごとのダミー入力
        if isinstance(model, VITS):
            dummy_input = (
                torch.randint(0, 100, (1, 50)),  # text
                torch.tensor([0]),  # speaker_id
            )
            traced_model = torch.jit.trace(model.infer, dummy_input)
        elif isinstance(model, BigVGANv2):
            dummy_input = torch.randn(1, 128, 100)  # mel spectrogram
            traced_model = torch.jit.trace(model, dummy_input)
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
            
        # 保存
        traced_model.save(str(model_dir / "model.pt"))
        
    def _export_onnx_model(
        self,
        model: nn.Module,
        model_dir: Path,
        model_name: str,
        input_shapes: Dict[str, List[int]],
        batch_size: int,
    ):
        """ONNXモデルのエクスポート"""
        import torch.onnx
        
        model.eval()
        
        # ダミー入力の作成
        dummy_inputs = {}
        for name, shape in input_shapes.items():
            # バッチサイズを設定
            shape_with_batch = [batch_size] + shape[1:]
            if 'ids' in name or 'mask' in name:
                dummy_inputs[name] = torch.randint(0, 100, shape_with_batch)
            else:
                dummy_inputs[name] = torch.randn(shape_with_batch)
                
        # ONNX変換
        output_path = model_dir / "model.onnx"
        
        # 動的軸の設定
        dynamic_axes = {}
        for name in input_shapes:
            dynamic_axes[name] = {0: 'batch_size'}
            
        torch.onnx.export(
            model,
            tuple(dummy_inputs.values()),
            output_path,
            input_names=list(input_shapes.keys()),
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
        )
        
    def _generate_config(
        self,
        model_name: str,
        backend: str,
        version: int,
        input_shapes: Dict[str, List[int]],
        output_shapes: Dict[str, List[int]],
        batch_size: int,
    ):
        """Triton設定ファイルの生成"""
        config_path = self.model_repository / model_name / "config.pbtxt"
        
        config = f"""
name: "{model_name}"
backend: "{backend}"
max_batch_size: {batch_size}

"""
        
        # 入力設定
        if input_shapes:
            for name, shape in input_shapes.items():
                # データ型の推定
                if 'ids' in name or 'mask' in name:
                    dtype = "TYPE_INT64"
                else:
                    dtype = "TYPE_FP32"
                    
                config += f"""
input [
  {{
    name: "{name}"
    data_type: {dtype}
    dims: {shape[1:]}
  }}
]
"""
        
        # 出力設定
        if output_shapes:
            for name, shape in output_shapes.items():
                config += f"""
output [
  {{
    name: "{name}"
    data_type: TYPE_FP32
    dims: {shape[1:]}
  }}
]
"""
        
        # 最適化設定
        config += """
optimization {
  graph {
    level: 1
  }
}

instance_group [
  {
    count: 2
    kind: KIND_GPU
  }
]

dynamic_batching {
  max_queue_delay_microseconds: 100
}
"""
        
        with open(config_path, 'w') as f:
            f.write(config)
            
    def export_ensemble(
        self,
        ensemble_name: str,
        models: List[Dict[str, Union[str, List[str]]]],
        version: int = 1,
    ):
        """アンサンブルモデルのエクスポート"""
        ensemble_dir = self.model_repository / ensemble_name
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        
        config = f"""
name: "{ensemble_name}"
platform: "ensemble"
max_batch_size: 8

"""
        
        # 入力マッピング
        for i, model_info in enumerate(models):
            if i == 0:
                # 最初のモデルの入力
                for input_name in model_info.get('inputs', ['input']):
                    config += f"""
input [
  {{
    name: "{input_name}"
    data_type: TYPE_FP32
    dims: [-1]
  }}
]
"""
        
        # モデルチェーン
        for i, model_info in enumerate(models):
            model_name = model_info['name']
            
            config += f"""
ensemble_scheduling {{
  step [
    {{
      model_name: "{model_name}"
      model_version: -1
"""
            
            # 入力マッピング
            if i == 0:
                for input_name in model_info.get('inputs', ['input']):
                    config += f"""
      input_map {{
        key: "{input_name}"
        value: "{input_name}"
      }}
"""
            else:
                # 前のモデルの出力を入力として使用
                prev_model = models[i-1]['name']
                for j, input_name in enumerate(model_info.get('inputs', ['input'])):
                    output_name = models[i-1].get('outputs', ['output'])[j]
                    config += f"""
      input_map {{
        key: "{input_name}"
        value: "{prev_model}_{output_name}"
      }}
"""
            
            # 出力マッピング
            for output_name in model_info.get('outputs', ['output']):
                config += f"""
      output_map {{
        key: "{output_name}"
        value: "{model_name}_{output_name}"
      }}
"""
            
            config += """
    }
  ]
}
"""
        
        # 最終出力
        last_model = models[-1]['name']
        for output_name in models[-1].get('outputs', ['output']):
            config += f"""
output [
  {{
    name: "{output_name}"
    data_type: TYPE_FP32
    dims: [-1]
  }}
]
"""
        
        with open(ensemble_dir / "config.pbtxt", 'w') as f:
            f.write(config)


class TritonTTSClient:
    """Triton Inference Serverクライアント"""
    
    def __init__(
        self,
        url: str = "localhost:8001",
        protocol: str = "grpc",
        verbose: bool = False,
    ):
        self.url = url
        self.protocol = protocol
        self.verbose = verbose
        
        # クライアントの初期化
        if protocol == "grpc":
            self.client = grpcclient.InferenceServerClient(url=url, verbose=verbose)
        else:
            self.client = httpclient.InferenceServerClient(url=url, verbose=verbose)
            
    def is_server_ready(self) -> bool:
        """サーバーの準備状態を確認"""
        try:
            return self.client.is_server_ready()
        except InferenceServerException:
            return False
            
    def get_model_metadata(self, model_name: str) -> Dict:
        """モデルメタデータの取得"""
        try:
            metadata = self.client.get_model_metadata(model_name)
            return {
                'name': metadata.name,
                'versions': metadata.versions,
                'platform': metadata.platform,
                'inputs': metadata.inputs,
                'outputs': metadata.outputs,
            }
        except InferenceServerException as e:
            raise RuntimeError(f"Failed to get model metadata: {e}")
            
    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        model_name: str = "tsukuyomi_tts",
    ) -> np.ndarray:
        """音声合成の実行"""
        # 入力の準備
        text_encoded = self._encode_text(text)
        
        # 入力テンソルの作成
        inputs = []
        
        if self.protocol == "grpc":
            inputs.append(grpcclient.InferInput('text', text_encoded.shape, "INT64"))
            inputs[0].set_data_from_numpy(text_encoded)
            
            inputs.append(grpcclient.InferInput('speaker_id', [1, 1], "INT64"))
            inputs[1].set_data_from_numpy(np.array([[speaker_id]], dtype=np.int64))
            
            # 出力の準備
            outputs = [grpcclient.InferRequestedOutput('audio')]
        else:
            inputs.append(httpclient.InferInput('text', text_encoded.shape, "INT64"))
            inputs[0].set_data_from_numpy(text_encoded)
            
            inputs.append(httpclient.InferInput('speaker_id', [1, 1], "INT64"))
            inputs[1].set_data_from_numpy(np.array([[speaker_id]], dtype=np.int64))
            
            outputs = [httpclient.InferRequestedOutput('audio')]
            
        # 推論実行
        try:
            response = self.client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs,
            )
            
            # 結果の取得
            audio = response.as_numpy('audio')
            return audio.squeeze()
            
        except InferenceServerException as e:
            raise RuntimeError(f"Inference failed: {e}")
            
    def synthesize_batch(
        self,
        texts: List[str],
        speaker_ids: List[int],
        model_name: str = "tsukuyomi_tts",
    ) -> List[np.ndarray]:
        """バッチ音声合成"""
        batch_size = len(texts)
        
        # テキストのエンコード
        text_batch = []
        max_len = 0
        
        for text in texts:
            encoded = self._encode_text(text)
            text_batch.append(encoded)
            max_len = max(max_len, len(encoded))
            
        # パディング
        padded_texts = np.zeros((batch_size, max_len), dtype=np.int64)
        for i, encoded in enumerate(text_batch):
            padded_texts[i, :len(encoded)] = encoded
            
        # 入力テンソルの作成
        inputs = []
        
        if self.protocol == "grpc":
            inputs.append(grpcclient.InferInput('text', padded_texts.shape, "INT64"))
            inputs[0].set_data_from_numpy(padded_texts)
            
            speaker_array = np.array(speaker_ids, dtype=np.int64).reshape(-1, 1)
            inputs.append(grpcclient.InferInput('speaker_id', speaker_array.shape, "INT64"))
            inputs[1].set_data_from_numpy(speaker_array)
            
            outputs = [grpcclient.InferRequestedOutput('audio')]
        else:
            inputs.append(httpclient.InferInput('text', padded_texts.shape, "INT64"))
            inputs[0].set_data_from_numpy(padded_texts)
            
            speaker_array = np.array(speaker_ids, dtype=np.int64).reshape(-1, 1)
            inputs.append(httpclient.InferInput('speaker_id', speaker_array.shape, "INT64"))
            inputs[1].set_data_from_numpy(speaker_array)
            
            outputs = [httpclient.InferRequestedOutput('audio')]
            
        # 推論実行
        try:
            response = self.client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs,
            )
            
            # 結果の分割
            audio_batch = response.as_numpy('audio')
            return [audio_batch[i] for i in range(batch_size)]
            
        except InferenceServerException as e:
            raise RuntimeError(f"Batch inference failed: {e}")
            
    def _encode_text(self, text: str) -> np.ndarray:
        """テキストのエンコード"""
        # 簡易的な文字エンコーディング
        encoded = [ord(c) % 256 for c in text]
        return np.array(encoded, dtype=np.int64)
        
    def benchmark(self, num_requests: int = 100) -> Dict[str, float]:
        """性能ベンチマーク"""
        import time
        import random
        import string
        
        results = {}
        
        # テストデータ
        texts = []
        for _ in range(num_requests):
            length = random.randint(10, 100)
            text = ''.join(random.choices(string.ascii_letters + ' ', k=length))
            texts.append(text)
            
        # シングルリクエスト
        start = time.time()
        for text in texts:
            _ = self.synthesize(text)
        single_time = time.time() - start
        results['single_avg_latency'] = single_time / num_requests
        results['single_throughput'] = num_requests / single_time
        
        # バッチリクエスト
        batch_sizes = [4, 8, 16, 32]
        for batch_size in batch_sizes:
            if batch_size > num_requests:
                continue
                
            start = time.time()
            for i in range(0, num_requests, batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_speakers = [0] * len(batch_texts)
                _ = self.synthesize_batch(batch_texts, batch_speakers)
            batch_time = time.time() - start
            
            results[f'batch_{batch_size}_avg_latency'] = batch_time / num_requests
            results[f'batch_{batch_size}_throughput'] = num_requests / batch_time
            
        return results


def create_docker_compose_triton():
    """Triton用Docker Composeファイルの生成"""
    compose_content = """version: '3.8'

services:
  triton:
    image: nvcr.io/nvidia/tritonserver:23.10-py3
    container_name: tsukuyomi-triton
    ports:
      - "8000:8000"  # HTTP
      - "8001:8001"  # gRPC
      - "8002:8002"  # Metrics
    volumes:
      - ./models:/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
    command: tritonserver --model-repository=/models --strict-model-config=false
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    
  triton-client:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: tsukuyomi-triton-client
    depends_on:
      - triton
    environment:
      - TRITON_URL=triton:8001
    volumes:
      - ./scripts:/workspace/scripts
      - ./src:/workspace/src
    command: tail -f /dev/null
"""
    
    with open("docker-compose.triton.yml", "w") as f:
        f.write(compose_content)
        
    print("Created docker-compose.triton.yml")


if __name__ == "__main__":
    # Triton用設定ファイルの生成
    create_docker_compose_triton()