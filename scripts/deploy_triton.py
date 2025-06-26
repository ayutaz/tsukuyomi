#!/usr/bin/env python3
"""Triton Inference Serverへのデプロイスクリプト"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.bigvgan import BigVGANv2
from src.models.vits import VITS
from src.models.xphonebert import XPhoneBERT
from src.serving.triton_server import TritonModelExporter, create_docker_compose_triton


def deploy_models(checkpoint_path: Path, model_repository: Path):
    """モデルのデプロイ"""
    print("モデルをTriton形式でエクスポートしています...")
    
    # エクスポーターの初期化
    exporter = TritonModelExporter(model_repository)
    
    # チェックポイントの読み込み
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # XPhoneBERTのエクスポート
    if 'model_xphonebert' in checkpoint:
        print("XPhoneBERTをエクスポート中...")
        xphonebert = XPhoneBERT(
            model_name="vinai/xphonebert-base",
            hidden_size=768,
            num_layers=12,
            num_heads=12,
        )
        xphonebert.load_state_dict(checkpoint['model_xphonebert'])
        
        exporter.export_model(
            xphonebert,
            "xphonebert",
            version=1,
            batch_size=32,
            input_shapes={
                'input_ids': [1, 512],
                'attention_mask': [1, 512],
                'language_ids': [1, 1],
            },
            output_shapes={
                'embeddings': [1, 512, 768],
            }
        )
        
    # VITSのエクスポート
    if 'model_acoustic' in checkpoint:
        print("音響モデルをエクスポート中...")
        vits = VITS(
            n_vocab=256,
            n_speakers=100,
            hidden_channels=192,
            inter_channels=192,
        )
        vits.load_state_dict(checkpoint['model_acoustic'])
        
        exporter.export_model(
            vits,
            "vits",
            version=1,
            batch_size=8,
        )
        
    # BigVGANのエクスポート
    if 'model_vocoder' in checkpoint:
        print("ボコーダーをエクスポート中...")
        bigvgan = BigVGANv2(
            num_mels=128,
            upsample_initial_channel=1536,
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        )
        bigvgan.load_state_dict(checkpoint['model_vocoder'])
        
        exporter.export_model(
            bigvgan,
            "bigvgan",
            version=1,
            batch_size=8,
        )
        
    # アンサンブルモデルの作成
    print("アンサンブルモデルを作成中...")
    exporter.export_ensemble(
        "tsukuyomi_tts",
        models=[
            {
                'name': 'xphonebert',
                'inputs': ['input_ids', 'attention_mask', 'language_ids'],
                'outputs': ['embeddings'],
            },
            {
                'name': 'vits',
                'inputs': ['text', 'speaker_id', 'embeddings'],
                'outputs': ['mel'],
            },
            {
                'name': 'bigvgan',
                'inputs': ['mel'],
                'outputs': ['audio'],
            },
        ],
        version=1,
    )
    
    print("エクスポート完了！")
    
    # Docker Composeファイルの生成
    create_docker_compose_triton()
    
    print("\nTritonサーバーを起動するには以下のコマンドを実行してください:")
    print("docker-compose -f docker-compose.triton.yml up -d")
    
    return True


def test_deployment(triton_url: str = "localhost:8001"):
    """デプロイメントのテスト"""
    from src.serving.triton_server import TritonTTSClient
    
    print(f"\nTritonサーバー {triton_url} に接続しています...")
    
    client = TritonTTSClient(url=triton_url, protocol="grpc")
    
    # サーバーの準備確認
    if not client.is_server_ready():
        print("エラー: Tritonサーバーが準備できていません")
        return False
        
    print("サーバー接続成功！")
    
    # モデルメタデータの確認
    try:
        metadata = client.get_model_metadata("tsukuyomi_tts")
        print("\nモデルメタデータ:")
        print(f"  名前: {metadata['name']}")
        print(f"  プラットフォーム: {metadata['platform']}")
        print(f"  バージョン: {metadata['versions']}")
    except Exception as e:
        print(f"メタデータ取得エラー: {e}")
        return False
        
    # テスト推論
    try:
        print("\nテスト推論を実行中...")
        test_text = "こんにちは、音声合成のテストです。"
        audio = client.synthesize(test_text, speaker_id=0)
        print(f"音声生成成功！ サンプル数: {len(audio)}")
        
        # バッチ推論テスト
        print("\nバッチ推論テスト中...")
        texts = ["テスト1", "テスト2", "テスト3"]
        audios = client.synthesize_batch(texts, [0, 1, 2])
        print(f"バッチ生成成功！ バッチサイズ: {len(audios)}")
        
    except Exception as e:
        print(f"推論エラー: {e}")
        return False
        
    # ベンチマーク
    print("\nベンチマークを実行中...")
    try:
        results = client.benchmark(num_requests=50)
        print("\nベンチマーク結果:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
    except Exception as e:
        print(f"ベンチマークエラー: {e}")
        
    return True


def main():
    parser = argparse.ArgumentParser(description="Tritonへのモデルデプロイ")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="チェックポイントファイルのパス",
    )
    parser.add_argument(
        "--model-repository",
        type=str,
        default="models",
        help="Tritonモデルリポジトリのパス",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="デプロイ後にテストを実行",
    )
    parser.add_argument(
        "--triton-url",
        type=str,
        default="localhost:8001",
        help="TritonサーバーのURL (テスト時)",
    )
    
    args = parser.parse_args()
    
    # 必要なインポート
    global torch
    import torch
    
    # デプロイの実行
    success = deploy_models(
        Path(args.checkpoint),
        Path(args.model_repository),
    )
    
    if not success:
        print("エラー: デプロイに失敗しました")
        sys.exit(1)
        
    # テストの実行
    if args.test:
        print("\n" + "="*50)
        print("デプロイメントテスト")
        print("="*50)
        
        if not test_deployment(args.triton_url):
            print("エラー: テストに失敗しました")
            sys.exit(1)
            
    print("\n✅ デプロイが完了しました！")


if __name__ == "__main__":
    main()
