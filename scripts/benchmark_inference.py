#!/usr/bin/env python3
"""推論性能ベンチマークスクリプト"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference.optimized_inference import InferenceConfig, OptimizedInferenceEngine


def benchmark_configurations(
    model_path: Path,
    test_texts: List[str],
) -> Dict[str, Dict[str, float]]:
    """異なる設定でのベンチマーク"""
    results = {}

    # テスト設定
    configs = {
        "baseline": InferenceConfig(
            dtype=torch.float32,
            use_compile=False,
            use_graph=False,
        ),
        "fp16": InferenceConfig(
            dtype=torch.float16,
            use_compile=False,
            use_graph=False,
        ),
        "compiled": InferenceConfig(
            dtype=torch.float16,
            use_compile=True,
            use_graph=False,
        ),
        "cuda_graph": InferenceConfig(
            dtype=torch.float16,
            use_compile=True,
            use_graph=True,
        ),
    }

    for name, config in configs.items():
        print(f"\nベンチマーク: {name}")

        # エンジンの初期化
        engine = OptimizedInferenceEngine(model_path, config)

        # ウォームアップ
        for _ in range(5):
            _ = engine.synthesize(test_texts[0])

        # ベンチマーク実行
        metrics = {}

        # レイテンシ測定
        latencies = []
        for text in test_texts:
            start = time.time()
            _, _ = engine.synthesize(text)
            latencies.append(time.time() - start)

        metrics["avg_latency"] = np.mean(latencies)
        metrics["p50_latency"] = np.percentile(latencies, 50)
        metrics["p95_latency"] = np.percentile(latencies, 95)
        metrics["p99_latency"] = np.percentile(latencies, 99)

        # スループット測定
        start = time.time()
        for _ in range(10):
            _ = engine.synthesize_batch(test_texts[:32], list(range(32)))
        total_time = time.time() - start
        metrics["throughput"] = (10 * 32) / total_time

        # メモリ使用量
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            metrics["gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()

        results[name] = metrics

        # クリーンアップ
        del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def benchmark_batch_sizes(
    model_path: Path,
    test_texts: List[str],
) -> Dict[int, Dict[str, float]]:
    """バッチサイズごとのベンチマーク"""
    results = {}

    config = InferenceConfig(
        dtype=torch.float16,
        use_compile=True,
    )
    engine = OptimizedInferenceEngine(model_path, config)

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    for batch_size in batch_sizes:
        if batch_size > len(test_texts):
            continue

        print(f"\nバッチサイズ {batch_size} のベンチマーク")

        texts = test_texts[:batch_size]
        speaker_ids = list(range(batch_size))

        # ウォームアップ
        _ = engine.synthesize_batch(texts, speaker_ids)

        # 測定
        times = []
        for _ in range(20):
            start = time.time()
            _ = engine.synthesize_batch(texts, speaker_ids)
            times.append(time.time() - start)

        metrics = {
            "avg_time": np.mean(times),
            "samples_per_sec": batch_size / np.mean(times),
            "time_per_sample": np.mean(times) / batch_size,
        }

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            metrics["gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024

        results[batch_size] = metrics

    return results


def generate_test_texts(num_texts: int = 100) -> List[str]:
    """テスト用テキストの生成"""
    texts = [
        "こんにちは、今日はいい天気ですね。",
        "音声合成技術は日々進化しています。",
        "最新の深層学習モデルを使用することで、より自然な音声を生成できます。",
        "この文章は少し長めですが、問題なく処理できるはずです。どうでしょうか。",
        "短い文。",
        "AIアシスタントは様々な場面で活用されており、私たちの生活をより便利にしています。",
        "天気予報によると、明日は雨が降るそうです。傘を忘れずに持って行きましょう。",
        "新しいレストランがオープンしました。とても美味しいと評判です。",
        "プログラミングは楽しいですね。特にPythonは初心者にも優しい言語です。",
        "週末は友達と映画を見に行く予定です。最新作が楽しみです。",
    ]

    # 追加のランダムテキスト
    import random

    templates = [
        "これは{}番目のテストテキストです。",
        "サンプル番号{}の音声を生成しています。",
        "ベンチマークテスト{}を実行中です。",
    ]

    while len(texts) < num_texts:
        template = random.choice(templates)
        texts.append(template.format(len(texts) + 1))

    return texts[:num_texts]


def print_results(results: Dict):
    """結果の表示"""
    print("\n" + "=" * 60)
    print("ベンチマーク結果")
    print("=" * 60)

    for key, metrics in results.items():
        print(f"\n[{key}]")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")


def main():
    parser = argparse.ArgumentParser(description="推論性能ベンチマーク")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="モデルチェックポイントのパス",
    )
    parser.add_argument(
        "--num-texts",
        type=int,
        default=100,
        help="テストテキスト数",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="結果の保存先（JSON）",
    )
    parser.add_argument(
        "--benchmark-type",
        type=str,
        choices=["config", "batch", "all"],
        default="all",
        help="ベンチマークタイプ",
    )

    args = parser.parse_args()

    # テストテキストの生成
    test_texts = generate_test_texts(args.num_texts)

    # ベンチマーク実行
    all_results = {}

    if args.benchmark_type in ["config", "all"]:
        print("設定別ベンチマークを実行中...")
        config_results = benchmark_configurations(Path(args.model), test_texts)
        all_results["configurations"] = config_results
        print_results({"設定別結果": config_results})

    if args.benchmark_type in ["batch", "all"]:
        print("\nバッチサイズ別ベンチマークを実行中...")
        batch_results = benchmark_batch_sizes(Path(args.model), test_texts)
        all_results["batch_sizes"] = batch_results
        print_results({"バッチサイズ別結果": batch_results})

    # 結果の保存
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n結果を {args.output} に保存しました。")

    # サマリー
    print("\n" + "=" * 60)
    print("サマリー")
    print("=" * 60)

    if "configurations" in all_results:
        baseline = all_results["configurations"]["baseline"]["avg_latency"]
        for name, metrics in all_results["configurations"].items():
            speedup = baseline / metrics["avg_latency"]
            print(f"{name}: {speedup:.2f}x 高速化")

    if "batch_sizes" in all_results:
        print("\nバッチ処理効率:")
        for size, metrics in sorted(all_results["batch_sizes"].items()):
            print(
                f"  バッチサイズ {size}: {metrics['samples_per_sec']:.1f} samples/sec"
            )


if __name__ == "__main__":
    main()
