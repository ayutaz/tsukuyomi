#!/usr/bin/env python3
"""テキストトークナイザーのテストスクリプト"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.text_tokenizer import JapaneseTextTokenizer


def main():
    """トークナイザーのテスト"""
    tokenizer = JapaneseTextTokenizer()
    
    print(f"語彙サイズ: {len(tokenizer)}")
    print(f"特殊トークン: PAD={tokenizer.pad_id}, UNK={tokenizer.unk_id}, "
          f"BOS={tokenizer.bos_id}, EOS={tokenizer.eos_id}")
    
    # テストテキスト（JVSデータセットのサンプル）
    test_texts = [
        "コンニチハ、キョーワイイテンキデスネ。",
        "オンセーゴーセーノケンキューヲシテイマス。",
        "ニホンゴノオンセーヲセーセーシマス。",
        "マタ、トージノヨーニ、ゴダイアキラオートヨバレル、シュヨーナミョーオーノチューオーニハイサレルコトモオーイ。"
    ]
    
    print("\n=== 単一テキストのエンコード ===")
    for text in test_texts[:2]:
        print(f"\n入力: {text}")
        encoding = tokenizer.encode(text, add_special_tokens=True, max_length=50)
        print(f"トークン: {encoding['tokens'][:20]}...")  # 最初の20トークンのみ表示
        print(f"トークンID: {encoding['input_ids'][:20]}...")
        print(f"長さ: {encoding['length']}")
        
        # デコード
        decoded = tokenizer.decode(encoding['input_ids'])
        print(f"デコード: {decoded}")
    
    print("\n=== バッチエンコード ===")
    batch_encoding = tokenizer.batch_encode(
        test_texts[:3],
        add_special_tokens=True,
        max_length=100,
        padding=True
    )
    print(f"バッチshape: {batch_encoding['input_ids'].shape}")
    print(f"長さ: {batch_encoding['lengths']}")
    
    # 各テキストをデコード
    for i, ids in enumerate(batch_encoding['input_ids']):
        decoded = tokenizer.decode(ids)
        print(f"デコード[{i}]: {decoded}")


if __name__ == "__main__":
    main()