#!/usr/bin/env python3
"""正しいフォーマットのメタデータファイルを作成"""

import csv
from pathlib import Path

def create_metadata_from_existing():
    """既存のメタデータから正しいフォーマットを作成"""
    data_root = Path("data/jvs_ljspeech")
    
    # 入力ファイル（問題のあるフォーマット）
    input_files = [
        "metadata_multispeaker.csv",
        "metadata.csv"
    ]
    
    for input_file in input_files:
        input_path = data_root / input_file
        if not input_path.exists():
            continue
            
        print(f"\n処理中: {input_file}")
        
        # 出力ファイル
        output_path = data_root / f"metadata_corrected_{input_file}"
        
        with open(input_path, 'r', encoding='utf-8') as f_in:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for i, line in enumerate(f_in):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split('|')
                    
                    # デバッグ情報
                    if i < 5:
                        print(f"行{i+1}: {len(parts)}カラム")
                        for j, part in enumerate(parts[:4]):
                            print(f"  カラム{j+1}: {part[:30]}...")
                    
                    # フォーマットを判定して修正
                    if len(parts) >= 2:
                        audio_id = parts[0]
                        
                        # ケース1: audio_id|speaker_id|text
                        if len(parts) >= 3 and parts[1].startswith("jvs") and len(parts[1]) <= 10:
                            speaker_id = parts[1]
                            text = parts[2]
                            # スピーカーIDをファイル名に含める形式に変換
                            new_audio_id = f"{speaker_id}_{audio_id.split('_')[-1]}" if '_' in audio_id else audio_id
                            f_out.write(f"{new_audio_id}|{text}|{text}\n")
                            if i < 5:
                                print(f"  → 修正: {new_audio_id}|{text[:20]}...")
                        
                        # ケース2: audio_id|text|normalized_text（正しいフォーマット）
                        elif len(parts) >= 3 and not parts[1].startswith("jvs"):
                            f_out.write(line + "\n")
                        
                        # ケース3: audio_id|text
                        else:
                            text = parts[1]
                            f_out.write(f"{audio_id}|{text}|{text}\n")
        
        print(f"\n修正済みファイルを作成: {output_path}")

if __name__ == "__main__":
    create_metadata_from_existing()