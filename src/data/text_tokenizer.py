"""
Japanese text tokenizer for TTS
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class JapaneseTextTokenizer:
    """日本語テキストトークナイザー
    
    カタカナテキストを音素列に変換し、トークンIDに変換する
    """
    
    def __init__(self, vocab_file: Optional[Path] = None):
        """
        Args:
            vocab_file: 語彙ファイルのパス
        """
        self.vocab = self._build_default_vocab()
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        
        # 特殊トークン
        self.pad_id = self.token_to_id["<pad>"]
        self.unk_id = self.token_to_id["<unk>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]
        
        logger.info(f"Initialized tokenizer with vocab size: {len(self.vocab)}")
        
    def _build_default_vocab(self) -> List[str]:
        """デフォルトの語彙を構築"""
        vocab = ["<pad>", "<unk>", "<bos>", "<eos>"]
        
        # カタカナ
        katakana = [
            "ア", "イ", "ウ", "エ", "オ",
            "カ", "キ", "ク", "ケ", "コ",
            "ガ", "ギ", "グ", "ゲ", "ゴ",
            "サ", "シ", "ス", "セ", "ソ",
            "ザ", "ジ", "ズ", "ゼ", "ゾ",
            "タ", "チ", "ツ", "テ", "ト",
            "ダ", "ヂ", "ヅ", "デ", "ド",
            "ナ", "ニ", "ヌ", "ネ", "ノ",
            "ハ", "ヒ", "フ", "ヘ", "ホ",
            "バ", "ビ", "ブ", "ベ", "ボ",
            "パ", "ピ", "プ", "ペ", "ポ",
            "マ", "ミ", "ム", "メ", "モ",
            "ヤ", "ユ", "ヨ",
            "ラ", "リ", "ル", "レ", "ロ",
            "ワ", "ヲ", "ン",
            "ッ", "ー", "ャ", "ュ", "ョ",
            "ァ", "ィ", "ゥ", "ェ", "ォ"
        ]
        
        # 記号
        symbols = ["、", "。", "！", "？", "・", "「", "」", " "]
        
        vocab.extend(katakana)
        vocab.extend(symbols)
        
        return vocab
    
    def tokenize(self, text: str) -> List[str]:
        """テキストをトークンに分割"""
        # カタカナテキストを1文字ずつ分割
        tokens = []
        for char in text:
            if char in self.token_to_id:
                tokens.append(char)
            else:
                tokens.append("<unk>")
        return tokens
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        return_tensors: bool = True
    ) -> Dict[str, torch.Tensor]:
        """テキストをトークンIDに変換
        
        Args:
            text: 入力テキスト
            add_special_tokens: 特殊トークンを追加するか
            max_length: 最大長
            padding: パディングするか
            return_tensors: テンソルで返すか
            
        Returns:
            トークンIDとその他の情報
        """
        # トークン化
        tokens = self.tokenize(text)
        
        # 特殊トークンの追加
        if add_special_tokens:
            tokens = ["<bos>"] + tokens + ["<eos>"]
            
        # トークンIDに変換
        token_ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        
        # 最大長で切り詰め
        if max_length is not None:
            token_ids = token_ids[:max_length]
            
        # パディング
        length = len(token_ids)
        if padding and max_length is not None:
            pad_length = max_length - len(token_ids)
            if pad_length > 0:
                token_ids = token_ids + [self.pad_id] * pad_length
                
        # テンソルに変換
        if return_tensors:
            token_ids = torch.tensor(token_ids, dtype=torch.long)
            length = torch.tensor(length, dtype=torch.long)
            
        return {
            "input_ids": token_ids,
            "length": length,
            "tokens": tokens
        }
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """トークンIDをテキストに変換"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        tokens = []
        for token_id in token_ids:
            if token_id == self.pad_id and skip_special_tokens:
                break
            token = self.id_to_token.get(token_id, "<unk>")
            if skip_special_tokens and token in ["<bos>", "<eos>", "<pad>"]:
                continue
            tokens.append(token)
            
        return "".join(tokens)
    
    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        return_tensors: bool = True
    ) -> Dict[str, torch.Tensor]:
        """複数のテキストを一括エンコード"""
        batch_encoding = []
        lengths = []
        
        # 各テキストをエンコード
        for text in texts:
            encoding = self.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=False,
                return_tensors=False
            )
            batch_encoding.append(encoding["input_ids"])
            lengths.append(encoding["length"])
            
        # パディング用の最大長を決定
        if max_length is None:
            max_length = max(len(ids) for ids in batch_encoding)
            
        # パディング
        if padding:
            padded_encodings = []
            for ids in batch_encoding:
                pad_length = max_length - len(ids)
                if pad_length > 0:
                    ids = ids + [self.pad_id] * pad_length
                padded_encodings.append(ids)
            batch_encoding = padded_encodings
            
        # テンソルに変換
        if return_tensors:
            batch_encoding = torch.tensor(batch_encoding, dtype=torch.long)
            lengths = torch.tensor(lengths, dtype=torch.long)
            
        return {
            "input_ids": batch_encoding,
            "lengths": lengths
        }
    
    def __len__(self) -> int:
        """語彙サイズ"""
        return len(self.vocab)
