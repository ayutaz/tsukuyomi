"""日本語テキスト前処理ユーティリティ"""

import re

import MeCab
import pykakasi


class JapaneseTextPreprocessor:
    """日本語テキストの前処理を行うクラス"""

    def __init__(self, use_mecab: bool = True):
        """
        初期化

        Args:
            use_mecab: MeCabを使用するか（Falseの場合はpykakasiを使用）
        """
        self.use_mecab = use_mecab

        if use_mecab:
            try:
                # MeCab with IPADIC
                self.mecab = MeCab.Tagger("-Oyomi")
            except:
                # Fallback to default
                self.mecab = MeCab.Tagger()
                self.use_mecab = False

        if not use_mecab:
            # pykakasi as fallback
            self.kakasi = pykakasi.kakasi()
            self.kakasi.setMode("H", "K")  # ひらがな → カタカナ
            self.kakasi.setMode("J", "K")  # 漢字 → カタカナ
            self.kakasi.setMode("r", "Hepburn")  # ローマ字はヘボン式
            self.converter = self.kakasi.getConverter()

    def text_to_katakana(self, text: str) -> str:
        """
        テキストをカタカナに変換

        Args:
            text: 入力テキスト

        Returns:
            カタカナ変換されたテキスト
        """
        # 前処理：記号の正規化
        text = self._normalize_symbols(text)

        if self.use_mecab:
            try:
                # MeCabで読みを取得
                result = self.mecab.parse(text)
                # 最後の改行とEOSを除去
                katakana = result.strip().replace("EOS", "")
                return katakana
            except:
                pass

        # pykakasiを使用
        return self.converter.do(text)

    def _normalize_symbols(self, text: str) -> str:
        """記号の正規化"""
        # 句読点を統一
        text = text.replace("。", "。")
        text = text.replace("、", "、")
        text = text.replace("！", "！")
        text = text.replace("？", "？")

        # 英数字を全角に
        text = text.translate(
            str.maketrans(
                "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
                "０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ",
            )
        )

        return text

    def preprocess_for_tts(self, text: str) -> str:
        """
        TTS用のテキスト前処理

        Args:
            text: 入力テキスト

        Returns:
            前処理されたテキスト
        """
        # カタカナに変換
        text = self.text_to_katakana(text)

        # 長音記号の正規化
        text = re.sub(r"ー+", "ー", text)

        # 連続する句読点を削除
        text = re.sub(r"。+", "。", text)
        text = re.sub(r"、+", "、", text)

        # 空白の正規化
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text


# シンプルなカタカナ変換（外部ライブラリなし）
def simple_text_to_katakana(text: str) -> str:
    """
    シンプルなカタカナ変換（ひらがな→カタカナのみ）
    外部ライブラリが使えない場合の簡易版

    Args:
        text: 入力テキスト

    Returns:
        カタカナ変換されたテキスト（ひらがなのみ変換、漢字はそのまま）
    """
    # ひらがな→カタカナ変換テーブル
    hiragana = "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをん"
    katakana = "ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲン"

    trans_table = str.maketrans(hiragana, katakana)
    return text.translate(trans_table)
