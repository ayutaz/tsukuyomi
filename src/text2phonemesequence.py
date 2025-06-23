"""
Mock implementation of text2phonemesequence for testing
This is a placeholder until the actual package is available
"""


class Text2PhonemeSequence:
    """Mock Text2PhonemeSequence for testing"""
    
    def __init__(self, language='jpn', is_cuda=False):
        self.language = language
        self.is_cuda = is_cuda
        self.language_map = {
            'jpn': self._japanese_phonemes,
            'eng': self._english_phonemes,
            'eng-us': self._english_phonemes,
            'cmn': self._chinese_phonemes,
        }
    
    def infer_sentence(self, text):
        """Convert text to phoneme sequence"""
        # Use language-specific converter if available
        converter = self.language_map.get(self.language, self._default_phonemes)
        return converter(text)
    
    def _japanese_phonemes(self, text):
        """Mock Japanese phoneme conversion"""
        # Very simple mock - just space-separate characters
        # Real implementation would use proper G2P
        phonemes = []
        for char in text:
            if char == 'は':
                phonemes.append('w a')
            elif char == 'を':
                phonemes.append('o')
            elif char in 'あいうえお':
                phonemes.append(char)
            elif char in 'かきくけこ':
                phonemes.append('k ' + {'か':'a','き':'i','く':'u','け':'e','こ':'o'}[char])
            elif char == ' ' or char == '　':
                continue
            else:
                phonemes.append(char)
        return ' '.join(phonemes)
    
    def _english_phonemes(self, text):
        """Mock English phoneme conversion"""
        # Very simple mock
        return ' '.join(text.lower().replace(',', '').replace('.', '').split())
    
    def _chinese_phonemes(self, text):
        """Mock Chinese phoneme conversion"""
        # Very simple mock
        return ' '.join(text)
    
    def _default_phonemes(self, text):
        """Default phoneme conversion"""
        return ' '.join(text)