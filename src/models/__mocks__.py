"""
Mock implementations for external dependencies that are not available on PyPI
"""


# Mock for text2phonemesequence
class Text2PhonemeSequence:
    def __init__(self, language="jpn", is_cuda=False):
        self.language = language
        self.is_cuda = is_cuda

    def infer_sentence(self, text):
        # Simple mock implementation
        return " ".join(list(text.replace(" ", "")))


# Mock for bigvgan
class BigVGAN:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, model_name, use_cuda_kernel=False):
        instance = cls()
        return instance

    def remove_weight_norm(self):
        pass

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, mel):
        import torch

        # Return mock waveform with appropriate size
        batch_size = mel.shape[0]
        time_steps = mel.shape[2] * 256  # hop_size = 256
        return torch.randn(batch_size, time_steps)


class bigvgan:
    BigVGAN = BigVGAN
