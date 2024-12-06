from dataclasses import dataclass

from simple_parsing import Serializable


@dataclass
class ModelArgs(Serializable):
    enc_dim: int = 257
    feat_dim: int = 128
    num_channels: int = 1
    sr: int = 16000
    num_layer: int = 1
    num_repeat: int = 6
    dropout: float = 0.0
