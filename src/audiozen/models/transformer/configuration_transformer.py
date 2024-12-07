from dataclasses import dataclass
from typing import Optional

from simple_parsing.helpers import Serializable

from audiozen.models.transformer.lora import LoraArgs


@dataclass
class ModelArgs(Serializable):
    dim: int
    n_layers: int
    n_heads: int
    head_dim: int
    hidden_dim: int
    norm_eps: float

    # For rotary embeddings. If not set, will be inferred
    rope_theta: Optional[float] = None

    lora: Optional[LoraArgs] = None  # If this is set, we will load LoRA linear layers instead of linear layers.
