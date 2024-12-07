from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class CustomTransformerConfig:
    vocab_size: int


class Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[CustomTransformerConfig] = None,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # Copied from transformers.models.bart.modeling_bart.BartAttention._shape with BART->whisper
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor):
        """Input shape: [batch_size, seq_len, embed_dim]"""
        bsz, tgt_len, _ = hidden_states.shape
        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)


class FlashAttention2(nn.Module):
    pass


ATTENTION_CLASSES = {
    "eager": Attention,
    "flash_attention_2": FlashAttention2,
}


class EncoderLayer(nn.Module):
    def __init__(self, conf: CustomTransformerConfig):
        super().__init__()

    def forward(self, x):
        pass
