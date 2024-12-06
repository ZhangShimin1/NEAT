import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
from einops import rearrange
from torch import nn

from audiozen.acoustics.audio_feature import istft, stft
from audiozen.models.transformer.configuration_transformer import ModelArgs
from audiozen.models.transformer.lora import LoRALinear


def maybe_lora(args: ModelArgs) -> Union[nn.Linear, LoRALinear]:
    if args.lora is None:
        return nn.Linear
    else:
        return partial(LoRALinear, rank=args.lora.rank, scaling=args.lora.scaling)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    out = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return out


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    # original code is 2 but it should be 3 as the dim of batch is missing
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        MaybeLora = maybe_lora(args)
        self.w1 = MaybeLora(args.dim, args.hidden_dim, bias=False)
        self.w2 = MaybeLora(args.hidden_dim, args.dim, bias=False)
        self.w3 = MaybeLora(args.dim, args.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.head_dim: int = args.head_dim

        self.scale = self.args.head_dim**-0.5

        MaybeLora = maybe_lora(args)
        self.wq = MaybeLora(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = MaybeLora(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wv = MaybeLora(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wo = MaybeLora(args.n_heads * args.head_dim, args.dim, bias=False)

        self.positional_embedding = PositionalEncoding(args.dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seqlen, dim]
        # output: [batch_size, seqlen, dim]
        # N: number of heads
        # D: head_dim
        # T: seqlen

        # We are now having an input with the shape of [B, T, E]
        # Now, we are using the linear layers to project the input into the query, key, and value with the shape of [B, T, N * D]
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # We are now splitting the query, key, and value into N heads with the shape of [B, T, N, D]
        xq = rearrange(xq, "b t (n d) -> b t n d", n=self.n_heads)
        xk = rearrange(xk, "b t (n d) -> b t n d", n=self.n_heads)
        xv = rearrange(xv, "b t (n d) -> b t n d", n=self.n_heads)

        # apply_rotary_emb
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        xq = rearrange(xq, "b t n d -> b n t d", n=self.n_heads)
        xk = rearrange(xk, "b t n d -> b n t d", n=self.n_heads)
        xv = rearrange(xv, "b t n d -> b n t d", n=self.n_heads)

        output = nn.functional.scaled_dot_product_attention(xq, xk, xv, scale=self.scale)
        output = rearrange(output, "b n t d -> b t (n d)")
        return self.wo(output)


class TransformerLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

        self.feed_forward: nn.Module
        self.feed_forward = FeedForward(args=args)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        r = self.attention(self.attention_norm(x), freqs_cis)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(args) for _ in range(args.n_layers)])

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, freqs_cis)
        return x


class TransformerEncoderWrapper(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.encoder = TransformerEncoder(args)

        self._precomputed_freqs_cis: Optional[torch.Tensor] = None

        self.stft = partial(stft, n_fft=512, hop_length=128, win_length=512)
        self.istft = partial(istft, n_fft=512, hop_length=128, win_length=512)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def freqs_cis(self) -> torch.Tensor:
        # We cache freqs_cis but need to take care that it is on the right device
        # and has the right dtype (complex64). The fact that the dtype is different
        # from the module's  dtype means we cannot register it as a buffer
        if self._precomputed_freqs_cis is None:
            # default to 10**6
            theta = self.args.rope_theta or 1000000.0
            self._precomputed_freqs_cis = precompute_freqs_cis(self.args.head_dim, 128_000, theta)

        if self._precomputed_freqs_cis.device != self.device:
            self._precomputed_freqs_cis = self._precomputed_freqs_cis.to(device=self.device)

        return self._precomputed_freqs_cis

    def forward(self, noisy_y: torch.Tensor) -> torch.Tensor:
        # noisy_y: [B, C, T] or [B, T]
        # enh_y: [B, T]
        if noisy_y.dim() == 3:
            assert noisy_y.size(1) == 1, "Only mono audio is supported."
            noisy_y = noisy_y.squeeze(1)

        num_samples = noisy_y.size(-1)
        mag_mix, phase_mix, real_mix, imag_mix = self.stft(noisy_y)

        # seq_len
        positions = torch.arange(0, mag_mix.shape[-1]).to(device=mag_mix.device, dtype=torch.long)
        freqs_cis = self.freqs_cis[positions]

        # remove direct component
        real_mix = real_mix[:, 1:]
        imag_mix = imag_mix[:, 1:]

        feat = torch.cat([real_mix, imag_mix], dim=1)  # [B, 2 * F, T]

        feat = rearrange(feat, "b f t -> b t f")
        mask = self.encoder(feat, freqs_cis)
        mask = rearrange(mask, "b t f -> b f t")
        mask_real = mask[:, :256]
        mask_imag = mask[:, 256:]

        enh_real = real_mix * mask_real - imag_mix * mask_imag
        enh_imag = real_mix * mask_imag + imag_mix * mask_real

        # add direct component
        enh_real = torch.cat([real_mix[:, 0:1], enh_real], dim=1)  # [B, F, T]
        enh_imag = torch.cat([imag_mix[:, 0:1], enh_imag], dim=1)

        enh = torch.complex(enh_real, enh_imag)

        enh = self.istft(enh, length=num_samples, input_type="complex")

        return enh


if __name__ == "__main__":
    args = ModelArgs(dim=512, n_layers=6, head_dim=256, hidden_dim=2048, n_heads=8, norm_eps=1e-6)
    model = TransformerEncoderWrapper(args)
    output = model(torch.rand(2, 1, 16000))
    print(output.shape)
