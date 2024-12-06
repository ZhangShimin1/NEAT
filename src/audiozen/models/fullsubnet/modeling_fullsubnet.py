from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from simple_parsing import Serializable
from torch.nn import functional

from audiozen.acoustics.audio_feature import stft


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


@dataclass
class ModelArgs(Serializable):
    n_fft: int = 512
    hop_length: int = 256
    win_length: int = 512

    num_freqs: int = 257
    look_ahead: int = 2
    fb_num_neighbors: int = 0
    fb_model_hidden_size: int = 512
    fb_num_layers: int = 2
    fb_output_activate_function: str = "ReLU"
    sb_num_neighbors: int = 15
    sb_output_activate_function: str = "ReLU"
    sb_model_hidden_size: int = 384

    patch_merge_layer_index: Optional[int] = None
    hidden_size: int = 512
    num_freq_out = 2


class FreqMerger(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.scale = args.hidden_size**-0.5
        self.norm = nn.LayerNorm(args.hidden_size)
        self.queries = nn.Parameter(torch.randn(args.num_freq_out, args.hidden_size))

    def forward(self, x):
        # Normalize the input
        x = self.norm(x)  # [B, T, F_in, H]

        # Compute the similarity between the input and the queries
        sim = self.queries @ x.t()  # [F_out, H] * [B, T, H, F_in] = [B, T, F_out, F_in]
        sim *= self.scale

        # Compute the attention
        attn = torch.softmax(sim, dim=-1)  # [B, T, F_out, F_in]

        return attn @ x  # [B, T, F_out, F_in] @ [B, T, F_in, H] = [B, T, F_out, H]


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.fb_layer_norm = nn.LayerNorm(args.num_freqs)

        self.fb_rnn = nn.LSTM(
            input_size=args.num_freqs,
            hidden_size=args.fb_model_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            proj_size=args.num_freqs,
        )

        self.sb_layer_norm = nn.LayerNorm((args.sb_num_neighbors * 2 + 1) + (args.fb_num_neighbors * 2 + 1))

        self.sb_rnn = nn.LSTM(
            input_size=(args.sb_num_neighbors * 2 + 1) + (args.fb_num_neighbors * 2 + 1),
            hidden_size=args.sb_model_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            proj_size=2,
        )

        self.patch_merge_layer_index = (
            default(args.patch_merge_layer_index, 4 // 2) - 1
        )  # default to mid-way through transformer, as shown in paper

        self.stft = partial(stft, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length)

        self.args = args

    @staticmethod
    def freq_unfold_3d(input, num_neighbors):
        """Split overlapped subband units from the frequencies axis.

        Args:
            input: three-dimension input with the shape [B, T, F]
            num_neighbors: number of neighbors in each side for each subband unit.

        Returns:
            Overlapped sub-band units specified as [B, N, T, F_s], where `F_s` represents the frequency axis of
            each sub-band unit and `N` is the number of sub-band unit, e.g., [2, 161, 1, 100, 19].
        """
        assert input.dim() == 3, f"The dim of the input is {input.dim()}. It should be three dim."

        # Convert to batched image-like tensors
        input = rearrange(input, "b t f -> b 1 f t")
        batch_size, num_channels, num_freqs, num_frames = input.size()

        # No change to the input if the number of neighbors is less than or equal to 0
        if num_neighbors <= 0:
            return rearrange(input, "b 1 f t -> b f t 1")  # where `f` is equal to the number of sub-band units

        # Pad the top and bottom of the original spectrogram
        output = functional.pad(input, [0, 0, num_neighbors, num_neighbors], mode="reflect")  # [B * C, 1, F, T]

        # Unfold the spectrogram into sub-band units by sliding along the frequency axis
        # [B * C, 1, F, T] => [B * C, sub_band_unit_size, num_frames, N], N is equal to the number of frequencies.
        sub_band_unit_size = num_neighbors * 2 + 1
        output = functional.unfold(output, kernel_size=(sub_band_unit_size, num_frames))
        assert output.shape[-1] == num_freqs, f"n_freqs != N (sub_band), {num_freqs} != {output.shape[-1]}"

        # Split the dimension of the unfolded feature
        output = rearrange(output, "b (fs t) n -> b n t fs", fs=sub_band_unit_size)

        return output

    def forward(self, input_features):
        """
        Args:
            noisy_mag: noisy magnitude spectrogram

        Returns:
            The real part and imag part of the enhanced spectrogram

        Shapes:
            noisy_mag: [B, 1, F, T]
            return: [B, 2, F, T]
        """
        assert input_features.dim() == 4, f"The dim of the input is {input_features.dim()}. It should be four dim."
        input_features = functional.pad(input_features, [0, self.args.look_ahead])  # Pad the look ahead
        batch_size, num_channels, num_freqs, num_frames = input_features.size()
        assert num_channels == 1, f"{self.__class__.__name__} takes the mag feature as inputs."

        # Convert to channel last format
        input_features = rearrange(input_features, "b c f t -> b t (c f)")

        # Fullband model
        fb_input = self.fb_layer_norm(input_features)
        fb_output, _ = self.fb_rnn(fb_input)  # [B, T, F]

        # Unfold fullband model's output, [B, N=F, T, F_s]. N is the number of sub-band units
        fb_output_unfolded = self.freq_unfold_3d(fb_output, num_neighbors=self.args.fb_num_neighbors)
        # Unfold noisy spectrogram, [B, N=F, T, F_s]
        noisy_mag_unfolded = self.freq_unfold_3d(input_features, num_neighbors=self.args.sb_num_neighbors)

        # Concatenation, [B, F, T, (F_s + F_f)]
        sb_input = torch.cat([noisy_mag_unfolded, fb_output_unfolded], dim=-1)
        sb_input = self.sb_layer_norm(sb_input)
        sb_input = rearrange(sb_input, "b f t fs -> (b f) t fs")
        sb_mask, _ = self.sb_rnn(sb_input)  #  [(B * F), T, 2]
        sb_mask = rearrange(sb_mask, "(b f) t ri -> b ri f t", b=batch_size, f=num_freqs)

        output = sb_mask[:, :, :, self.args.look_ahead :]
        return output


if __name__ == "__main__":
    with torch.no_grad():
        noisy_mag = torch.rand(1, 1, 257, 63)
        model = Model(ModelArgs())
        print(model(noisy_mag).shape)
