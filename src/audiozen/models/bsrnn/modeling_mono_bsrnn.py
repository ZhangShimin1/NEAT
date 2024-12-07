from functools import partial

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from audiozen.acoustics.audio_feature import istft, stft
from audiozen.models.bsrnn.configuration_mono_bsrnn import ModelArgs


class ResRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True, residual=True):
        super(ResRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.residual = residual
        self.eps = 1e-6

        self.norm = nn.GroupNorm(1, input_size, self.eps)
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=bidirectional)

        # linear projection layer
        self.proj = nn.Linear(hidden_size * (int(bidirectional) + 1), input_size)

    def forward(self, input):
        # input shape: batch, dim, seq

        rnn_output, _ = self.rnn(self.norm(input).transpose(1, 2).contiguous())
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(
            input.shape[0], input.shape[2], input.shape[1]
        )
        rnn_output = rnn_output.transpose(1, 2).contiguous()

        if self.residual:
            return input + rnn_output
        else:
            return rnn_output


class BSNet(nn.Module):
    def __init__(self, in_channel, num_bands=7, num_layer=1):
        super(BSNet, self).__init__()

        self.num_bands = num_bands
        self.feat_dim = in_channel // num_bands

        self.band_rnn = []
        for _ in range(num_layer):
            self.band_rnn.append(ResRNN(self.feat_dim, self.feat_dim * 2, bidirectional=True))
        self.band_rnn = nn.Sequential(*self.band_rnn)
        self.band_comm = ResRNN(self.feat_dim, self.feat_dim * 2, bidirectional=True)

    def forward(self, input):
        if isinstance(input, list):
            input, active_nband, bsnet_layer = (input[0], input[1], input[2])

        # input shape: B, nband*N, T
        B, N, T = input.shape
        # Fuse the tsvad_embedding and subband feature

        input = input.view(B * self.num_bands, self.feat_dim, -1)

        if active_nband is None:
            band_output = self.band_rnn(input).view(B, self.num_bands, -1, T)
            # band comm
            band_output = band_output.permute(0, 3, 2, 1).contiguous().view(B * T, -1, self.num_bands)
            output = self.band_comm(band_output).view(B, T, -1, self.num_bands).permute(0, 3, 2, 1).contiguous()
            return output.view(B, N, T)
        else:
            band_output = self.band_rnn(input).view(B, active_nband, -1, T)

            # band comm
            band_output = band_output.permute(0, 3, 2, 1).contiguous().view(B * T, -1, active_nband)
            output = self.band_comm(band_output).view(B, T, -1, active_nband).permute(0, 3, 2, 1).contiguous()

            return [output.view(B, N, T), active_nband, bsnet_layer + 1]


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super(Model, self).__init__()
        self.args = args

        # 0-1k (100 hop), 1k-4k (250 hop), 4k-8k (500 hop), 8k-16k (1k hop), 16k-20k (2k hop), 20k-inf
        bandwidth_100 = int(np.floor(100 / (args.sr / 2.0) * args.enc_dim))
        bandwidth_250 = int(np.floor(250 / (args.sr / 2.0) * args.enc_dim))
        bandwidth_500 = int(np.floor(500 / (args.sr / 2.0) * args.enc_dim))
        bandwidth_1k = int(np.floor(1000 / (args.sr / 2.0) * args.enc_dim))
        bandwidth_2k = int(np.floor(2000 / (args.sr / 2.0) * args.enc_dim))
        self.band_width = [bandwidth_100] * 10
        self.band_width += [bandwidth_250] * 12
        self.band_width += [bandwidth_500] * 8
        # self.band_width += [bandwidth_1k] * 8
        # self.band_width += [bandwidth_2k] * 2
        self.band_width.append(args.enc_dim - np.sum(self.band_width))
        self.n_band = len(self.band_width)
        print(self.band_width)

        self.eps = 1e-6
        ri_dim = 2

        self.BN = nn.ModuleList([])
        for i in range(self.n_band):
            self.BN.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.band_width[i] * ri_dim * args.num_channels, self.eps),
                    nn.Conv1d(self.band_width[i] * ri_dim * args.num_channels, args.feat_dim, 1),
                )
            )

        self.separator = []
        for i in range(args.num_repeat):
            self.separator.append(BSNet(self.n_band * args.feat_dim, self.n_band, args.num_layer))
        self.separator = nn.Sequential(*self.separator)

        self.mask = nn.ModuleList([])
        for i in range(self.n_band):
            self.mask.append(
                nn.Sequential(
                    nn.GroupNorm(1, args.feat_dim, self.eps),
                    nn.Conv1d(args.feat_dim, args.feat_dim * 2, 1),
                    nn.Tanh(),
                    nn.Conv1d(args.feat_dim * 2, args.feat_dim * 2, 1),
                    nn.Tanh(),
                    nn.Conv1d(args.feat_dim * 2, self.band_width[i] * 4, 1),
                )
            )

        self.args = args
        self.stft = partial(stft, n_fft=512, hop_length=128, win_length=512)
        self.istft = partial(istft, n_fft=512, hop_length=128, win_length=512)

    def forward(self, noisy_y):
        # noisy_y: [B, C, T] or [B, T]
        # enh_y: [B, T]
        if noisy_y.dim() == 2:
            noisy_y = noisy_y.unsqueeze(1)

        nsample = noisy_y.size(-1)
        mag_mix, phase_mix, real_mix, imag_mix = self.stft(noisy_y)  # [B, C, F, T]
        batch_size, num_channels, n_frame = noisy_y.shape

        # Sub-band-wise RNN
        spec_RI = torch.cat([real_mix, imag_mix], 1)  # B, C * ri, F, T
        spec = torch.complex(real_mix, imag_mix)  # B, C, F, T
        subband_spec_RI = []
        subband_spec_complex = []
        band_idx = 0
        for i in range(len(self.band_width)):
            # B, C * ri, Fs, T
            subband_spec_ri_i = spec_RI[:, :, band_idx : band_idx + self.band_width[i]].contiguous()
            # B, Fs, T
            subband_spec_complex_i = spec[:, 0, band_idx : band_idx + self.band_width[i]].contiguous()
            subband_spec_RI.append(subband_spec_ri_i)
            subband_spec_complex.append(subband_spec_complex_i)
            band_idx += self.band_width[i]

        subband_feat = []
        for i in range(len(self.band_width)):
            subband_spec_RI_i = rearrange(subband_spec_RI[i], "b c fs t -> b (c fs) t")  # [B, ri * C * Fs, T]
            subband_feat.append(self.BN[i](subband_spec_RI_i))  # [B, e, T]
        subband_feat = torch.stack(subband_feat, 1)  # [B, n_band, e, T],

        # separator
        # [B, n_band*N, T]
        sep_output = self.separator([rearrange(subband_feat, "b n e t -> b (n e) t"), self.n_band, 0])

        if isinstance(sep_output, list):
            sep_output = sep_output[0]
        sep_output = sep_output.view(batch_size, self.n_band, self.args.feat_dim, -1)

        sep_subband_spec = []
        for i in range(len(self.band_width)):
            this_output = self.mask[i](sep_output[:, i]).view(batch_size, 2, 2, self.band_width[i], -1)
            this_mask = this_output[:, 0] * torch.sigmoid(this_output[:, 1])  # B*nch, 2, BW, T
            this_mask_real = this_mask[:, 0]  # B*nch, BW, T
            this_mask_imag = this_mask[:, 1]  # B*nch, BW, T
            est_spec_real = (
                subband_spec_complex[i].real * this_mask_real - subband_spec_complex[i].imag * this_mask_imag
            )  # B*nch, BW, T
            est_spec_imag = (
                subband_spec_complex[i].real * this_mask_imag + subband_spec_complex[i].imag * this_mask_real
            )  # B*nch, BW, T
            sep_subband_spec.append(torch.complex(est_spec_real, est_spec_imag))

        est_spec = torch.cat(sep_subband_spec, 1)  # B*nch, F, T
        if spec.shape[1] > est_spec.shape[1]:
            est_spec = torch.cat([est_spec, spec[:, est_spec.shape[1] :, :]], 1)

        output = self.istft(est_spec, length=nsample, input_type="complex")

        return output


if __name__ == "__main__":
    from torchinfo import summary

    mixture = torch.rand(2, 1, 16000)
    args = ModelArgs()
    model = Model(args)
    output = model(mixture)

    summary(model, input_size=(2, 1, 16000))
