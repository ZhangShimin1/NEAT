import importlib
from functools import partial
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor
from torch.autograd import Function

from audiozen.acoustics.audio_feature import istft, stft


def numpy_to_torch(*args):
    out = []
    for arg in args:
        if isinstance(arg, ndarray):
            out.append(torch.from_numpy(arg))
        else:
            out.append(arg)

    return out


class SNRLoss(nn.Module):
    def __init__(self, return_neg=True, eps=1e-8, reduction="mean"):
        super().__init__()
        self.return_neg = return_neg
        self.eps = eps
        self.reduction = reduction

    def forward(self, est: Tensor, ref: Tensor) -> Tensor:
        """Calculate SNR loss.

        Args:
            est: The estimated signal. Shape: [B, T]
            ref: The reference signal. Shape: [B, T]

        Returns:
            The SNR loss value with negative sign. Shape: [B]
        """
        est, ref = numpy_to_torch(est, ref)

        noise = est - ref

        snr = 20 * (
            torch.log10(torch.norm(ref, p=2, dim=1).clamp(min=self.eps))
            - torch.log10(torch.norm(noise, p=2, dim=1).clamp(min=self.eps))
        )

        if self.reduction == "mean":
            snr = snr.mean()
        elif self.reduction == "sum":
            snr = snr.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

        return -snr if self.return_neg else snr


class SISNRLoss(nn.Module):
    def __init__(self, return_neg=True):
        super().__init__()
        self.return_neg = return_neg

    def forward(self, est, ref):
        est, ref = numpy_to_torch(est, ref)

        if est.shape != ref.shape:
            raise RuntimeError(f"Dimension mismatch when calculating SI-SNR, {est.shape=} vs {ref.shape=}")

        eps = torch.finfo(est.dtype).eps
        s_est = est - torch.mean(est, dim=-1, keepdim=True)
        s_ref = ref - torch.mean(ref, dim=-1, keepdim=True)

        # <s, s'> / ||s||**2 * s
        pair_wise_dot = torch.sum(s_ref * s_est, dim=-1, keepdim=True)
        s_ref_norm = torch.sum(s_ref**2, dim=-1, keepdim=True)
        pair_wise_proj = pair_wise_dot * s_ref / s_ref_norm

        e_noise = s_est - pair_wise_proj

        pair_wise_sdr = torch.sum(pair_wise_proj**2, dim=-1) / (torch.sum(e_noise**2, dim=-1) + eps)

        si_snr = torch.mean(10 * torch.log10(pair_wise_sdr + eps))

        return -si_snr if self.return_neg else si_snr


class angle(Function):
    """Similar to torch.angle but robustify the gradient for zero magnitude."""

    @staticmethod
    def forward(ctx, x: Tensor):
        ctx.save_for_backward(x)
        return torch.atan2(x.imag, x.real)

    @staticmethod
    def backward(ctx, grad: Tensor):
        eps = torch.finfo(grad.dtype).eps
        (x,) = ctx.saved_tensors
        grad_inv = grad / (x.real.square() + x.imag.square()).clamp_min_(eps)
        return torch.view_as_complex(torch.stack((-x.imag * grad_inv, x.real * grad_inv), dim=-1))


class MultiResSpecLoss(nn.Module):
    def __init__(self, n_ffts, gamma=1, factor=1, f_complex=None):
        super().__init__()
        self.n_fft_list = n_ffts
        self.gamma = gamma
        self.factor = factor
        self.factor_complex = f_complex

    @staticmethod
    def stft(y, n_fft, hop_length=None, win_length=None):
        num_dims = y.dim()
        assert num_dims == 2, "Only support 2D input."

        hop_length = hop_length or n_fft // 4
        win_length = win_length or n_fft

        batch_size, num_samples = y.size()

        # [B, F, T] in complex-valued
        complex_stft = torch.stft(
            y,
            n_fft,
            hop_length,
            win_length,
            window=torch.hann_window(n_fft, device=y.device),
            return_complex=True,
            normalized=True,
        )

        return complex_stft

    def forward(self, est: Tensor, target: Tensor) -> Tensor:
        eps = torch.finfo(est.dtype).eps
        loss = torch.zeros((), device=est.device, dtype=est.dtype)

        for n_fft in self.n_fft_list:
            Y = self.stft(est, n_fft)
            S = self.stft(target, n_fft)
            Y_abs = Y.abs()
            S_abs = S.abs()
            if self.gamma != 1:
                Y_abs = Y_abs.clamp_min(eps).pow(self.gamma)
                S_abs = S_abs.clamp_min(eps).pow(self.gamma)

            # magnitude loss
            loss += F.mse_loss(Y_abs, S_abs) * self.factor

            # real/imaginary loss
            if self.factor_complex is not None:
                if self.gamma != 1:
                    Y = Y_abs * torch.exp(1j * angle.apply(Y))
                    S = S_abs * torch.exp(1j * angle.apply(S))
                loss += F.mse_loss(torch.view_as_real(Y), torch.view_as_real(S)) * self.factor_complex
        return loss


class CombineLoss(nn.Module):
    def __init__(
        self,
        n_ffts: Iterable[int],
        gamma: float = 1,
        factor: float = 1,
        f_complex: float = None,
    ):
        super().__init__()
        self.n_ffts = n_ffts
        self.gamma = gamma
        self.f = factor
        self.f_complex = f_complex

        self.mulres_loss = MultiResSpecLoss(n_ffts, gamma, factor, f_complex)
        self.l1_loss = nn.L1Loss()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss1 = self.mulres_loss(input, target)
        loss2 = self.l1_loss(input, target)
        return loss1 + loss2


def freq_MAE(estimation, target, win=2048, stride=512, srs=None, sudo_sr=None):
    est_spec = torch.stft(
        estimation.view(-1, estimation.shape[-1]),
        n_fft=win,
        hop_length=stride,
        window=torch.hann_window(win).to(estimation.device).float(),
        return_complex=True,
    )
    est_target = torch.stft(
        target.view(-1, target.shape[-1]),
        n_fft=win,
        hop_length=stride,
        window=torch.hann_window(win).to(estimation.device).float(),
        return_complex=True,
    )

    if srs is None:
        return (est_spec.real - est_target.real).abs().mean() + (est_spec.imag - est_target.imag).abs().mean()
    else:
        loss = 0
        for i, sr in enumerate(srs):
            max_freq = int(est_spec.shape[-2] * sr / sudo_sr)
            loss += (est_spec[i][:max_freq].real - est_target[i][:max_freq].real).abs().mean() + (
                est_spec[i][:max_freq].imag - est_target[i][:max_freq].imag
            ).abs().mean()
        loss = loss / len(srs)
        return loss


def mag_MAE(estimation, target, win=2048, stride=512, srs=None, sudo_sr=None):
    est_spec = torch.stft(
        estimation.view(-1, estimation.shape[-1]),
        n_fft=win,
        hop_length=stride,
        window=torch.hann_window(win).to(estimation.device).float(),
        return_complex=True,
    )
    est_target = torch.stft(
        target.view(-1, target.shape[-1]),
        n_fft=win,
        hop_length=stride,
        window=torch.hann_window(win).to(estimation.device).float(),
        return_complex=True,
    )
    if srs is None:
        return (est_spec.abs() - est_target.abs()).abs().mean()
    else:
        loss = 0
        for i, sr in enumerate(srs):
            max_freq = int(est_spec.shape[-2] * sr / sudo_sr)
            loss += (est_spec[i][:max_freq].abs() - est_target[i][:max_freq].abs()).abs().mean()
        loss = loss / len(srs)
    return loss


class BinauralLoss(nn.Module):
    def __init__(
        self, n_fft, win_length, hop_length, sr, ild_weight=0.1, ipd_weight=1.0, stoi_weight=0, snr_loss_weight=0.1
    ):
        super().__init__()
        self.stft = partial(stft, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        self.istft = partial(istft, win_length=win_length, hop_length=hop_length)

        # dynamically import the torch_stoi library
        torch_stoi = importlib.import_module("torch_stoi")
        self.stoi_loss = torch_stoi.NegSTOILoss(sample_rate=sr)
        self.snr_loss = SNRLoss()

        self.ild_weight = ild_weight
        self.ipd_weight = ipd_weight
        self.stoi_weight = stoi_weight
        self.snr_loss_weight = snr_loss_weight

    def _amplitude_to_db(self, x: Tensor, eps: float = 1e-8):
        # Convert amplitude to dB
        return 20 * torch.log10(x.clamp(min=eps))

    def _ild_in_dB(est: Tensor, ref: Tensor, eps=1e-8) -> Tensor:
        """Calculate ILD in dB.

        Args:
            est: the estimated magnitude spectrogram. Shape: [B, F, T]
            ref: the reference magnitude spectrogram. Shape: [B, F, T]
        """
        est_in_db = 20 * torch.log10(est.clamp(min=eps))
        ref_in_db = 20 * torch.log10(ref.clamp(min=eps))

        return est_in_db - ref_in_db

    def _ipd_in_rad(est: Tensor, ref: Tensor) -> Tensor:
        """Calculate IPD in rad.

        Args:
            est: the estimated STFT spectrogram. Shape: [B, F, T]
            ref: the reference STFT spectrogram. Shape: [B, F, T]
        """
        est_phase = torch.angle(est)
        ref_phase = torch.angle(ref)

        return est_phase - ref_phase

    def _build_energy_mask(self, mag_spec_l, mag_spec_r, threshold_in_dB=20):
        """Build speech mask from magnitude spectrogram.

        Assuming a threshold. For each frequency, if the energy of the left and right channel is above the threshold,
        then the mask is 1, otherwise 0. The mask is computed in dB.

        Args:
            mag_spec_l: The magnitude spectrogram of the left channel. Shape: [B, F, T]
            mag_spec_r: The magnitude spectrogram of the right channel. Shape: [B, F, T]
            threshold: The threshold in dB. Default: 15 dB.

        Returns:
            The speech mask (int). Shape: [B, F, T]
        """
        batch_size, num_freqs, num_frames = mag_spec_l.size()

        # Find the the maximum energy computed for each frequency bin
        threshold_l = torch.max(mag_spec_l**2, dim=2)
        threshold_l_in_dB = 10 * torch.log10(threshold_l) - threshold_in_dB
        threshold_l_in_dB = threshold_l_in_dB.unsqueeze(2).repeat(1, 1, num_frames)

        threshold_r = torch.max(mag_spec_r**2, dim=2)
        threshold_r_in_dB = 10 * torch.log10(threshold_r) - threshold_in_dB
        threshold_r_in_dB = threshold_r_in_dB.unsqueeze(2).repeat(1, 1, num_frames)

        # Cho
        binary_mask_l = (self._amplitude_to_db(mag_spec_l) > threshold_l_in_dB).float()
        binary_mask_r = (self._amplitude_to_db(mag_spec_r) > threshold_r_in_dB).float()

        binaural_mask = torch.bitwise_and(binary_mask_l.int(), binary_mask_r.int())

        return binaural_mask

    def forward(self, est, ref):
        # est: [B, 2, T]
        # ref: [B, 2, T]

        # STFT
        est_mag, est_phase, *_ = self.stft(est)  # [B, 2, F, T]
        ref_mag, ref_phase, *_ = self.stft(ref)

        est_mag_l, est_mag_r, est_phase_l, est_phase_r = est_mag[:, 0], est_mag[:, 1], est_phase[:, 0], est_phase[:, 1]
        ref_mag_l, ref_mag_r, ref_phase_l, ref_phase_r = ref_mag[:, 0], ref_mag[:, 1], ref_phase[:, 0], ref_phase[:, 1]

        loss = 0

        if self.snr_loss_weight > 0.0:
            snr_l = self.snr_loss(est[:, 0], ref[:, 0])
            snr_r = self.snr_loss(est[:, 1], ref[:, 1])

            snr_loss = (snr_l + snr_r) / 2

            loss += self.snr_loss_weight * snr_loss

        if self.stoi_weight > 0.0:
            stoi_l = self.stoi_loss(est[:, 0], ref[:, 0])
            stoi_r = self.stoi_loss(est[:, 1], ref[:, 1])

            stoi_loss = (stoi_l + stoi_r) / 2
            stoi_loss = stoi_loss.mean()  # [B] -> []

            loss += self.stoi_weight * stoi_loss

        if self.ild_weight > 0.0:
            est_ild = self._ild_in_dB(est_mag_l, est_mag_r)
            ref_ild = self._ild_in_dB(ref_mag_l, ref_mag_r)

            ild_loss = torch.abs(est_ild - ref_ild)  # [B, F, T]

            binaural_mask = self._build_energy_mask(est_mag_l, est_mag_r)  # [B, F, T]

            masked_ild_loss = (ild_loss * binaural_mask).sum(dim=(1, 2)) / binaural_mask.sum(dim=(1, 2))

            loss += self.ild_weight * masked_ild_loss.mean()

        if self.ipd_weight > 0.0:
            ref_ipd = ref_phase_l - ref_phase_r
            est_ipd = est_phase_l - est_phase_r

            ipd_loss = torch.abs(est_ipd - ref_ipd)

            binaural_mask = self._build_energy_mask(est_mag_l, est_mag_r, threshold_in_dB=20)  # [B, F, T]

            masked_ipd_loss = (ipd_loss * binaural_mask).sum(dim=(1, 2)) / binaural_mask.sum(dim=(1, 2))

            loss += self.ipd_weight * masked_ipd_loss.mean()

        return loss


if __name__ == "__main__":
    n_ffts = [240, 480, 960, 1440]
    gamma = 0.3
    factor = 1
    f_complex = 1

    loss = CombineLoss(n_ffts, gamma, factor, f_complex)
    input = torch.rand(2, 16000)
    target = torch.rand(2, 16000)
    print(loss(input, target))
