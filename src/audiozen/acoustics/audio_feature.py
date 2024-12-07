from pathlib import Path
from typing import Literal, Optional, Union

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from numpy.typing import NDArray
from torch import Tensor

from audiozen.constant import EPSILON


def find_files(path_or_path_list: Union[list, str, Path], offset: int = 0, limit: Union[int, None] = None):
    """Find wav files from a directory, or a list of files, or a txt file, or the combination of them.

    Args:
        path_or_path_list: path to wav file, str, pathlib.Path, or a list of them.
        offset: offset of samples to load.
        limit: limit of samples to load.

    Returns:
        A list of wav file paths.

    Examples:
        >>> # Load 10 files from a directory
        >>> wav_paths = file_loader(path="~/dataset", limit=10, offset=0)
        >>> # Load files from a directory, a txt file, and a wav file
        >>> wav_paths = file_loader(path=["~/dataset", "~/scp.txt", "~/test.wav"])
    """
    if not isinstance(path_or_path_list, list):
        path_or_path_list = [path_or_path_list]

    output_paths = []
    for path in path_or_path_list:
        path = Path(path).resolve()

        if path.is_dir():
            wav_paths = librosa.util.find_files(path, ext="wav")
            output_paths += wav_paths

        if path.is_file():
            if path.suffix == ".wav":
                output_paths.append(path.as_posix())
            else:
                for line in open(path, "r"):
                    line = line.rstrip("\n")
                    line = Path(line).resolve()
                    output_paths.append(line.as_posix())

    if offset > 0:
        output_paths = output_paths[offset:]

    if limit:
        output_paths = output_paths[:limit]

    return output_paths


def is_audio(y_s: Union[list[NDArray], NDArray]):
    if not isinstance(y_s, list):
        y_s = [y_s]

    for y in y_s:
        assert y.ndim in (1, 2), "Only support signals with the shape of [C, T] or [T]."


def compute_rms(y: NDArray) -> float:
    """Compute the Root Mean Square (RMS) of the given signal."""
    return np.sqrt(np.mean(y**2))


def loudness_max_norm(
    y: NDArray, scalar: Union[float, None] = None, ref_mic: int = 0, eps: float = EPSILON
) -> tuple[NDArray, float]:
    """Maximum loudness normalization to signals."""
    if scalar is None:
        if y.ndim == 1:
            scalar = np.max(np.abs(y)) + eps
        else:
            scalar = np.max(np.abs(y[ref_mic, :])) + eps

    assert scalar is not None
    return y / scalar, scalar


def loudness_rms_norm(
    y: NDArray, scalar: Union[float, None] = None, lvl: float = -25, ref_mic: int = 0, eps: float = EPSILON
) -> tuple[NDArray, float]:
    """Loudness normalize a signal based on the Root Mean Square (RMS).

    Normalize the RMS of signals to a given RMS based on Decibels Relative to Full Scale (dBFS).

    Args:
        y: [C, T] or [T,].
        scalar: scalar to normalize the RMS, default to None.
        target_rms: target RMS in dBFS.
        ref_mic: reference mic for multi-channel signals.

    Returns:
        Loudness normalized signal and scalar.

    Note:
        A small amount of signal samples would be clipped after normalization, but it does not matter.
    """
    if not scalar:
        current_level = compute_rms(y) if y.ndim == 1 else compute_rms(y[ref_mic, :])
        scalar = 10 ** (lvl / 20) / (current_level + eps)

    return y * scalar, scalar


def active_rms(clean, noise, sr=16000, energy_threshold=-50, eps=EPSILON):
    """Compute the active RMS of clean and noise signals based on the energy threshold (dB)."""
    window_size = 100  # ms
    window_samples = int(sr * window_size / 1000)
    sample_start = 0

    noise_active_segments = []
    clean_active_segments = []

    while sample_start < len(noise):
        sample_end = min(sample_start + window_samples, len(noise))
        noise_win = noise[sample_start:sample_end]
        clean_win = clean[sample_start:sample_end]
        noise_seg_rms = compute_rms(noise_win)

        if noise_seg_rms > energy_threshold:
            noise_active_segments = np.append(noise_active_segments, noise_win)
            clean_active_segments = np.append(clean_active_segments, clean_win)

        sample_start += window_samples

    if len(noise_active_segments) != 0:
        noise_rms = compute_rms(noise_active_segments)
    else:
        noise_rms = eps

    if len(clean_active_segments) != 0:
        clean_rms = compute_rms(clean_active_segments)
    else:
        clean_rms = eps

    return clean_rms, noise_rms


def normalize_segmental_rms(audio, rms, target_lvl=-25, eps=EPSILON):
    """Normalize the RMS of a segment to a target level.

    Args:
        audio: audio segment.
        rms: RMS of the audio segment.
        target_lvl: target level in dBFS.
        eps: a small value to avoid dividing by zero.

    Returns:
        Normalized audio segment.
    """
    scalar = 10 ** (target_lvl / 20) / (rms + eps)
    return audio * scalar


def sxr2gain(meaningful: NDArray, meaningless: NDArray, desired_ratio: float, eps: float = EPSILON) -> float:
    """Generally calculate the gains of interference to fulfill a desired SXR (SNR or SIR) ratio.

    Args:
        meaningful: meaningful input, like target clean.
        meaningless: meaningless or unwanted input, like background noise.
        desired_ratio: SNR or SIR ratio.

    Returns:
        Gain, which can be used to adjust the RMS of the meaningless signals to satisfy the given ratio.
    """
    meaningful_rms = compute_rms(meaningful)
    meaningless_rms = compute_rms(meaningless)
    scalar = meaningful_rms / (10 ** (desired_ratio / 20)) / (meaningless_rms + eps)

    return scalar


def load_wav(wav_path: str, sr: int = 16000) -> NDArray:
    """Load a wav file.

    Args:
        file: file path.
        sr: sample rate. Defaults to 16000.

    Returns:
        Waveform with shape of [C, T] or [T].
    """
    wav_path = Path(wav_path).resolve()
    y, _ = librosa.load(wav_path, sr=sr, mono=False)
    return y


def save_wav(data, fpath, sr):
    if data.ndim != 1:
        data = data.reshape(-1)

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    sf.write(fpath, data, sr)


def mag_phase(complex_valued_tensor: Tensor) -> tuple[Tensor, Tensor]:
    """Get magnitude and phase of a complex-valued tensor.

    Args:
        complex_valued_tensor: complex-valued tensor.

    Returns:
        magnitude and phase spectrogram.
    """
    mag, phase = torch.abs(complex_valued_tensor), torch.angle(complex_valued_tensor)
    return mag, phase


def stft(
    y,
    n_fft,
    hop_length,
    win_length,
    window: Optional[str] = "hann",
    output_type: Literal["mag_phase", "real_imag", "complex"] | None = None,
    **kwargs,
):
    """Wrapper of the official ``torch.stft`` for single-channel and multichannel signals.

    Args:
        y (`torch.Tensor` of shape `(batch_size, num_channels, num_samples) or `(batch_size, num_samples)`):
            single-/multichannel signals.
        n_fft: num of FFT.
        hop_length: hop length.
        win_length: hanning window size.
        output_type: "mag_phase", "real_imag", "complex", or None. Defaults to None.
        kwargs: other arguments for ``torch.stft``.

    Returns:
        If the input is single-channel, return the spectrogram with shape of [B, F, T], otherwise [B, C, F, T].
        If output_type is "mag_phase", return a list of magnitude and phase spectrogram.
        If output_type is "real_imag", return a list of real and imag spectrogram.
        If output_type is None, return a list of magnitude, phase, real, and imag spectrogram.
    """
    # Support [B, T] and [B, C, T]
    if y.ndim not in [2, 3]:
        raise ValueError(f"Only support single-/multi-channel signals. Received {y.ndim=}.")

    is_multi_channel = y.ndim == 3

    batch_size, *_, num_samples = y.shape

    # Compatible with multi-channel signals
    if is_multi_channel:
        y = y.reshape(-1, num_samples)

    if window is not None:
        window_func = getattr(torch, f"{window}_window")
        window = window_func(win_length, device=y.device)
    else:
        window = torch.ones(n_fft, device=y.device)

    complex_stft = torch.stft(
        input=y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        **kwargs,
    )

    # Reshape back to original if the input is multi-channel
    if is_multi_channel:
        complex_stft = complex_stft.reshape(batch_size, -1, *complex_stft.shape[-2:])

    if output_type == "mag_phase":
        return mag_phase(complex_stft)
    elif output_type == "real_imag":
        return complex_stft.real, complex_stft.imag
    elif output_type == "complex":
        return complex_stft
    else:
        mag, phase = mag_phase(complex_stft)
        return mag, phase, complex_stft.real, complex_stft.imag


def istft(
    feature: Union[Tensor, tuple[Tensor, ...], list[Tensor]],
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: Optional[str] = "hann",
    length: Optional[int] = None,
    input_type: Literal["mag_phase", "real_imag", "complex", "complex_view_as_real"] = "mag_phase",
) -> Tensor:
    """Wrapper of the official ``torch.istft`` for single-channel signals.

    Args:
        features (`torch.Tensor` of shape `(batch_size, num_channels, num_freqs, num_frames)` or list/tuple of tensors):
            single-channel spectrogram(s).
        n_fft: num of FFT.
        hop_length: hop length.
        win_length: hanning window size.
        length: expected length of istft.
        input_type: "real_image", "complex", or "mag_phase". Defaults to "mag_phase".

    Returns:
        Single-channel singal with the shape shape of [B, T].
    """
    # Compatible with multiple inputs
    match input_type:
        case "real_imag":
            if (isinstance(feature, tuple) or isinstance(feature, list)) and len(feature) == 2:
                real, imag = feature
                complex_valued_features = torch.complex(real=real, imag=imag)
            else:
                raise ValueError(f"Only support tuple or list. Received {type(feature)} with {len(feature)} elements.")
        case "complex":
            if not (isinstance(feature, Tensor) and torch.is_complex(feature)):
                raise ValueError(f"Only support complex-valued tensor. Received {type(feature)}")
            complex_valued_features = feature
        case "complex_view_as_real":
            assert feature.ndim == 4, f"Only support 4D tensor. Received {feature.ndim=}"
            assert feature.shape[1] == 2, f"Only support 2 channels. Received {feature.shape[1]=}"
            real, imag = feature[:, 0, ...], feature[:, 1, ...]
            complex_valued_features = torch.complex(real=real, imag=imag)
        case "mag_phase":
            if not ((isinstance(feature, tuple) or isinstance(feature, list)) and len(feature) != 2):
                raise ValueError(f"Only support tuple or list. Received {type(feature)} with {len(feature)} elements.")
            mag, phase = feature
            complex_valued_features = torch.polar(mag, phase)
        case _:
            raise ValueError(f"Only support 'real_imag', 'complex', and 'mag_phase'. Received {input_type=}")

    if window is not None:
        window_func = getattr(torch, f"{window}_window")
        window = window_func(win_length, device=complex_valued_features.device)
    else:
        window = torch.ones(n_fft, device=complex_valued_features.device)

    return torch.istft(
        input=complex_valued_features,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=length,
    )


def norm_amplitude(y: NDArray, scalar: Optional[float] = None, eps: float = EPSILON):
    if not scalar:
        scalar = np.max(np.abs(y)) + eps

    return y / scalar, scalar


def is_clipped(y: NDArray, clipping_threshold: float = 0.999) -> bool:
    """Check if the input signal is clipped."""
    return (np.abs(y) > clipping_threshold).any()  # type: ignore


def tune_dB_FS(y, target_dB_FS=-26, eps=EPSILON):
    """Tune the RMS of the input signal to a target level.

    Args:
        y: Audio signal with any shape.
        target_dB_FS: Target dB_FS. Defaults to -25.
        eps: A small value to avoid dividing by zero. Defaults to EPSILON.

    Returns:
        Scaled audio signal with the same shape as the input.
    """
    if isinstance(y, torch.Tensor):
        rms = torch.sqrt(torch.mean(y**2))
        scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
        y *= scalar
        return y, rms, scalar
    else:
        rms = np.sqrt(np.mean(y**2))
        scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
        y *= scalar
        return y, rms, scalar


def activity_detector(audio, fs=16000, activity_threshold=0.13, target_level=-25, eps=EPSILON):
    """Return the percentage of the time the audio signal is above an energy threshold.

    Args:
        audio:
        fs:
        activity_threshold:
        target_level:
        eps:

    Returns:

    """
    audio, _ = loudness_rms_norm(audio, target_level)
    window_size = 50  # ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    cnt = 0
    prev_energy_prob = 0
    active_frames = 0

    a = -1
    b = 0.2
    alpha_rel = 0.05
    alpha_att = 0.8

    while sample_start < len(audio):
        sample_end = min(sample_start + window_samples, len(audio))
        audio_win = audio[sample_start:sample_end]
        frame_rms = 20 * np.log10(sum(audio_win**2) + eps)
        frame_energy_prob = 1.0 / (1 + np.exp(-(a + b * frame_rms)))

        if frame_energy_prob > prev_energy_prob:
            smoothed_energy_prob = frame_energy_prob * alpha_att + prev_energy_prob * (1 - alpha_att)
        else:
            smoothed_energy_prob = frame_energy_prob * alpha_rel + prev_energy_prob * (1 - alpha_rel)

        if smoothed_energy_prob > activity_threshold:
            active_frames += 1
        prev_energy_prob = frame_energy_prob
        sample_start += window_samples
        cnt += 1

    perc_active = active_frames / cnt
    return perc_active


def build_complex_ideal_ratio_mask(noisy_real, noisy_imag, clean_real, clean_imag) -> torch.Tensor:
    """Build the complex ratio mask.

    Args:
        noisy: [B, F, T], noisy complex-valued stft coefficients
        clean: [B, F, T], clean complex-valued stft coefficients

    References:
        https://ieeexplore.ieee.org/document/7364200

    Returns:
        [B, F, T, 2]
    """
    denominator = torch.square(noisy_real) + torch.square(noisy_imag) + EPSILON

    mask_real = (noisy_real * clean_real + noisy_imag * clean_imag) / denominator
    mask_imag = (noisy_real * clean_imag - noisy_imag * clean_real) / denominator

    complex_ratio_mask = torch.stack((mask_real, mask_imag), dim=-1)

    return compress_cIRM(complex_ratio_mask, K=10, C=0.1)


def compress_cIRM(mask, K=10, C=0.1):
    """Compress the value of cIRM from (-inf, +inf) to [-K ~ K].

    References:
        https://ieeexplore.ieee.org/document/7364200
    """
    if torch.is_tensor(mask):
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - torch.exp(-C * mask)) / (1 + torch.exp(-C * mask))
    else:
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - np.exp(-C * mask)) / (1 + np.exp(-C * mask))
    return mask


def decompress_cIRM(mask, K=10, limit=9.9):
    """Decompress cIRM from [-K ~ K] to [-inf, +inf].

    Args:
        mask: cIRM mask
        K: default 10
        limit: default 0.1

    References:
        https://ieeexplore.ieee.org/document/7364200
    """
    mask = limit * (mask >= limit) - limit * (mask <= -limit) + mask * (torch.abs(mask) < limit)
    mask = -K * torch.log((K - mask) / (K + mask))
    return mask


def complex_mul(noisy_r, noisy_i, mask_r, mask_i):
    r = noisy_r * mask_r - noisy_i * mask_i
    i = noisy_r * mask_i + noisy_i * mask_r
    return r, i


class Mask:
    def __init__(self) -> None:
        pass

    @staticmethod
    def plot_mask(mask, title="Mask", xlim=None):
        mask = mask.numpy()
        figure, axis = plt.subplots(1, 1)
        img = axis.imshow(mask, cmap="viridis", origin="lower", aspect="auto")
        figure.suptitle(title)
        plt.colorbar(img, ax=axis)
        plt.show()


class IRM(Mask):
    def __init__(self) -> None:
        super(IRM, self).__init__()

    @staticmethod
    def generate_mask(clean, noise, ref_channel=0):
        """Generate an ideal ratio mask.

        Args:
            clean: Complex STFT of clean audio with the shape of [B, C, F, T].
            noise: Complex STFT of noise audio with the same shape of [B, 1, F, T].
            ref_channel: The reference channel to compute the mask if the STFTs are multi-channel.

        Returns:
            Speech mask and noise mask with the shape of [B, 1, F, T].
        """
        mag_clean = clean.abs() ** 2
        mag_noise = noise.abs() ** 2
        irm_speech = mag_clean / (mag_clean + mag_noise)
        irm_noise = mag_noise / (mag_clean + mag_noise)

        if irm_speech.ndim == 4:
            irm_speech = irm_speech[ref_channel, :][:, None, ...]
            irm_noise = irm_noise[ref_channel, :][:, None, ...]

        return irm_speech, irm_noise


def drop_band(input, num_groups=2):
    """
    Reduce computational complexity of the sub-band part in the FullSubNet model.

    Shapes:
        input: [B, C, F, T]
        return: [B, C, F // num_groups, T]
    """
    batch_size, _, num_freqs, _ = input.shape
    assert (
        batch_size > num_groups
    ), f"Batch size = {batch_size}, num_groups = {num_groups}. The batch size should larger than the num_groups."

    if num_groups <= 1:
        # No demand for grouping
        return input

    # Each sample must has the same number of the frequencies for parallel training.
    # Therefore, we need to drop those remaining frequencies in the high frequency part.
    if num_freqs % num_groups != 0:
        input = input[..., : (num_freqs - (num_freqs % num_groups)), :]
        num_freqs = input.shape[2]

    output = []
    for group_idx in range(num_groups):
        samples_indices = torch.arange(group_idx, batch_size, num_groups, device=input.device)
        freqs_indices = torch.arange(group_idx, num_freqs, num_groups, device=input.device)

        selected_samples = torch.index_select(input, dim=0, index=samples_indices)
        selected = torch.index_select(selected_samples, dim=2, index=freqs_indices)  # [B, C, F // num_groups, T]

        output.append(selected)

    return torch.cat(output, dim=0)


def hanning_hugh(win_length):
    win = np.hanning(win_length)

    w_half = win[: win_length // 2]

    w_half = (w_half + (1 - np.flipud(w_half))) // 2

    w = np.zeros(win_length)
    w[: win_length // 2] = w_half
    w[-win_length // 2 :] = np.flipud(w_half)

    return w


def asymmetric_analysis_window(k, m, d=0):
    rising_sqrt_hann = np.sqrt(np.hanning(2 * (k - m - d) + 1)[: 2 * (k - m - d)])
    falling_sqrt_hann = np.sqrt(np.hanning(2 * m + 1)[: 2 * m])

    win = np.zeros(k)
    win[:d] = 0
    win[d : k - m] = rising_sqrt_hann[: k - m - d]
    win[k - m :] = falling_sqrt_hann[-m:]

    return win


def asymmetric_synthesis_window(k, m, d=0):
    rising_sqrt_hann = np.sqrt(np.hanning(2 * (k - m - d) + 1)[: 2 * (k - m - d)])
    rasing_norm_hann = np.hanning(2 * m + 1)[:m] / rising_sqrt_hann[k - 2 * m - d : k - m - d]
    falling_sqrt_hann = np.sqrt(np.hanning(2 * m + 1)[: 2 * m])

    win = np.zeros(k)
    win[: -2 * m] = 0
    win[-2 * m : -m] = rasing_norm_hann
    win[-m:] = falling_sqrt_hann[-m:]

    return win


def asymmetric_stft(y, analysis_win, win_length=None, hop_length=None):
    """Forward STFT using PyTorch.

    Args:
        y: The input signal with a shape of (batch_size, signal_length).
        n_fft: The number of FFT points.
        window: The window function to apply to each frame.
        win_length: The length of the window function.

    Note:
        `num_windows x win_len` frames always longer than the original signal to make sure that the first and last samples
        are included in the output.

    Returns:
        Complex-valued matrix of short-term Fourier transform coefficients with a shape of (batch_size, num_freqs, num_frames).
    """
    # Ensure the input is 2D or 3D
    if y.ndim < 2 or y.ndim > 3:
        raise ValueError(f"Input signal must have shape [B, T] or [B, C, T], but received {y.shape=}.")

    is_multi_channel = y.ndim == 3

    if is_multi_channel:
        origin_batch_size, num_channels, num_samples = y.shape
        y = y.reshape(-1, num_samples)

    device = y.device

    if win_length is None:
        win_length = len(analysis_win)

    if hop_length is None:
        hop_length = win_length // 4

    if isinstance(analysis_win, np.ndarray):
        analysis_win = torch.tensor(analysis_win, dtype=torch.float32, device=device)

    # Pad the input signal to ensure that the first and last samples are included in the output although there are some redundant computations
    y = torch.nn.functional.pad(y, (win_length, win_length), mode="constant")
    batch_size, num_samples = y.shape

    # Calculate the number of windows
    num_windows = ((num_samples - win_length) // hop_length) + 1

    # Create a complex-valued tensor to store the STFT
    stft_matrix = torch.empty((batch_size, num_windows, win_length // 2 + 1), dtype=torch.cfloat, device=device)

    # Create indices for sliding windows. This results in a shape of (win_len, num_windows) for all batches
    indices = torch.arange(win_length, device=device).unsqueeze(0) + torch.arange(
        0, num_windows * hop_length - 1, hop_length, device=device
    ).unsqueeze(1)

    # Gather the windows from each signal in the batch
    # Use `unsqueeze(1)` to replicate indices across the batch dimension
    windows = y[..., indices]  # Shape: (batch_size, win_len, num_windows)

    # Apply the window function to each of the gathered segments
    windowed_segments = analysis_win.unsqueeze(0) * windows  # Shape: (batch_size, win_len, num_windows)

    # Perform the FFT on all windowed segments in batch
    stft_matrix = torch.fft.rfft(
        windowed_segments.view(-1, win_length)
    )  # Reshape to (batch_size*num_windows, win_len)

    # Reshape the output to (batch_size, num_windows, win_len // 2 + 1)
    stft_matrix = stft_matrix.view(batch_size, num_windows, -1)
    stft_matrix = stft_matrix.permute(0, 2, 1).contiguous()

    if is_multi_channel:
        stft_matrix = stft_matrix.reshape(origin_batch_size, num_channels, -1, num_windows)

    return stft_matrix


def asymmetric_istft(stft_matrix, analysis_win, synthesis_win, win_length, hop_length, length=None):
    """Inverse STFT using PyTorch.

    Note:
        The input STFT matrix must have a shape of (num_freqs, num_frames).
        No support for multi-channel signals.
    """
    batch_size, num_freqs, num_frames = stft_matrix.shape
    device = stft_matrix.device

    if isinstance(analysis_win, np.ndarray):
        analysis_win = torch.tensor(analysis_win, dtype=torch.float32, device=device)

    if isinstance(synthesis_win, np.ndarray):
        synthesis_win = torch.tensor(synthesis_win, dtype=torch.float32, device=device)

    # Initialize the output signal and overlap sum tensors with zeros
    output_length = num_frames * hop_length + win_length
    x = torch.zeros(batch_size, output_length, device=stft_matrix.device)
    win_sum = torch.zeros(batch_size, output_length, device=stft_matrix.device)

    # Pre-compute overlap indices for all frames
    indices = torch.arange(num_frames) * hop_length  # shape: (num_frames,)

    # Compute inverse FFT for all frames at once, utilizing broadcasting
    irfft_frames = torch.fft.irfft(stft_matrix, n=win_length, dim=1)  # shape: (batch_size, win_length, num_frames)

    # Apply window to the results
    for frame_idx in range(num_frames):
        i = indices[frame_idx]

        irfft_frame = irfft_frames[..., frame_idx]  # shape: (batch_size, win_length)
        x[:, i : i + win_length] += irfft_frame * synthesis_win  # Broadcasting

        win_sum[:, i : i + win_length] += analysis_win * synthesis_win  # Broadcasting

    # Avoid division by zero by creating a mask
    pos = win_sum != 0

    # Normalize using the overlap sum
    x[pos] /= win_sum[pos]

    if length is not None:
        # Remove padding of the left (always) and right (if any)
        x = x[:, win_length : win_length + length]

    return x.type(torch.float32)


if __name__ == "__main__":
    # sr: 32 kHz
    # win_length: 512 (16 ms)
    # hop_length: 64 (2 ms)

    win_length = 512
    hop_length = 64
    # sig_length = 1988
    sig_length = 151

    analysis_win = asymmetric_analysis_window(win_length, hop_length)
    synthesis_win = asymmetric_synthesis_window(win_length, hop_length)

    input = torch.rand(2, 2, sig_length)
    torch_frames = asymmetric_stft(input, analysis_win, win_length, hop_length)
    print(torch_frames.shape)

    input = torch.rand(2, sig_length)
    torch_frames = asymmetric_stft(input, analysis_win, win_length, hop_length)
    torch_output = asymmetric_istft(
        torch_frames, analysis_win, synthesis_win, win_length, hop_length, length=sig_length
    )
    print(torch_output.shape)

    print(f"allclose  : {torch.allclose(input[0], torch_output[0])}")
    print(f"mean abs  : {torch.mean(torch.abs(input[0] - torch_output[0]))}")
    print(f"allclose  : {torch.allclose(input[1], torch_output[1])}")
    print(f"mean abs  : {torch.mean(torch.abs(input[1] - torch_output[1]))}")

    print(f"numpy allclose  : {np.allclose(input[0].numpy(), torch_output[0].numpy())}")
    print(f"numpy mean abs  : {np.mean(np.abs(input[0].numpy() - torch_output[0].numpy()))}")
