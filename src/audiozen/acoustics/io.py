from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf


def load_audio(
    path: Union[Path, str], duration: Optional[float] = None, sr: Optional[int] = None, mode: str = "wrap", **kwargs
) -> Tuple[np.ndarray, int]:
    """Load the audio using soundfile and resample if necessary.

    Compared to sf.read, this function supports:
    - loading the random segment of a audio when duration is specified.
    - resampling the audio when sr is specified.

    Args:
        path: Path to the audio file.
        duration: Duration of the audio segment in seconds. If None, the whole audio is loaded.
        sr: Sampling rate of the audio. If None, the original sampling rate is used.
        mode: Padding mode when duration is longer than the original audio.
        **kwargs: Additional keyword arguments for `np.pad`.

    Returns:
        A tuple of the audio signal and the sampling rate.

    Examples:
        >>> load_audio("test.wav", duration=2.0, sr=16000)
    """
    path = str(path)

    with sf.SoundFile(path) as sf_desc:
        orig_sr = sf_desc.samplerate

        if duration is not None:
            frame_orig_duration = sf_desc.frames
            frame_duration = int(duration * orig_sr)

            if frame_duration < frame_orig_duration:
                # Randomly select a segment
                offset = np.random.randint(frame_orig_duration - frame_duration)
                sf_desc.seek(offset)
                y = sf_desc.read(frames=frame_duration, dtype=np.float32, always_2d=True).T
            else:
                y = sf_desc.read(dtype=np.float32, always_2d=True).T  # [C, T]
                if frame_duration > frame_orig_duration:
                    y = np.pad(y, ((0, 0), (0, frame_duration - frame_orig_duration)), mode=mode, **kwargs)
        else:
            y = sf_desc.read(dtype=np.float32, always_2d=True).T

    if y.shape[0] == 1:
        y = y.flatten()

    if sr is not None and sr != orig_sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        orig_sr = sr

    return y, orig_sr


def extract_segment(
    data: np.ndarray,
    segment_length: int,
    start_idx: int = -1,
    return_start_idx: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """Sample a segment from the N-dimensional data with the last dimension as the time dimension.

    Args:
        data: Any dimensional data. The last dimension is the time dimension.
        segment_length: The length of the segment to be sampled.
        start_idx: The start index of the segment to be sampled. If less than 0, a random index is generated.
        return_start_idx: Whether to return the start index along with the data segment.

    Returns:
        The sampled data segment, and optionally the start index.

    Raises:
        ValueError: If segment_length is negative.

    Examples:
        >>> subsample(np.zeros(10), 5)
        >>> subsample(np.zeros(10), 5, start_idx=0)
        >>> subsample(np.zeros(10), 5, start_idx=0, return_start_idx=True)
    """
    # make sure int
    segment_length = int(segment_length)

    if segment_length < 0:
        raise ValueError("`segment_length` must be non-negative")

    data_len = data.shape[-1]
    ndim = data.ndim

    if data_len > segment_length:
        if start_idx > 0:
            end_idx = start_idx + segment_length
        else:
            start_idx = np.random.randint(data_len - segment_length)
            end_idx = start_idx + segment_length
        data = data[..., start_idx:end_idx]
    elif data_len < segment_length:
        padding = segment_length - data_len
        # data = np.pad(data, (0, padding), "constant")
        pad_width = [(0, 0)] * (ndim - 1) + [(0, padding)]
        data = np.pad(data, pad_width, "constant")
        start_idx = 0  # When padding, the start_idx is effectively 0

    return (data, start_idx) if return_start_idx else data
