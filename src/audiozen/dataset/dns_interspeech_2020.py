from dataclasses import dataclass
from pathlib import Path

import librosa
from simple_parsing import Serializable
from torch.utils.data import Dataset


@dataclass
class SyntheticDatasetArgs(Serializable):
    """DNS Challenge Interspeech 2020 synthetic dataset.

    Note:
        The dataset must have the same number of noisy and clean audio files.
        The noisy and clean audio files have the different filenames.

    Args:
        dataset_root_dir: The root directory of the dataset.
        audio_sample_len_in_sec: The length of the audio samples in seconds.
        offset: The index of the first audio file to use.
        limit: The maximum number of audio files to use.
        sr: The sample rate of the audio files.
        noisy_subdir: The name of the subdirectory containing the noisy audio files.
        clean_subdir: The name of the subdirectory containing the clean audio files.
    """

    dataset_root_dir: str = "datasets/VoiceBank-Demand"
    audio_sample_len_in_sec: int = 4
    offset: int = -1
    limit: int = -1
    sr: int = 16000
    noisy_subdir: str = "noisy"
    clean_subdir: str = "clean"


class SyntheticDataset(Dataset):
    def __init__(self, args: SyntheticDatasetArgs):
        super().__init__()
        self.args = args

        dataset_root_dir = Path(args.dataset_root_dir).expanduser().resolve()

        self.noisy_root_dir = dataset_root_dir / args.noisy_subdir
        self.clean_root_dir = dataset_root_dir / args.clean_subdir

        if not self.noisy_root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.noisy_root_dir}")

        if not self.clean_root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.clean_root_dir}")

        self.noisy_audio_fpath_list = librosa.util.find_files(self.noisy_root_dir)
        self.clean_audio_fpath_list = librosa.util.find_files(self.clean_root_dir)

        assert len(self.noisy_audio_fpath_list) == len(self.clean_audio_fpath_list), (
            f"Number of noisy and clean audio files must be equal. "
            f"Found {len(self.noisy_audio_fpath_list)} noisy files and "
            f"{len(self.clean_audio_fpath_list)} clean files."
        )

        if args.offset > 0:
            self.noisy_audio_fpath_list = self.noisy_audio_fpath_list[args.offset :]

        if args.limit > 0:
            step = max(len(self.noisy_audio_fpath_list) // args.limit, 1)
            self.noisy_audio_fpath_list = self.noisy_audio_fpath_list[::step]

        if args.audio_sample_len_in_sec > 0:
            self.audio_sample_len_in_samples = int(args.audio_sample_len_in_sec * args.sr)

    def __len__(self):
        return len(self.noisy_audio_fpath_list)

    def __getitem__(self, idx):
        noisy_fpath = self.noisy_audio_fpath_list[idx]
        stem = Path(noisy_fpath).stem
        file_id = stem.split("_")[-1]

        clean_fname = f"clean_fileid_{file_id}.wav"
        clean_fpath = (self.clean_root_dir / clean_fname).as_posix()

        noisy_y, _ = librosa.load(noisy_fpath, sr=self.args.sr)
        clean_y, _ = librosa.load(clean_fpath, sr=self.args.sr)

        return noisy_y, clean_y, stem
