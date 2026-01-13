import glob
import json
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


class RawAudioParser(object):
    """
    :param normalize_waveform
        whether to N(0,1) normalize audio waveform
    """

    def __init__(self, normalize_waveform=False):
        super().__init__()
        self.normalize_waveform = normalize_waveform
        if self.normalize_waveform:
            print("ATTENTION!!! Mean/AVG norm on")

    def __call__(self, audio):
        output = torch.from_numpy(audio.astype("float32")).float()
        if self.normalize_waveform:
            mean = output.mean()
            std = output.std()
            output = (output - mean) / (std + 1e-9)
        output = output.unsqueeze(0)
        return output, None


def load_audio(
    f,
    sr,
    min_duration: float = 5.0,
    read_cropped=False,
    frames_to_read=-1,
    audio_size=None,
):
    if min_duration is not None:
        min_samples = int(sr * min_duration)
    else:
        min_samples = None
    # x, clip_sr = torchaudio.load(f, channels_first=False)
    # x = x.squeeze().cpu().numpy()
    if read_cropped:
        assert audio_size
        assert frames_to_read != -1
        if frames_to_read >= audio_size:
            start_idx = 0
        else:
            start_idx = random.randint(0, audio_size - frames_to_read - 1)
        x, clip_sr = sf.read(f, frames=frames_to_read, start=start_idx)
        # print("start_idx: {} | clip size: {} | frames_to_read:{}".format(start_idx, len(x), frames_to_read))
        min_samples = frames_to_read
    else:
        x, clip_sr = sf.read(f)  # sound file is > 3x faster than torchaudio sox_io
    x = x.astype("float32")  # .cpu().numpy()
    assert clip_sr == sr

    # min filtering and padding if needed
    if min_samples is not None:
        if len(x) < min_samples:
            tile_size = (min_samples // x.shape[0]) + 1
            x = np.tile(x, tile_size)[:min_samples]
    return x


class RawWaveformDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        labels_map,
        audio_config,
        dataset_path=None,  # ← 新增
        augment=False,
        mode="multilabel",
        delimiter=",",
        mixer=None,
        transform=None,
        is_val=False,
        cropped_read=False,
        frontend="fbank",
    ):
        super(RawWaveformDataset, self).__init__()
        assert os.path.isfile(labels_map)
        assert os.path.splitext(labels_map)[-1] == ".json"
        assert audio_config is not None
        self.mode = mode
        self.transform = transform
        self.mixer = mixer
        self.cropped_read = cropped_read
        self.is_val = is_val
        self.frontend = frontend

        with open(labels_map, "r") as fd:
            self.labels_map = json.load(fd)
        self.labels_delim = delimiter
        self.parse_audio_config(audio_config)
        if self.background_noise_path is not None:
            if os.path.exists(self.background_noise_path):
                self.bg_files = glob.glob(
                    os.path.join(self.background_noise_path, "*.wav")
                )
        else:
            self.bg_files = None
        df = pd.read_csv(manifest_path)

        files = df["files"].values.tolist()
        labels = df["labels"].values.tolist()
        if dataset_path is not None:
            dataset_path = os.path.abspath(dataset_path)
            files = [os.path.normpath(os.path.join(dataset_path, f)) for f in files]

        self.files = files
        self.labels = labels

        if self.cropped_read:
            self.durations = df["durations"].values.tolist()
        self.spec_parser = RawAudioParser(normalize_waveform=self.normalize)
        self.length = len(self.files)

    def parse_audio_config(self, audio_config):
        self.sr = int(audio_config.get("sample_rate", "22050"))
        self.normalize = bool(audio_config.get("normalize", False))
        self.min_duration = float(audio_config.get("min_duration", 2.5))
        self.background_noise_path = audio_config.get("bg_files", None)
        if self.cropped_read:
            self.num_frames = int(audio_config.get("random_clip_size") * self.sr)
        else:
            self.num_frames = -1

        delim = audio_config.get("delimiter", None)
        if delim is not None:
            print("Reassigning delimiter from audio_config")
            self.labels_delim = delim

    def __get_feature__(self, audio) -> Tuple[torch.Tensor, torch.Tensor]:
        real, comp = self.spec_parser(audio)
        return real, comp

    def __get_item_helper__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lbls = self.labels[index]
        if self.cropped_read and not self.is_val:
            dur = self.durations[index]
        else:
            dur = None
        preprocessed_audio = load_audio(
            self.files[index],
            self.sr,
            self.min_duration,
            read_cropped=self.cropped_read,
            frames_to_read=self.num_frames,
            audio_size=dur,
        )
        real, comp = self.__get_feature__(preprocessed_audio)
        label_tensor = self.__parse_labels__(lbls)

        if self.transform is not None:
            real = self.transform(real)
        return real, comp, label_tensor

    def __parse_labels__(self, lbls: str) -> torch.Tensor:
        if self.mode == "multilabel":
            label_tensor = torch.zeros(len(self.labels_map)).float()
            for lbl in lbls.split(self.labels_delim):
                label_tensor[self.labels_map[lbl]] = 1

            return label_tensor
        elif self.mode == "multiclass":
            # print("multiclassssss")
            return self.labels_map[lbls]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real, comp, label_tensor = self.__get_item_helper__(index)
        if self.mixer is not None:
            real, final_label = self.mixer(self, real, label_tensor)
            if self.mode != "multiclass":
                return real, final_label

        if self.frontend == "fbank" or self.frontend == "Spiking_fbank":
            feat = torchaudio.compliance.kaldi.fbank(real, num_mel_bins=40).unsqueeze(0)
        else:
            feat = real.unsqueeze(0)
        return real, feat, label_tensor

    def __len__(self):
        return self.length
