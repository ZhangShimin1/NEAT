import torch
import torchaudio
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import random
from torchvision import transforms
import os
import json
import math

from os.path import join
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt

# Dataset definition
n_fft =  320 #480 #320 #960 #480 #1280 #2560.0
sample_rate = 16000.0
frame_length = n_fft / sample_rate * 1000.0  # 120 ms
frame_shift = frame_length / 2.0  # 60 ms

params = {
    "channel": 0,
    "dither": 0.0,
    "window_type": "hanning",
    "num_mel_bins": 40,
    "frame_length": frame_length,
    "frame_shift": frame_shift,
    "remove_dc_offset": False,
    "round_to_power_of_two": False,
    "sample_frequency": sample_rate,
}

class vadDataset(Dataset):
    def __init__(self, dataDir, is1dconv=False, isMask=False, sample_rate=16000, data_args=None):
        self.filePath = dataDir['FilePaths']
        self.labelPath = dataDir['LabelPaths']
        self.file_length = 120  # 120 seconds
        self.file_samples = int(self.file_length * sample_rate)
        self.is1dconv = is1dconv
        self.isMask = isMask
        self.data_args = data_args
    def __getitem__(self, index):
        filename = self.filePath[index]
        label_path = self.labelPath[index]
        # file_length = self.readLength(label_path)
        label = self.labelGenerator(label_path, self.file_length)

        # extract fbank feature
        waveform, _ = torchaudio.load(filename)
        waveform = F.pad(waveform, (0, self.file_samples - waveform.shape[1]))  # zero-pad the waveforms
        specgram = torchaudio.compliance.kaldi.fbank(waveform[0:self.file_samples - 1], **params)

        if self.is1dconv:
            if self.isMask:
                clean_filename = filename.replace("QUT-NOISE-TIMIT", "QUT-NOISE-TIMIT_clean")
                clean, _ = torchaudio.load(clean_filename)
                clean = F.pad(clean, (0, self.file_samples - clean.shape[1]))  # zero-pad the waveforms    
                return waveform, clean, label
            
            else:
                return waveform, label
        
        else:
            if self.isMask:
                clean_filename = filename.replace("QUT-NOISE-TIMIT", "QUT-NOISE-TIMIT_clean")
                clean, _ = torchaudio.load(clean_filename)
                clean = F.pad(clean, (0, self.file_samples - clean.shape[1]))  # zero-pad the waveforms 
                clean_specgram = torchaudio.compliance.kaldi.fbank(clean[0:self.file_samples - 1], **params)
                return specgram, clean_specgram, label
   

        

        return specgram, label

    def __len__(self):
        return len(self.labelPath)

    def readLength(self, path):
        with open(path, 'r') as f:
            lines = f.read().splitlines()
            last_line = lines[-1]
            data = last_line.split()

        return int(data[1])

    def labelGenerator(self, path, file_length):
        num_frames = int((file_length * 1000 - frame_length) // frame_shift + 1)
        label = np.zeros(num_frames)

        with open(path, 'r') as f:
            for line in f:
                data = line.split()
                if data[2] == 'speech':
                    start_idx = int((float(data[0]) * 1000) // frame_shift)
                    end_idx = int((float(data[1]) * 1000 - frame_length) // frame_shift + 1)
                    label[start_idx:end_idx] = 1
                else:
                    continue

        return label
