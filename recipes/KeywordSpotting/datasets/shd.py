'''
Descripttion: Dataloader for Spiking/Non-Spiking Heidelberg Digits
              References code from Sparch (https://github.com/idiap/sparch/tree/main)
version: 
Author: Shimin Zhang
Date: 2024-11-23 10:50:21
'''
import h5py
import numpy as np

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from torchaudio_augmentations import ComposeMany
from torchaudio_augmentations import Gain
from torchaudio_augmentations import Noise
from torchaudio_augmentations import PolarityInversion
from torchaudio_augmentations import RandomApply
from torchaudio_augmentations import Reverb


class SpikingDatasets(Dataset):
    def __init__(self, split, nb_steps=100, nb_units=700, max_time=1.4):
        self.device = "cpu"  # for memory allocation
        self.split_data = f'/datasets/kws/spiking_heidelberg_digits/shd_{split}.h5'
        self.nb_steps = nb_steps
        self.nb_units = nb_units
        self.time_bins = np.linspace(0, max_time, num=self.nb_steps)
        self.firing_times, self.units_fired, self.labels = self.load_spiking_data(self.split_data)
    
    def __getitem__(self, index):
        times = np.digitize(self.firing_times[index], self.time_bins)
        units = self.units_fired[index]

        x_idx = torch.LongTensor(np.array([times, units])).to(self.device)
        x_val = torch.FloatTensor(np.ones(len(times))).to(self.device)
        x_size = torch.Size([self.nb_steps, self.nb_units])

        x = torch.sparse.FloatTensor(x_idx, x_val, x_size).to(self.device)
        y = self.labels[index]

        return x.to_dense(), y
    
    def __len__(self):

        return len(self.labels)
    
    def load_spiking_data(self, path):
        data = h5py.File(path, "r")
        firing_times = data["spikes"]["times"]
        units_fired = data["spikes"]["units"]
        labels = np.array(data["labels"], dtype=np.int8)

        return firing_times, units_fired, labels

    def generate_batch(self, batch):
        xs, ys = zip(*batch)
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        # xlens = torch.tensor([x.shape[0] for x in xs])
        ys = torch.LongTensor(ys).to(self.device)

        return xs, ys
    

class NonSpikingDatasets(Dataset):
    def __init__(self, split, aug=False, aug_params=[0.0001, 0.9, 0.1]):
        self.split = split
        self.file_list = self.get_file_list()
        self.aug = aug
        self.aug_params = aug_params

    def __getitem__(self, index):
        file_name = self.file_list[index]
        spect = self.frontend(file_name)
        # e.g. file name: "lang-german_speaker-11_trial-64_digit-2.flac"
        target = int(file_name[-6])
        if file_name[5] == "g":
            target += 10

        return spect, target

    def __len__(self):

        return len(self.file_list)
    
    def get_file_list(self):
        file_name = f'/datasets/kws/spiking_heidelberg_digits/heidelberg_digits/{self.split}_filenames.txt'
        with open(file_name, "r") as f:
            file_list = f.read().splitlines()

        return file_list

    def frontend(self, file_name):
        file_path = f'/datasets/kws/spiking_heidelberg_digits/heidelberg_digits/audio/{file_name}'
        x, _ = torchaudio.load(file_path)

        if self.aug and self.split == "train":
            min_snr, max_snr, p_noise = self.aug_params
            x = self.augmentation(x, min_snr, max_snr, p_noise).squeeze(0)

        x = torchaudio.compliance.kaldi.fbank(x, num_mel_bins=40)
        return x

    def augmentation(self, x, min_snr, max_snr, p_noise):
        transforms = [
            RandomApply([PolarityInversion()], p=0.8),
            RandomApply([Noise(min_snr, max_snr)], p_noise),
            RandomApply([Gain()], p=0.3),
            RandomApply([Reverb(sample_rate=16000)], p=0.6),
        ]
        transf = ComposeMany(transforms, num_augmented_samples=1)
        transf_x = transf(x)

        return transf_x

    def generate_batch(self, batch):
        # TODO: the length of each batch is inconsistent
        xs, ys = zip(*batch)
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        ys = torch.LongTensor(ys)

        return xs, ys

if __name__ == "__main__":
    # spiking version
    train_dataset = SpikingDatasets(split="train")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1, 
                              collate_fn=train_dataset.generate_batch, pin_memory=True)
    
    for idx, (spect, target) in enumerate(train_loader):
        print(spect.shape, target.shape)

    # # origin non-spiking version
    # train_dataset = NonSpikingDatasets(split="train", aug=True)

    # train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1, 
    #                           collate_fn=train_dataset.generate_batch, pin_memory=True)
    
    # for idx, (spect, target) in enumerate(train_loader):
    #     print(spect.shape, target.shape)