'''
Descripttion: Dataloader for Spiking Speech Commands
              References code from Sparch (https://github.com/idiap/sparch/tree/main)
version: 
Author: Shimin Zhang
Date: 2024-11-23 10:49:23
'''
import h5py
import numpy as np

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader


class SpikingDatasets(Dataset):
    def __init__(self, split, nb_steps=100, nb_units=700, max_time=1.4):
        self.device = "cpu"  # for memory allocation
        self.split_data = f'/datasets/kws/spiking_speech_commands/ssc_{split}.h5'
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


if __name__ == "__main__":
    train_dataset = SpikingDatasets(split="train")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1, 
                              collate_fn=train_dataset.generate_batch, pin_memory=True)
    
    for idx, (spect, target) in enumerate(train_loader):
        print(spect.shape, target.shape)
        