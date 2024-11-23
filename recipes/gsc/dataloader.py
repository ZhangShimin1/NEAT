import os
import os.path
import urllib.request
import tarfile
import shutil

import torch
import torch.nn as nn
import torchaudio
import torch.utils.data as data
from torchaudio_augmentations import ComposeMany
from torchaudio_augmentations import Gain
from torchaudio_augmentations import Noise
from torchaudio_augmentations import PolarityInversion
from torchaudio_augmentations import RandomApply
from torchaudio_augmentations import Reverb


def get_keywords(version, if_command, split_dir):
    # excluded _background_noise_ folder
    all_classes = [keyword for keyword in os.listdir(split_dir) if keyword != '_background_noise_']
    if version == 1:
        if if_command:
            keywords = {'stop': 0, 'go': 1, 'yes': 2, 'no': 3, 'up': 4, 'down': 5, 'left': 6, 'right': 7, 'on': 8, 'off': 9}
        else:
            keywords = {value: index for index, value in enumerate(all_classes)}
    elif version == 2:
        if if_command:
            keywords = {'stop': 0, 'go': 1, 'yes': 2, 'no': 3, 'up': 4, 'down': 5, 'left': 6, 'right': 7, 'on': 8, 'off': 9,
                        'backward': 10, 'forward': 11, 'Follow': 12, 'Learn': 13}
        else:
            keywords = {value: index for index, value in enumerate(all_classes)}
    
    return keywords

def is_audio_file(filename):
    AUDIO_EXTENSIONS = ['.wav', '.WAV']  # check extensions

    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

def check_data_process(version):
    data_dir = f'/datasets/kws/google_speech_command_{version}'
    
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    raw_data_dir = data_dir + '/raw'
    if os.path.isdir(raw_data_dir):
        print(f"GSC-v{version} already downloaded")
    else:
        print(f"Downloading GSC-v{version}...")
        os.mkdir(raw_data_dir)
        download(raw_data_dir, version=version)
        print(f"GSC-v{version} download finished")

    processed_data_dir = data_dir + '/processed'
    if os.path.isdir(processed_data_dir):
        print(f"GSC-v{version} already processed")
    else:
        os.mkdir(processed_data_dir)
        make_dataset(raw_data_dir, processed_data_dir)
        print("Finished processing data")

def download(raw_data_save_dir, version):
    url = f'http://download.tensorflow.org/data/speech_commands_v0.0{version}.tar.gz'
    tarname = raw_data_save_dir + f'/google_speech_commands_v{version}.tar.gz'
    urllib.request.urlretrieve(url, tarname)
    # unzip
    tar = tarfile.open(tarname, "r:gz")
    tar.extractall(raw_data_save_dir)
    tar.close()

def move_files(original_fold, data_fold, data_filename):
    with open(data_filename) as f:
        for line in f.readlines():
            vals = line.split('/')
            dest_fold = os.path.join(data_fold, vals[0])
            if not os.path.exists(dest_fold):
                os.mkdir(dest_fold)
            shutil.move(os.path.join(original_fold, line[:-1]), os.path.join(data_fold, line[:-1]))

def create_train_fold(original_fold, data_fold, test_fold):
    # list dirs
    dir_names = list()
    for file in os.listdir(original_fold):
        if os.path.isdir(os.path.join(original_fold, file)):
            dir_names.append(file)

    # build train fold
    for file in os.listdir(original_fold):
        if os.path.isdir(os.path.join(original_fold, file)) and file in dir_names:
            shutil.move(os.path.join(original_fold, file), os.path.join(data_fold, file))

def make_dataset(gcommands_fold, out_path):
    validation_path = os.path.join(gcommands_fold, 'validation_list.txt')
    test_path = os.path.join(gcommands_fold, 'testing_list.txt')

    valid_fold = os.path.join(out_path, 'valid')
    test_fold = os.path.join(out_path, 'test')
    train_fold = os.path.join(out_path, 'train')

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(valid_fold):
        os.mkdir(valid_fold)
    if not os.path.exists(test_fold):
        os.mkdir(test_fold)
    if not os.path.exists(train_fold):
        os.mkdir(train_fold)

    move_files(gcommands_fold, test_fold, test_path)
    move_files(gcommands_fold, valid_fold, validation_path)
    create_train_fold(gcommands_fold, train_fold, test_fold)

def index_commands(split_dir, categ_to_idx):
    spects = []
    all_classes = [keyword for keyword in os.listdir(split_dir) if keyword != '_background_noise_']
    for command in sorted(all_classes):
        command_dir = os.path.join(split_dir, command)
        for root, _, fnames in sorted(os.walk(command_dir)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, categ_to_idx.get(command, 0))
                    spects.append(item)
    
    return spects


class datasets(data.Dataset):
    def __init__(self, split, version, if_command=True, aug=False, aug_paras=[0.0001, 0.9, 0.1]):
        check_data_process(version)
        self.split = split
        self.split_dir = f'/datasets/kws/google_speech_command_{version}/processed/{split}'
        self.class_index = get_keywords(version, if_command, self.split_dir)
        self.spects = index_commands(self.split_dir, self.class_index)
        self.aug = aug
        self.aug_paras = aug_paras

    def __getitem__(self, index):
        path, command_index = self.spects[index]
        spect = self.frontend(path, self.aug)
        
        return spect, command_index

    def __len__(self):
        
        return len(self.spects)

    def frontend(self, path, aug=True):
        x, _ = torchaudio.load(path)
        if aug and self.split == "train":
            min_snr, max_snr, p_noise = self.aug_paras
            x = self.augmentation(x, min_snr, max_snr, p_noise).squeeze(0)
        
        x = torchaudio.compliance.kaldi.fbank(x, num_mel_bins=40)
        if x.shape[0] > 98:
            print(path)
        x = self.pad_seq(x)

        return x

    def pad_seq(self, data, len=98):
        if data.shape[0] < len:
            padding = len - data.shape[0]
            padded_data = torch.nn.functional.pad(data, (0, 0, 0, padding))
        else:
            padded_data = data

        return padded_data

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


if __name__ == "__main__":
    train_dataset = datasets(split="train", version=2, if_command=True, aug=False)
    
    train_loader = data.DataLoader(train_dataset, batch_size=256, shuffle=True, 
                                   num_workers=1, pin_memory='cpu', sampler=None)
    
    for ids, (spect, target) in enumerate(train_loader):
        print(spect.shape, target.shape)