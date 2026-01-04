import os
from random import shuffle

import numpy as np
import scipy.io as scio
from torch.utils.data import DataLoader, Dataset


class EEGDataset(Dataset):
    def __init__(self, data, labels):
        """
        Custom PyTorch Dataset for EEG data.
        Args:
            data (numpy.ndarray): The EEG data, shape [N, 1, channels, time].
            labels (numpy.ndarray): The corresponding labels, shape [N].
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single data-label pair at the given index.
        """
        return self.data[idx], self.labels[idx]


class AADWSDataset:
    def __init__(self, dataset, subject_id, fold_num=1):
        """
        Load datasets for AAD (Within-Subject), including KUL and DTU.
        Args:
            path (str): Path to the dataset directory.
            subject_id (str): Identifier for the subject.
            fold_num (int): Fold number for cross-validation (1-5).
            batch_size (int): Batch size for the DataLoader.
        """
        self.path = f"/datasets/AAD/{dataset}/1s/"
        self.subject_id = subject_id
        self.fold_num = fold_num
        self.data = None
        self.labels = None

        # Load the data during initialization
        self._load_data()

    def _load_data(self):
        """Loads EEG data and labels from .mat files."""
        data_file = os.path.join(self.path, f"data_{self.subject_id}.mat")
        label_file = os.path.join(self.path, f"label_{self.subject_id}.mat")

        self.data = scio.loadmat(data_file)[f"data_{self.subject_id}"]
        self.labels = scio.loadmat(label_file)[f"label_{self.subject_id}"]

        # Select the first 64 channels
        self.data = self.data[:, :, :64]

    def _split_data(self):
        """
        Splits the data into training, validation, and test sets using 5-fold cross-validation.
        Returns:
            tuple: Train, validation, and test datasets and their labels.
        """
        n_samples = self.data.shape[0]
        indices = list(range(n_samples))
        shuffle(indices)

        fold_size = n_samples // 5
        val_fold = self.fold_num - 1
        test_fold = self.fold_num % 5

        val_start = val_fold * fold_size
        test_start = test_fold * fold_size

        val_indices = indices[val_start : val_start + fold_size]
        test_indices = indices[test_start : test_start + fold_size]
        train_indices = [
            i for i in indices if i not in val_indices and i not in test_indices
        ]

        # Split the data and labels
        train_data, val_data, test_data = (
            self.data[train_indices],
            self.data[val_indices],
            self.data[test_indices],
        )
        train_labels, val_labels, test_labels = (
            self.labels[train_indices],
            self.labels[val_indices],
            self.labels[test_indices],
        )

        return train_data, train_labels, val_data, val_labels, test_data, test_labels

    @staticmethod
    def _preprocess_data(data, labels):
        """
        Preprocesses EEG data and labels for PyTorch compatibility.
        Args:
            data (numpy.ndarray): EEG data, shape [N, channels, time].
            labels (numpy.ndarray): Corresponding labels.

        Returns:
            tuple: Preprocessed data and labels.
        """
        # Normalize labels to start from 0
        labels = labels.squeeze() - 1

        # Reshape data to [N, 1, channels, time] for PyTorch
        data = np.expand_dims(data, axis=1).transpose(0, 1, 3, 2).astype(np.float32)

        return data, labels.astype(np.int64)

    def get_datasets(self):
        """
        Splits the dataset, preprocesses it, and creates PyTorch DataLoaders.
        Returns:
            tuple: Train, validation, and test DataLoaders.
        """
        # Split the data
        train_data, train_labels, val_data, val_labels, test_data, test_labels = (
            self._split_data()
        )

        # Preprocess the data
        train_data, train_labels = self._preprocess_data(train_data, train_labels)
        val_data, val_labels = self._preprocess_data(val_data, val_labels)
        test_data, test_labels = self._preprocess_data(test_data, test_labels)

        # Create PyTorch Datasets
        train_dataset = EEGDataset(train_data, train_labels)
        val_dataset = EEGDataset(val_data, val_labels)
        test_dataset = EEGDataset(test_data, test_labels)

        # # Create DataLoaders
        # train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        # test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Parameters
    subject_id = "1"
    fold_num = 5
    batch_size = 32

    aad = AADWSDataset(dataset="DTU", subject_id=subject_id, fold_num=fold_num)
    train_dataset, val_dataset, test_dataset = aad.get_datasets()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Example usage
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
        # print(targets)
