import torch
from torch.utils.data import Dataset
# from torchvision import transforms


class Single(Dataset):
    def __init__(self, data):
        self.data = data  # e.g. X: MxT matrix

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        sample = self.data[:, idx]
        sample = torch.Tensor(sample)
        return sample, idx


class Pair(Dataset):
    def __init__(self, data):
        self.data = data  # e.g. X: MxT matrix

    def __len__(self):
        return self.data.shape[1] - 1

    def __getitem__(self, idx):
        sample = self.data[:, idx]
        sample2 = self.data[:, idx+1]
        sample = torch.Tensor(sample)
        sample2 = torch.Tensor(sample2)
        return sample, sample2, idx
