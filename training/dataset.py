import numpy as np
import torch

class FireDataset(torch.utils.data.Dataset):
    def __init__(self, features_path, labels_path):
        self.X = np.load(features_path)
        self.y = np.load(labels_path)

    def __len__(self):
        return 1  # single sample (low RAM)

    def __getitem__(self, idx):
        x = torch.tensor(self.X).float()
        y = torch.tensor(self.y).float()
        return x, y