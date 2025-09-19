"""
Defines the PyTorch Dataset for loading sensor data.
"""
import torch
from torch.utils.data import Dataset
from typing import Dict

class SensorDataset(Dataset):
    """PyTorch Dataset for sensor time-series data."""
    def __init__(self, features, labels, label_map: Dict[int, int], is_2d: bool = False):
        # features are expected to be a list of numpy arrays
        features_tensor = torch.tensor(features, dtype=torch.float32)

        if is_2d:
            # For 2D CNN, shape should be (N, 1, H, W)
            # H=num_sensors, W=num_timesteps
            self.features = features_tensor.unsqueeze(1)
        else:
            # For 1D CNN, shape should be (N, 1, L)
            self.features = features_tensor.unsqueeze(1)

        self.labels = torch.tensor(labels, dtype=torch.long)
        self.label_map = label_map
        self.labels = torch.tensor([self.label_map[l.item()] for l in self.labels], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]