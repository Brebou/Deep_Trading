import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


class Dataset(TensorDataset):
    def __init__(self, data_tensor, target_tensor, context_length):
        self.context_length = context_length
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __len__(self):
        return len(self.data_tensor) - self.context_length
    
    def __getitem__(self, idx):
        x = self.data_tensor[idx:idx + self.context_length]
        y = self.target_tensor[idx + 1: idx + self.context_length + 1]
        return x, y
    
def dataset_from_numpy(data_numpy, target_numpy, context_length, batch_size, shuffle = True):
    data_tensor = torch.tensor(data_numpy, dtype = torch.float32)
    target_tensor = torch.tensor(target_numpy, dtype = torch.float32)
    dataset = Dataset(data_tensor, target_tensor, context_length)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
    return dataloader