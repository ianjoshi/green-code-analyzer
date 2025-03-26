import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DummyDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))
        

def blocking_data_loaders():

    # Create dataset
    dataset = DummyDataset()
    
    # Inefficient: Using blocking DataLoader without num_workers
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0  # Blocking I/O operations
    )
    
    return dataloader
