import torch
import pandas as pd
import os
import urllib.request
from torch.utils.data import Dataset

class AirPassengersDataset(Dataset):
    def __init__(self, url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv", 
                 seq_len=24, context_len=12, train=True, data_root="data", name="airline-passengers"):
        self.seq_len = seq_len
        self.context_len = context_len
        
        # Ensure data directory exists
        self.data_dir = os.path.join(data_root, name)
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.file_path = os.path.join(self.data_dir, "airline-passengers.csv")
        
        if not os.path.exists(self.file_path):
            print(f"Downloading dataset from {url} to {self.file_path}...")
            urllib.request.urlretrieve(url, self.file_path)
        
        df = pd.read_csv(self.file_path, usecols=[1], engine='python')
        data = df.values.astype('float32')
        
        # Normalize
        self.mean = data.mean()
        self.std = data.std()
        data = (data - self.mean) / self.std
        
        self.data = torch.tensor(data)
        
        # Split train/test (simple 80/20 split)
        split_idx = int(len(self.data) * 0.8)
        if train is True:
            self.data = self.data[:split_idx]
        elif train is False:
            self.data = self.data[split_idx:]
        else:
            # train is None or other -> use full data
            pass
            
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        # We want to forecast the NEXT value given context
        # But for flow matching, we usually model a distribution.
        # Let's say we predict x_{k} given x_{k-context_len...k-1}
        # For simplicity in this "Tiny" example, let's just predict 1 step ahead.
        
        window = self.data[idx : idx + self.context_len + 1]
        context = window[:-1]
        target = window[-1]
        
        return context, target
