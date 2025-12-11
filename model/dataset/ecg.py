import os
import numpy as np
from sklearn.preprocessing import StandardScaler

class ECGSegLoader:
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = np.load(f"{data_path}/ECG_train.npy")
        print(f"Raw train data shape: {train_data.shape}")
        self.scaler.fit(train_data)
        self.train = self.scaler.transform(train_data)

        test_data = np.load(f"{data_path}/ECG_test.npy")
        print(f"Raw test data shape: {test_data.shape}")
        self.test = self.scaler.transform(test_data)

        self.test_labels = np.load(f"{data_path}/ECG_test_label.npy")
        print(f"Processed train shape: {self.train.shape}")
        print(f"Processed test shape: {self.test.shape}")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.zeros(self.win_size, dtype=np.float32)
        elif self.mode == "test":
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])