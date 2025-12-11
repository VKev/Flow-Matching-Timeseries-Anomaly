import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
class PDSegLoader:
    def __init__(self, data_path, win_size, step, mode="train"):
        """
        Expects directory structure:
          data_path/
            train/power_data.pkl
            test/power_data.pkl
        """
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        fn = "power_data.pkl"
        train_path = os.path.join(data_path, "train", fn)
        test_path = os.path.join(data_path, "test", fn)

        tr_df = pd.DataFrame(pd.read_pickle(open(train_path, "rb")))
        te_df = pd.DataFrame(pd.read_pickle(open(test_path, "rb")))

        train_data = tr_df[[0]].to_numpy()
        self.scaler.fit(train_data)
        data = self.scaler.transform(train_data)

        test_data = te_df[[0]].to_numpy()
        test_label = te_df[1].to_numpy().flatten()

        print(f"Raw train data shape: {train_data.shape}")
        print(f"Raw test data shape: {test_data.shape}")

        self.test = self.scaler.transform(test_data)
        self.train = data
        self.test_labels = test_label

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
