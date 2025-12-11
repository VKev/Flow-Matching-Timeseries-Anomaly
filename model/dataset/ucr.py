import os
import numpy as np
from sklearn.preprocessing import StandardScaler


class UCRSegLoader:
    def __init__(
        self,
        data_path,
        win_size,
        step,
        mode="train",
        train_only_normal=True,
        require_train_labels=True,
    ):
        """
        Files expected in `data_path`:
          - UCR_train.npy         # train time series (assumed normal)
          - UCR_test.npy          # test time series
          - UCR_test_label.npy    # labels for test time series (0/1)
          - optional UCR_train_label.npy to filter normal-only training
        """
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # ---------- Train (normal only) ----------
        train = np.load(os.path.join(data_path, "UCR_train.npy"))
        print(f"Raw UCR train data shape: {train.shape}")

        if train.ndim == 1:
            train = train[:, None]
        elif train.ndim > 2:
            raise ValueError(
                f"Unexpected UCR train ndim={train.ndim}, please reshape offline."
            )

        train_labels_path = os.path.join(data_path, "UCR_train_label.npy")
        if os.path.exists(train_labels_path):
            train_labels = np.load(train_labels_path).reshape(-1)
            if train_labels.shape[0] != train.shape[0]:
                raise ValueError(
                    f"Train labels length {train_labels.shape[0]} does not match train data {train.shape[0]}"
                )
            if train_only_normal:
                normal_mask = train_labels == 0
                train = train[normal_mask]
                print(
                    f"Filtered UCR train to normal only: {normal_mask.sum()}/{normal_mask.shape[0]}"
                )
        elif train_only_normal and require_train_labels:
            raise FileNotFoundError(
                f"UCR_train_label.npy not found in {data_path}. "
                "Provide train labels or disable label requirement."
            )

        self.scaler.fit(train)
        self.train = self.scaler.transform(train)

        # ---------- Test ----------
        test = np.load(os.path.join(data_path, "UCR_test.npy"))
        print(f"Raw UCR test data shape: {test.shape}")

        if test.ndim == 1:
            test = test[:, None]
        elif test.ndim > 2:
            raise ValueError(
                f"Unexpected UCR test ndim={test.ndim}, please reshape offline."
            )

        self.test = self.scaler.transform(test)

        # ---------- Labels ----------
        labels = np.load(os.path.join(data_path, "UCR_test_label.npy"))
        print(f"Raw UCR test label shape: {labels.shape}")
        self.test_labels = labels.reshape(-1).astype(np.float32)

        print(f"Processed UCR train shape: {self.train.shape}")
        print(f"Processed UCR test shape: {self.test.shape}")
        print(f"Processed UCR label shape: {self.test_labels.shape}")

    def __len__(self):
        if self.mode == "train":
            T = self.train.shape[0]
        elif self.mode == "test":
            T = self.test.shape[0]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return (T - self.win_size) // self.step + 1

    def __getitem__(self, index):
        idx = index * self.step

        if self.mode == "train":
            x = self.train[idx : idx + self.win_size]
            y = np.zeros(self.win_size, dtype=np.float32)
        elif self.mode == "test":
            x = self.test[idx : idx + self.win_size]
            y = self.test_labels[idx : idx + self.win_size]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return np.float32(x), np.float32(y)
