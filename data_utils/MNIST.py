from .Dataset import Dataset
import numpy as np
from neural.Tensor import Tensor


class MNIST(Dataset):
    def __init__(self, images_path, labels_path, name="MNIST"):
        super().__init__(name)

        with open(images_path, "rb") as f:
            f.read(16)  # skip header
            self.images = (
                np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784) / 255.0
            )

        with open(labels_path, "rb") as f:
            f.read(8)  # skip header
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    @staticmethod
    def collate_fn(batch):
        images = Tensor(np.stack([x[0] for x in batch]))
        labels = np.array([x[1] for x in batch])

        return images, labels
