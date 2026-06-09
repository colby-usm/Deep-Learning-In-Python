from .Dataset import Dataset
import numpy as np
from neural.Tensor import Tensor


class MNIST_VIT(Dataset):
    def __init__(self, images_path, labels_path, patch_size=7, name="MNIST"):
        super().__init__(name)

        self.patch_size = patch_size

        with open(images_path, "rb") as f:
            f.read(16)
            self.images = (
                np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28) / 255.0
            )

        with open(labels_path, "rb") as f:
            f.read(8)
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)

    def __len__(self):
        return len(self.images)

    def _to_patches(self, img):
        """Convert (28,28) -> (num_patches, patch_dim)"""
        p = self.patch_size
        H, W = img.shape

        patches = []
        for i in range(0, H, p):
            for j in range(0, W, p):
                patch = img[i : i + p, j : j + p].reshape(-1)
                patches.append(patch)

        return np.stack(patches)  # (num_patches, patch_dim)

    def __getitem__(self, idx):
        img = self.images[idx]  # (28,28)
        img = self._to_patches(img)  # (16,49)
        label = self.labels[idx]
        return img, label

    @staticmethod
    def collate_fn(batch):
        images = Tensor(np.stack([x[0] for x in batch]))  # (B, N, P)
        labels = np.array([x[1] for x in batch], dtype=np.int32)

        one_hot = np.zeros((len(labels), 10), dtype=np.float32)
        one_hot[np.arange(len(labels)), labels] = 1.0

        return images, one_hot
