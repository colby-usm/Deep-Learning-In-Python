import queue
import threading
from .Dataset import Dataset
import numpy as np


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size=1,
        shuffle=(False, 42),
        num_workers=0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle, self.shuffle_seed = shuffle
        self.num_workers = num_workers

        if self.num_workers > 0:
            raise NotImplementedError("multi-worker not implemented")

    def __iter__(self):
        return _SingleProcessDataLoaderIter(self)


class _SingleProcessDataLoaderIter:
    """
    A class that handles loading data with a single background worker for the Dataloader
    """

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.dataset = loader.dataset

        self.queue = queue.Queue(maxsize=loader.batch_size)
        self._idx = 0

        self.thread = threading.Thread(target=self._producer, daemon=True)
        self.thread.start()

    def __iter__(self):
        """
        How a loop accesses the loader
        """
        return self

    def __next__(self):
        """
        When the for loop asks for the next element
        """
        item = self.queue.get()

        if item is None:
            raise StopIteration

        return item

    def _producer(self):
        n = len(self.dataset)
        indices = (
            np.random.default_rng(self.loader.shuffle_seed).permutation(n)
            if self.loader.shuffle
            else np.arange(n)
        )
        for start in range(0, n, self.loader.batch_size):
            batch_indices = indices[start : start + self.loader.batch_size]
            batch = [self.dataset[i] for i in batch_indices]
            self.queue.put(batch)
        self.queue.put(None)
