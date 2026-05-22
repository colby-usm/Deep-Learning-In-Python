import queue
import threading
from .Dataset import Dataset


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
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

        self.queue = queue.Queue(maxsize=2 * loader.batch_size)
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
        """
        The producer is the background process which keeps the iterator's queue full
        """
        n = len(self.dataset)

        while self._idx < n:
            batch = []

            for _ in range(self.loader.batch_size):
                # this handles loading the batch. the queue will have batch_size elements at each idx
                if self._idx >= n:
                    break
                batch.append(self.dataset[self._idx])
                self._idx += 1

            self.queue.put(batch)

        self.queue.put(None)
