class Dataset:
    """
    Base dataset abstraction.

    Subclasses must implement:
        - __len__
        - __getitem__
    """

    def __init__(self, name="Dataset"):
        self.name = name

    @staticmethod
    def collate_fn(batch):
        return batch

    def __str__(self) -> str:
        return f"{self.name} with len()={len(self)})"

    def __getitem__(self, _):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
