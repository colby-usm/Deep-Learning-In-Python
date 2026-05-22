class Dataset:
    """
    Base dataset abstraction.

    Subclasses must implement:
        - __len__
        - __getitem__
    """

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
