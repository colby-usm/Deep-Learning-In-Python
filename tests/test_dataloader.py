import time
from data_utils.DataLoader import DataLoader
from data_utils.Dataset import Dataset


class SimpleListDataset(Dataset):
    def __init__(self, list):
        self.data = list

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


dataset = SimpleListDataset(["a", "b", "c", "d", "e", "f", "g"])

dataloader = DataLoader(dataset, batch_size=4)

for i in dataloader:
    print(i)
    time.sleep(2)
