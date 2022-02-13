import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    input_size = 2

    def __init__(self, tag, size=1, x_0=1, y_0=1):
        super(DummyDataset, self).__init__()

        self.tasks = {
            'left': [0, 'toy1'],
            'right': [1, 'toy2']
        }

        self.data = torch.ones([size, 2], dtype=torch.float)
        self.data[:, 0] *= x_0
        self.data[:, 1] *= y_0

        self.target = torch.zeros((size, 2, 1)).unbind(dim=1)

    def __getitem__(self, index):
        return self.data[index], [t[index] for t in self.target]

    def __len__(self):
        return self.data.size(0)
