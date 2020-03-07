from torch import LongTensor
from torch.utils.data import Dataset


class NumberLoader(Dataset):
    def __init__(self, x, y, inp_len=3, out_len=3):
        if len(x) != len(y):
            raise ValueError("len(x) != len(y)")
        self.x = [[x[i + j] for j in range(inp_len)] for i in range(len(x) - inp_len + 1)]
        self.y = [[y[i + j] for j in range(out_len)] for i in range(len(y) - out_len + 1)]

    def __getitem__(self, index):
        return LongTensor(self.x[index]), LongTensor([0] + self.y[index])

    def __len__(self):
        return len(self.x)