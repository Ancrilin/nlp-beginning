from torch.utils.data import Dataset
import torch
import numpy as np


class MyDataset(Dataset):
    def __init__(self, dataset, config):
        self.dataset = np.array(dataset)
        self.config = config

    def __getitem__(self, item: int):
        label, sentence1, sentence2, seq1, seq2 = self.dataset[item]
        return (torch.tensor(sentence1),
                torch.tensor(sentence2),
                torch.tensor(label),
                torch.tensor(seq1),
                torch.tensor(seq2),
                )

    def __len__(self):
        return len(self.dataset)