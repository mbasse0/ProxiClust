import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, device):
        self.sequences = list(sequences)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.sequences[idx]
        label = self.labels[idx]
        return item, label
