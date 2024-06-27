import torch
from torch.utils.data import Dataset

class TokenizedDataset(Dataset):
    def __init__(self, tokenized_data, labels, device):
        self.input_ids = tokenized_data['input_ids'].to(device)
        self.attention_mask = tokenized_data['attention_mask'].to(device)
        self.labels = torch.tensor(labels).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        label = self.labels[idx]
        return input_id, attention_mask, label