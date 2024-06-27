from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from torch import nn

class CNNDecoder(nn.Module):
    def __init__(self, input_dim, num_filters, output_dim):
        super(CNNDecoder, self).__init__()
        # Assuming input_dim is the number of features each input sequence element has
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters // 2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(num_filters // 2 * (input_dim // 4), output_dim)
        self.fc = nn.Linear(6400, output_dim)

    def forward(self, x):
        # Assuming x is of shape [batch_size, sequence_length, features]
        # Conv1D expects [batch_size, features, sequence_length]
        x = x.permute(0, 2, 1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ESMWithCNN(nn.Module):
    def __init__(self, device, input_dim, num_filters=256, output_dim=1):
        super(ESMWithCNN, self).__init__()
        self.device = device
        # Load the pre-trained ESM model
        self.esm = AutoModelForMaskedLM.from_pretrained('models').to(device)
        # Instantiate the CNN decoder
        self.decoder = CNNDecoder(input_dim=input_dim, num_filters=num_filters, output_dim=output_dim)
        self.tokenizer = AutoTokenizer.from_pretrained('models')

    def forward(self, sequences, attention_mask=None):
        def postprocess_sequence(tensor):
            # This function reverses the preprocessing done in preprocess_sequence
            # Assuming tensor is a 1D tensor of float32 dtype, where each element represents an encoded character
            sequence = ''.join(chr(int(c) + ord('A')) for c in tensor)
            return sequence

        sequences = [postprocess_sequence(sequence) for sequence in sequences]
        # Similar preprocessing as before
        input_ids, attention_mask = self.tokenizer(sequences, return_tensors='pt', padding='longest', truncation=True, max_length=512).values()
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        # mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        # sum_hidden_states = torch.sum(hidden_states * mask_expanded, 1)
        # sum_mask = mask_expanded.sum(1)
        # pooled_hidden_states = sum_hidden_states / sum_mask
        # # Forward pass through CNN
        # pooled_hidden_states = pooled_hidden_states.unsqueeze(1)
        cnn_output = self.decoder(hidden_states)
        return cnn_output
