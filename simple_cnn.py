from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, input_dim, num_filters, output_dim):
        super(SimpleCNN, self).__init__()
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
        x = x.float()  # Ensuring the input tensor is the correct type

        # Assuming x is of shape [batch_size, sequence_length, features]
        # Conv1D expects [batch_size, features, sequence_length]
        x = x.permute(0, 2, 1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x