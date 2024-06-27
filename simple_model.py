from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from torch import nn


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        x = x.float()  # Ensuring the input tensor is the correct type
        output = self.layers(x)
        output = output.mean(dim=1)  # Average pooling over the sequence length
        return output