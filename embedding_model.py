from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


class EmbeddingESMWithMLP(nn.Module):
    def __init__(self, device, decoder_input_dim, output_dim=1):
        super(EmbeddingESMWithMLP, self).__init__()
        self.device = device
        # Load the pre-trained ESM model
        self.esm = AutoModelForMaskedLM.from_pretrained('models').to(device)
        # Instantiate the MLP
        self.decoder = MLP(input_dim=decoder_input_dim, output_dim=output_dim)
        self.tokenizer = AutoTokenizer.from_pretrained('models')

    
    def forward(self, pooled_hidden_states):
        # Forward pass through ESM model
        pooled_hidden_states = pooled_hidden_states.to(self.device)
        # Forward pass through MLP
        output = self.decoder(pooled_hidden_states)
        return output