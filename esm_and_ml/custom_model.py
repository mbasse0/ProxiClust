
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


class ESMWithMLP(nn.Module):
    def __init__(self, device, decoder_input_dim, output_dim=1):
        super(ESMWithMLP, self).__init__()
        self.device = device
        # Load the pre-trained ESM model
        self.esm = AutoModelForMaskedLM.from_pretrained('models').to(device)
        # Instantiate the MLP
        self.decoder = MLP(input_dim=decoder_input_dim, output_dim=output_dim)
        self.tokenizer = AutoTokenizer.from_pretrained('models')

    
    def forward(self, sequences, attention_mask=None):
        def postprocess_sequence(tensor):
            # This function reverses the preprocessing done in preprocess_sequence
            # Assuming tensor is a 1D tensor of float32 dtype, where each element represents an encoded character
            sequence = ''.join(chr(int(c) + ord('A')) for c in tensor)
            return sequence

        sequences = [postprocess_sequence(sequence) for sequence in sequences]
        # Forward pass through ESM model
        input_ids, attention_mask = self.tokenizer(sequences, return_tensors='pt', padding='longest', truncation=True, max_length=512).values()
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device).float()
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
        sum_hidden_states = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        pooled_hidden_states = sum_hidden_states / sum_mask
        # Forward pass through MLP
        mlp_output = self.decoder(pooled_hidden_states.float())
        return mlp_output