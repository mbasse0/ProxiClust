from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from torch import nn
from custom_model3 import ESMWithCNN

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


class EmbeddingESMWithCNN(nn.Module):
    def __init__(self, device, decoder_input_dim, output_dim=1):
        super(EmbeddingESMWithCNN, self).__init__()
        self.device = device
        # Load the pre-trained ESM model
        self.esm = AutoModelForMaskedLM.from_pretrained('models').to(device)
    
        # Load full model state dict
        full_model_state_dict = torch.load('models/esm_with_cnn_4layers_trained_1800train_samples_4epochs_4batch_size.pt')
        
        # Extract the ESM state dict using a dictionary comprehension
        # Here we assume that the state dict keys for esm start with 'esm.'
        esm_state_dict = {k[len('esm.'):]: v for k, v in full_model_state_dict.items() if k.startswith('esm.')}
        
        # Load the ESM weights from the extracted part of the state dictionary
        self.esm.load_state_dict(esm_state_dict)
        
        # Instantiate the MLP
        self.decoder = CNNDecoder(input_dim=decoder_input_dim, num_filters=256, output_dim=output_dim)
        self.tokenizer = AutoTokenizer.from_pretrained('models')

    
    def forward(self, embeddings):
        # Forward pass through ESM model
        embeddings = embeddings.to(self.device)
        outputs = self.decoder(embeddings)
        # hidden_states = outputs.hidden_states[-1]
        # mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        # sum_hidden_states = torch.sum(hidden_states * mask_expanded, 1)
        # sum_mask = mask_expanded.sum(1)
        # pooled_hidden_states = sum_hidden_states / sum_mask
        # Forward pass through MLP
        # cnn_output = self.decoder(hidden_states)
        return outputs