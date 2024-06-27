
import torch
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer


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

class ESMWithDualHead(nn.Module):
    def __init__(self, device, input_dim, num_filters=256, output_dim=1):
        super(ESMWithDualHead, self).__init__()
        self.device = device
        # Load the pre-trained ESM model
        self.esm = AutoModelForMaskedLM.from_pretrained('models').to(device)
        # Instantiate the CNN decoder as the regression head
        self.decoder = MLP(input_dim=input_dim)
        # Classification head to determine outliers
        self.classification_head = nn.Sequential(
            nn.Linear(input_dim, 256),  # Adjust input_dim as necessary
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.tokenizer = AutoTokenizer.from_pretrained('models')

    def forward(self, sequences, attention_mask=None):
        # Tokenize and encode sequences
        def postprocess_sequence(tensor):
            # This function reverses the preprocessing done in preprocess_sequence
            # Assuming tensor is a 1D tensor of float32 dtype, where each element represents an encoded character
            sequence = ''.join(chr(int(c) + ord('A')) for c in tensor)
            return sequence

        sequences = [postprocess_sequence(sequence) for sequence in sequences]
        
        input_ids, attention_mask = self.tokenizer(sequences, return_tensors='pt', padding='longest', truncation=True, max_length=512).values()
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Get hidden states from ESM
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        
        # Use the last layer's hidden state for classification
        classification_logits = self.classification_head(hidden_states[:,0,:])  # using the [CLS] token's embeddings
        is_outlier = classification_logits > 0.5  # Binary classification threshold

        # Apply regression only to non-outliers
        regression_output = torch.zeros_like(classification_logits)
        non_outlier_indices = ~is_outlier.squeeze()
        if not is_outlier.all():  # Check if all are outliers
    
            regression_input = hidden_states[non_outlier_indices]
            regression_output[non_outlier_indices] = self.regression_head(regression_input)

        # Assign a fixed value (e.g., 5) for outliers
        regression_output[is_outlier] = 5.0

        # return classification_logits, regression_output
        return regression_output

