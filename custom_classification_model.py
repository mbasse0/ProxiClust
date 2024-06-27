import torch
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer

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

class ESMWithDualHead(nn.Module):
    def __init__(self, device, input_dim, num_filters=256, output_dim=1):
        super(ESMWithDualHead, self).__init__()
        self.device = device
        # Load the pre-trained ESM model
        self.esm = AutoModelForMaskedLM.from_pretrained('models').to(device)
        # Instantiate the CNN decoder as the regression head
        self.regression_head = CNNDecoder(input_dim=input_dim, num_filters=num_filters, output_dim=output_dim)
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
        input_ids, attention_mask = self.tokenizer(sequences, return_tensors='pt', padding='longest', truncation=True, max_length=512).values()
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Get hidden states from ESM
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        
        # Use the last layer's hidden state for classification
        classification_logits = self.classification_head(hidden_states)  # using the [CLS] token's embeddings
        is_outlier = classification_logits > 0.5  # Binary classification threshold

        # Apply regression only to non-outliers
        regression_output = torch.zeros_like(classification_logits)
        if not is_outlier.all():  # Check if all are outliers
            non_outlier_indices = ~is_outlier.squeeze()
            regression_input = hidden_states[non_outlier_indices]
            regression_output[non_outlier_indices] = self.regression_head(regression_input)

        # Assign a fixed value (e.g., 5) for outliers
        regression_output[is_outlier] = 5.0

        # return classification_logits, regression_output
        return regression_output

