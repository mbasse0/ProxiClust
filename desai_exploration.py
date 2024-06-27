import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
from torch import nn
from sklearn.model_selection import train_test_split

print('FINISHED LIBRARY IMPORTS')

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, device):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        # Tokenize the sequence
        # tokenized = self.tokenizer(sequence, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        # # Prepare the output as a dictionary
        # tokenized = {key: val.squeeze(0).to(self.device) for key, val in tokenized.items()}
        return sequence, torch.tensor(label).to(self.device)
print('Loading data')
df = pd.read_csv('df_Desai_15loci_complete.csv')
# TEMPORARY crop the dataset for faster runtime
df = df.iloc[:100]
df = df.drop(columns=['Unnamed: 0'])
# print(df.head())

# Assuming the first column is sequences and 'log10Kd_ACE2' is the target
sequences = df.iloc[:, 0].tolist()
print('sequences', sequences[:5], len(sequences))  
print("length of each sequence in amino acids: ", len(sequences[0]))
targets = df['log10Kd_ACE2'].values
print('targets', targets[:5], len(targets))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device', device)

print('Loading model and tokenizer')

tokenizer = AutoTokenizer.from_pretrained('models')
model = AutoModelForMaskedLM.from_pretrained('models').to(device)
model.eval()

def extract_embeddings_and_save(model, sequences, tokenizer, device, save_path):
    tok = tokenizer(sequences, return_tensors='pt', padding='longest').to(device)
    with torch.no_grad():
        input_ids, attention_mask = tok['input_ids'], tok['attention_mask']
        output = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = output.hidden_states[-1]
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden_states = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1)
        pooled_hidden_states = sum_hidden_states / sum_mask
    print('SAVING EMBEDDINGS')
    embeddings = pooled_hidden_states.cpu()
    torch.save(embeddings, save_path)
    return embeddings

# Extract embeddings for training dataset
esm_embeddings = extract_embeddings_and_save(model, sequences, tokenizer, device, 'train_embeddings.pt')

# MLP for prediction

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Instantiate and train the MLP
# Model instantiation
print("dimension of our embeddings is ", esm_embeddings.shape[1])
mlp = MLP(input_dim=esm_embeddings.shape[1]).to(device)  # assuming ESM model's output dimension is 768
# Loss function
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
# Optimizer
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs = inputs['input_ids']  # Adjust based on your dataset handling
            
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

        # Validation phase
        validate_model(model, test_loader, criterion)

def validate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs = inputs['input_ids']  # Adjust based on your dataset handling
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.unsqueeze(1).float())
            total_loss += loss.item()
    print(f"Validation Loss: {total_loss/len(test_loader)}")


print('TRAINING MLP MODEL')
# Run training and validation

# Split data into training and test sets
train_sequences, test_sequences, train_labels, test_labels = train_test_split(
    esm_embeddings, targets, test_size=0.2, random_state=42
)

# Create train and test datasets
train_dataset = SequenceDataset(train_sequences, train_labels, tokenizer, device)
test_dataset = SequenceDataset(test_sequences, test_labels, tokenizer, device)

# # Create DataLoaders for train and test datasets
# def collate_fn(batch):
#     seqs, labels = zip(*batch)
#     return list(seqs), torch.tensor(labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_model(mlp, train_loader, test_loader, criterion, optimizer, num_epochs=5)


# Save the model
torch.save(mlp.state_dict(), 'mlp_model.pt')
print('Model saved')

# Test the mlp model with Pearson correlation
