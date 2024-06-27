from custom_model import ESMWithMLP
from sequence_dataset import SequenceDataset
import torch
from captum.attr import LayerIntegratedGradients
import pandas as pd
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)



def compute_token_importance(model, tokenizer, baseline_ids, sequence):
    """
    Compute the token importance for each token in a sequence using Integrated Gradients.

    Args:
    - model (torch.nn.Module): The model that you want to interpret.
    - tokenizer: The tokenizer used for converting the sequence to token ids.
    - sequence (str): The input text sequence for which to compute token importances.

    Returns:
    - token_importances (list of tuples): A list of tuples containing tokens and their corresponding importances.
    """

    # Tokenize the input sequence and convert to tensor
    model.eval()
    input_ids = tokenizer.encode(sequence, add_special_tokens=True, return_tensors='pt')
    

    # Initialize Integrated Gradients
    lig = LayerIntegratedGradients(model, model.esm.esm.embeddings)
    
    # Compute attributions using Layer Integrated Gradients
    attributions = lig.attribute(input_ids,
                                 baselines=baseline_ids,
                                 target=0,
                                 return_convergence_delta=False)

    # Sum attributions across the embedding dimension if necessary
    attributions = attributions.sum(dim=-1).squeeze(0)

    # Normalize attributions for each token
    attributions = attributions / torch.norm(attributions)

    # Convert token ids back to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    # Pair tokens with their corresponding attributions
    token_importances = list(zip(tokens, attributions.cpu().detach().numpy()))

    return token_importances


df = pd.read_csv('desai_short.csv')
seq_id = -1
seq_len = 50
example_seq = df['mutant_sequence'].iloc[seq_id]
example_seq = example_seq[:seq_len]
print("example sequence to find importance")

model = ESMWithMLP(device=device, decoder_input_dim=1280).to(device)

model.load_state_dict(torch.load('esm_with_mlp_2layers_trained_50train_samples_10epochs_4batch_size.pt'))


# Generate a baseline (e.g., zero embedding)
first_seq = df["mutant_sequence"].iloc[0]
baseline = first_seq[:seq_len]
baseline_ids = model.tokenizer(baseline, return_tensors='pt')['input_ids']
print("baseline ids", baseline_ids)
# Remove the last two dimensions to account for the extra tokens [CLS] and [SEP]

importance = compute_token_importance(model, model.tokenizer, baseline_ids, example_seq)

print("Importances of tokens:", importance)

#Plot the importances

# Unzip the token-importance tuples into separate lists
tokens, importances = zip(*importance)

# Plot the bar chart
plt.figure(figsize=(10, 6))
for token, importance in importance:
    plt.bar(token, importance)
plt.xlabel('Token')
plt.ylabel('Importance')
plt.title('Token Importances')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels

plt.savefig('token_importances.png')
