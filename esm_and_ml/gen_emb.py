
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from plot_regression import plot_regression
import random 
import seaborn as sns
from itertools import combinations
import os
import datetime
import torch
import tqdm
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForMaskedLM, AutoTokenizer


# Load the dataframe
df =  pd.read_csv("CH65_SI06_processed.csv")

df["onehot"] = df["onehot"].apply(lambda x: [int(i) for i in x.replace('[','').replace(']','').split(', ')])
# df_sorted["mean_representation"] = df_sorted["mean_representation"].apply(lambda x: [float(i) for i in x.replace('[','').replace(']','').split(', ')])


antibody = "log10Kd_pinned"

print(antibody)
targets_sorted = np.array(df[antibody].to_list())
## Load the pdb file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def get_embeddings(model, tokenizer, seq_light, seq_heavy, df):
    # Tokenize the sequences
    sequences_heavy_tokens = tokenizer(seq_heavy, return_tensors='pt', padding='longest', truncation=True, max_length=512)
    sequences_light_tokens = tokenizer(seq_light, return_tensors='pt', padding='longest', truncation=True, max_length=512)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(sequences_heavy_tokens['input_ids'], sequences_heavy_tokens['attention_mask'], sequences_light_tokens['input_ids'], sequences_light_tokens['attention_mask'])
    loader = DataLoader(dataset, batch_size=150, shuffle=False)  # Set shuffle to False to maintain order
    
    # Initialize the 'mean_representation' column
    df["mean_representation"] = None
    
    model.eval()
    emb_list = []
    with torch.no_grad():
        idx = 0  # Initialize an index to keep track of the DataFrame row
        for batch in tqdm(loader):
            input_ids_h, attention_mask_h, input_ids_l, attention_mask_l = [b.to(model.device) for b in batch]
            
            output_light = model(input_ids=input_ids_l, attention_mask=attention_mask_l, output_hidden_states=True)
            hidden_light = output_light.hidden_states[-1].detach().cpu()
            embedding_light = torch.mean(hidden_light, dim=1)
            output_heavy = model(input_ids=input_ids_h, attention_mask=attention_mask_h, output_hidden_states=True)
            hidden_heavy = output_heavy.hidden_states[-1].detach().cpu()
            embedding_heavy = torch.mean(hidden_heavy, dim=1)
            mean_emb = (embedding_heavy + embedding_light) / 2
            # Assign the embeddings to the corresponding DataFrame rows
            for emb in mean_emb:
                # df.at[idx, 'mean_representation'] = emb.numpy().tolist()
                emb_list.append(emb.detach().numpy().tolist())
    df["mean_representation"] = emb_list
    print(emb_list)
    np.save("emb_list.npy", emb_list)
    return df




# def get_embeddings(model, tokenizer,seq_light, seq_heavy):
#     ids_h, mask_h = tokenizer(seq_heavy, return_tensors='pt', padding='longest', truncation=True, max_length=512).values()
#     ids_l, mask_l = tokenizer(seq_light, return_tensors='pt', padding='longest', truncation=True, max_length=512).values()
#     ids_l = ids_l.to(device)
#     mask_l = mask_l.to(device)
#     ids_h = ids_h.to(device)
#     mask_h = mask_h.to(device)

#     output_light = model(input_ids=ids_h, attention_mask=mask_h, output_hidden_states=True)
#     hidden_light = output_light.hidden_states[-1]
#     sum_hidden_states = torch.sum(hidden_light, dim=1)
#     embedding_light = sum_hidden_states / len(seq_light)

#     output_heavy = model(input_ids=ids_l, attention_mask=mask_l, output_hidden_states=True)
#     hidden_heavy = output_heavy.hidden_states[-1]
#     sum_hidden_states = torch.sum(hidden_heavy, dim=1)
#     embedding_heavy = sum_hidden_states / len(seq_heavy)

   
#     mean_emb =  (embedding_heavy + embedding_light)/2

#     return [mean_emb[i].detach().cpu().numpy().tolist() for i in range(mean_emb.shape[0])]
 

seq_list_light = df["mutant_light"].tolist()
seq_list_heavy = df["mutant_heavy"].tolist()

esm = AutoModelForMaskedLM.from_pretrained('models').to(device)
tokenizer = AutoTokenizer.from_pretrained('models')


new_df = get_embeddings(esm, tokenizer, seq_list_light, seq_list_heavy, df)

print("Done with new df")
new_df.to_csv("CH65_SI06_processed.csv", index=False)

print(new_df["mean_representation"])