from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from torch import nn

'''
class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        # Feed-forward network
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        src = src + self.dropout1(self.self_attn(src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0])
        src2 = self.norm2(src)
        src = src + self.dropout2(self.linear2(self.dropout(torch.relu(self.linear1(src2)))))
        return src
'''


class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads):
        super(TransformerDecoder, self).__init__()
        self.transformer_layers = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=input_dim,
            dropout=0.1
        )
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_layers, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output = self.transformer_decoder(x, x)  # Self-attention in the decoder

        # Mean pooling to get only one value for the whole sequence
        output = output.mean(dim=0)
        return self.output_layer(output)


class ESMWithTransformer(nn.Module):
    def __init__(self, device, decoder_input_dim):
        super(ESMWithTransformer, self).__init__()
        self.device = device
        # Load the pre-trained ESM model
        self.esm = AutoModelForMaskedLM.from_pretrained('models').to(device)
        # Instantiate the TransformerDecoder
        self.decoder = TransformerDecoder(input_dim=decoder_input_dim, hidden_dim=1280, output_dim=1, num_layers=2, num_heads=4)
        self.tokenizer = AutoTokenizer.from_pretrained('models')

    def forward(self, sequences, attention_mask=None):
        # Forward pass through ESM model
        input_ids, attention_mask = self.tokenizer(sequences, return_tensors='pt', padding='longest', truncation=True, max_length=512).values()
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]

        # Adjusting the shape to [Seq Len, Batch, Features] for Transformer
        hidden_states = hidden_states.permute(1, 0, 2)  # Permute to match Transformer input expectations


        # mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        # sum_hidden_states = torch.sum(hidden_states * mask_expanded, 1)
        # sum_mask = mask_expanded.sum(1)
        # pooled_hidden_states = sum_hidden_states / sum_mask
        # print("pooled_hidden states shape", pooled_hidden_states.shape)
        # Forward pass through TransformerDecoder
        decoder_output = self.decoder(hidden_states)

        return decoder_output
