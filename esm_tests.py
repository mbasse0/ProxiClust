import torch
import transformers
from tqdm import tqdm
# import accelerate
import pandas as pd
import accelerate

class FastaBatchedDataset(object):
    def __init__(self, sequence_labels, sequence_strs):
        self.sequence_labels = list(sequence_labels)
        self.sequence_strs = list(sequence_strs)

    @classmethod
    def from_file(cls, fasta_file):
        sequence_labels, sequence_strs = [], []
        cur_seq_label = None
        buf = []

        def _flush_current_seq():
            nonlocal cur_seq_label, buf
            if cur_seq_label is None:
                return
            sequence_labels.append(cur_seq_label)
            sequence_strs.append("".join(buf))
            cur_seq_label = None
            buf = []

        with open(fasta_file, "r") as infile:
            for line_idx, line in enumerate(infile):
                if line.startswith(">"):  # label line
                    _flush_current_seq()
                    line = line[1:].strip()
                    if len(line) > 0:
                        cur_seq_label = line
                    else:
                        cur_seq_label = f"seqnum{line_idx:09d}"
                else:  # sequence line
                    buf.append(line.strip())

        _flush_current_seq()

        assert len(set(sequence_labels)) == len(
            sequence_labels
        ), "Found duplicate sequence labels"

        return cls(sequence_labels, sequence_strs)

    def __len__(self):
        return len(self.sequence_labels)

    def __getitem__(self, idx):
        return self.sequence_labels[idx], self.sequence_strs[idx]

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        sizes = [(len(s), i) for i, s in enumerate(self.sequence_strs)]
        sizes.sort()
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0

        for sz, i in sizes:
            sz += extra_toks_per_seq
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)

        _flush_current_buf()
        return batches


# accelerator = accelerate.Accelerator()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
# device = accelerator.device


# MODEL_NAME = 'facebook/esm2_t48_15B_UR50D'

LOCAL_MODEL_PATH = 'models'


my_dataset = FastaBatchedDataset.from_file('unique_mutant_sequence.fasta')
batch_size = 32
dl = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
# dl = accelerator.prepare(dl)



# tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
# model = transformers.AutoModelForMaskedLM.from_pretrained(MODEL_NAME)#, low_cpu_mem_usage=True)

tokenizer = transformers.AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = transformers.AutoModelForMaskedLM.from_pretrained(LOCAL_MODEL_PATH).to(device)

model.eval()

WILDTYPE = 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'

WILDTYPE_TOK = tokenizer(WILDTYPE, return_tensors='pt', padding='longest').to(device)

outputs = []
for batch in tqdm(dl):
    labels, seqs = batch[0], batch[1]
    tok = tokenizer(seqs, return_tensors='pt', padding='longest').to(device)
    with torch.no_grad():
        input_ids, attention_mask = tok['input_ids'], tok['attention_mask']
        output = model(input_ids, attention_mask, output_hidden_states=True)
        hidden_states = output.hidden_states[-1]  # Get the last layer hidden-states
        print("hidden states shape", hidden_states.shape)
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden_states = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1)  # Count non-padded elements
        pooled_hidden_states = sum_hidden_states / sum_mask  # Mean over non-padded tokens

        outputs.append(pooled_hidden_states.cpu())

        # Optionally, print or further process your embeddings
        # print(print("pooled hidden states shape", pooled_hidden_states.shape))
        # Convert to a DataFrame
    embeds_to_save = outputs  # Move tensor to CPU
    torch.save(embeds_to_save, 'esm_embeddings.pt')
