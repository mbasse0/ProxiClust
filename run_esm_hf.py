# srun -p gpu-rtx6k -A h2lab --time=3-00:00:00 -c 32 --mem=256G --gres=gpu:4 --pty /bin/bash
# conda activate improve
# accelerate launch --num_processes 4 --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_backward_prefetch_policy BACKWARD_PRE --fsdp_offload_params false --fsdp_sharding_strategy 1 --fsdp_state_dict_type FULL_STATE_DICT --fsdp_transformer_layer_cls_to_wrap EsmLayer run_esm_hf.py

import argparse
import json
import os
import pickle
from tqdm import tqdm
import torch
import transformers
import accelerate

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='1v', choices=['1v', '2'])
    parser.add_argument('--model_size', type=str, default='650m', choices=['650m', '15b'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--ds_name', type=str, required=True)
    parser.add_argument('--file_name', type=str, required=True)
    parser.add_argument('--grammaticality', default=False, action='store_true')
    args = parser.parse_args()
    return args
args = get_args()

MODEL_NAME = None
if args.version == '1v':
    if args.model_size == '650m':
        MODEL_NAME = 'facebook/esm1v_t33_650M_UR90S_1'
elif args.version == '2':
    if args.model_size == '650m':
        MODEL_NAME = 'facebook/esm2_t33_650M_UR50D'
    elif args.model_size == '15b':
        MODEL_NAME = 'facebook/esm2_t48_15B_UR50D'

# # This is for runs 0,1,2,3
# MUTATIONS = [0, 32, 34, 36, 78, 101, 107, 138, 139, 145, 154, 157, 159, 162, 166]
# This is for runs 4,5,6
WILDTYPE = 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'

accelerator = accelerate.Accelerator()
device = accelerator.device

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

ds = FastaBatchedDataset.from_file(f'data/{args.ds_name}/{args.file_name}.fasta')
dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
dl = accelerator.prepare(dl)

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = transformers.AutoModelForMaskedLM.from_pretrained(MODEL_NAME)#, low_cpu_mem_usage=True)
model.eval()
model = accelerator.prepare(model)

WILDTYPE_TOK = tokenizer(WILDTYPE, return_tensors='pt', padding='longest').to(device)

outputs = []
for batch in tqdm(dl):
    labels, seqs = batch[0], batch[1]
    tok = tokenizer(seqs, return_tensors='pt', padding='longest').to(device)
    with torch.no_grad():
        input_ids, attention_mask = tok.input_ids.clone(), tok.attention_mask.clone()
        # if args.grammaticality:
        #     for b in range(len(seqs)):
        #         mutations = [1 + i for i, (x, y) in enumerate(zip(seqs[b], WILDTYPE)) if x != y] # +1 because of the [CLS] token
        #         input_ids[b, mutations] = tokenizer.mask_token_id
        output = model(input_ids, attention_mask, output_hidden_states=True)
        last_hidden_states = output.hidden_states[-1] # (B, L, D)
        cls_reps = last_hidden_states[:, 0, :].tolist() # (B, D)
        mean_reps = last_hidden_states[:, 1:-1, :].mean(dim=1).tolist() # (B, D)
        max_reps = last_hidden_states[:, 1:-1, :].max(dim=1).values.tolist() # (B, D)
        # mean15_reps = torch.index_select(last_hidden_states[:, 1:, :], dim=1, index=torch.tensor(MUTATIONS, device=last_hidden_states.device)).mean(dim=1).tolist() # (B, D)
        mean_delta_reps = []
        max_delta_reps = []
        for b in range(len(seqs)):
            delta_mask = torch.tensor([int(x != y) for x, y in zip(seqs[b], WILDTYPE)], device=last_hidden_states.device)
            if delta_mask.sum() == 0:
                print(f'WARNING: skipping a sequence because it is identical to WILDTYPE')
                mean_delta_rep = None
                max_delta_rep = None
            else:
                mean_delta_rep = ((last_hidden_states[b, 1:-1, :] * delta_mask[:, None]).sum(dim=0) / delta_mask.sum()).tolist()
                max_delta_rep = (last_hidden_states[b, 1:-1, :] * delta_mask[:, None]).max(dim=0).values.tolist()
            mean_delta_reps.append(mean_delta_rep)
            max_delta_reps.append(max_delta_rep)
        logits = output.logits # (B, L, V)
        probsss = torch.softmax(logits, dim=-1) # (B, L, V)
        probss = probsss.gather(dim=-1, index=tok.input_ids.unsqueeze(-1)).squeeze(-1) # (B, L)
        wildtype_probss = probsss.gather(dim=-1, index=WILDTYPE_TOK.input_ids.expand(len(seqs), -1).unsqueeze(-1)).squeeze(-1) # (B, L)

        if args.grammaticality:
            for b in range(len(seqs)):
                mutations = [1 + i for i, (x, y) in enumerate(zip(seqs[b], WILDTYPE)) if x != y] # +1 because of the [CLS] token
                new_input_ids = input_ids[b].clone().unsqueeze(0).expand(len(mutations), -1) # (M, L)
                new_attention_mask = attention_mask[b].clone().unsqueeze(0).expand(len(mutations), -1) # (M, L)
                for m, mutation in enumerate(mutations):
                    new_input_ids[m, mutation] = tokenizer.mask_token_id
                new_output = model(new_input_ids, new_attention_mask, output_hidden_states=True)
                new_logits = new_output.logits # (M, L, V)
                new_probsss = torch.softmax(new_logits, dim=-1) # (M, L, V)
                for m, mutation in enumerate(mutations):
                    probss[b, mutation] = new_probsss[m, mutation, tok.input_ids[b, mutation]]
                    wildtype_probss[b, mutation] = new_probsss[m, mutation, WILDTYPE_TOK.input_ids[0, mutation]]

        probsss = probsss[:, 1:-1, :].tolist()
        probss = probss[:, 1:-1].tolist()
        wildtype_probss = wildtype_probss[:, 1:-1].tolist()
    for label, seq, cls_rep, mean_rep, max_rep, mean_delta_rep, max_delta_rep, probs, wildtype_probs, all_probs in zip(labels, seqs, cls_reps, mean_reps, max_reps, mean_delta_reps, max_delta_reps, probss, wildtype_probss, probsss):
        if args.grammaticality:
            output = {
                'label': label,
                'sequence': seq,
                'probs': probs,
                'wildtype_probs': wildtype_probs,
                # 'all_probs': all_probs,
            }
        else:
            output = {
                'label': label,
                'sequence': seq,
                'cls_representation': cls_rep,
                'mean_representation': mean_rep,
                'max_representation': max_rep,
                'mean_delta_representation': mean_delta_rep,
                'max_delta_representation': max_delta_rep,
            }
        outputs.append(output)

local_rank = accelerator.state.local_process_index
with open(f'data/{args.ds_name}/{"single_wt_masked_prob." if args.grammaticality else ""}{args.file_name}_emb_esm{args.version}_{args.model_size}.jsonl.{local_rank}', 'w') as f:
    for output in outputs:
        f.write(json.dumps(output) + '\n')

accelerator.wait_for_everyone()
if accelerator.is_main_process:
    outputs = []
    for i in range(accelerator.state.num_processes):
        with open(f'data/{args.ds_name}/{"single_wt_masked_prob." if args.grammaticality else ""}{args.file_name}_emb_esm{args.version}_{args.model_size}.jsonl.{i}', 'r') as f:
            outputs += [json.loads(line) for line in f]
    def _get_key(x):
        if '|' in x['label'] or '_' in x['label']:
            return int(x['label'].split('|')[0]) if '|' in label else int(x['label'].split('_')[1])
        return int(x['label'])
    outputs = sorted(outputs, key=_get_key)

    # de-duplucate by key
    exists = set()
    new_outputs = []
    for output in outputs:
        if output['label'] not in exists:
            exists.add(output['label'])
            new_outputs.append(output)
    outputs = new_outputs
    
    with open(f'data/{args.ds_name}/{"single_wt_masked_prob." if args.grammaticality else ""}{args.file_name}_emb_esm{args.version}_{args.model_size}.jsonl', 'w') as f:
        for output in outputs:
            f.write(json.dumps(output) + '\n')
    with open(f'data/{args.ds_name}/{"single_wt_masked_prob." if args.grammaticality else ""}{args.file_name}_emb_esm{args.version}_{args.model_size}.pkl', 'wb') as f:
        pickle.dump(outputs, f, protocol=4)
    for i in range(accelerator.state.num_processes):
        os.remove(f'data/{args.ds_name}/{"single_wt_masked_prob." if args.grammaticality else ""}{args.file_name}_emb_esm{args.version}_{args.model_size}.jsonl.{i}')

# all_seqs = ['MQIFVKTLTGKTITLEVEPSTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG']

# for i in range(0, len(all_seqs), args.batch_size):
#     batch_seqs = all_seqs[i:i+args.batch_size]
#     tok = tokenizer(batch_seqs, return_tensors='pt').to(device)
#     with torch.no_grad():
#         last_hidden_states = model(**tok, output_hidden_states=True).hidden_states[-1] # (B, L, D)
#         representations = last_hidden_states[:, 0, :] # (B, D)
#     for seq, rep in zip(batch_seqs, representations):
#         print(seq, rep)
