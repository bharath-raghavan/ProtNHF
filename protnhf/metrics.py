import math
import numpy as np
import random
from collections import Counter
import torch
import esm
from .dataset import Data

def reversibilty(model):
    num_points = random.randint(1, 200)
    num_batch = random.randint(1, 5)
    h = torch.randint(low=0, high=model.n_types, size=(num_points*num_batch,))
    batch = torch.cat([torch.zeros(num_points)+i for i in range(num_batch)])
    data = Data(h, batch)
    with torch.no_grad():
        p0, q0, qT_original = model.forward(data, train=False)
        pT, qT = model.reverse(p0, q0, data.batch)
    return torch.abs(qT_original-qT).max()

class ESM2Handle:
    def __init__(self):
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model.eval()  # set to evaluation mode
        self.batch_converter = self.alphabet.get_batch_converter()
    
    def get_pppl(self, seq):
        # ESM expects (name, sequence) tuples
        data = [("seq1", seq)]
        batch_labels, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens  # shape [1, L]

        L = batch_tokens.size(1)

        # Prepare batch of masked sequences: each position masked once
        masked_tokens = batch_tokens.repeat(L, 1)  # [L, seq_len]
        # Exclude BOS/EOS tokens from masking (mask indices 1..L-2)
        for i in range(1, L-1):
            masked_tokens[i, i] = self.alphabet.mask_idx

        # Forward pass
        logits = self.model(masked_tokens, repr_layers=[], return_contacts=False)["logits"]  # [L, seq_len, vocab_size]

        # Compute log-probabilities
        log_probs = torch.log_softmax(logits, dim=-1)

        # True token log-probabilities
        # For each row i, take log-prob of the token at position i
        token_log_probs = torch.tensor([
            log_probs[i, i, batch_tokens[0, i]].item()
            for i in range(1, L-1)
        ])

        # Sum over positions
        ll = token_log_probs.sum().item()
        # pseudo-perplexity
        return math.exp(-ll / (L - 2))  # exclude BOS/EOS

def shannon_entropy(seq):
    cnt = dict(Counter(seq))
    n = len(seq)
    ent = 0.0
    for v in cnt.values():
        p = v/n
        ent -= p * math.log2(p)
    return ent

def seg_low_complexity(seq: str, window: int = 12, entropy_thresh: float = 2.2):
    """
    Return list of (start, end) windows flagged as low complexity.
    - window: sliding window size
    - entropy_thresh: Shannon entropy (bits) threshold below which window is low-complexity
    """
    L = len(seq)
    mask = np.zeros(L, dtype=bool)
    for i in range(L - window + 1):
        w = seq[i:i+window]
        ent = shannon_entropy(w)
        if ent < entropy_thresh:
             mask[i:i+window] = True
    
    return mask.mean()*100

class NetCharge:
    def __init__(self, device='cpu'):
        charge_table_dict = {'A': 0,'C': 0,'D': -1,'E': -1,'F': 0,'G': 0,'H': +0.1,'I': 0,'K': 1,'L': 0,'M': 0,'N':0,'P':0,'Q': 0,'R': 1,'S': 0,'T': 0,'V': 0,'W': 0,'Y': 0}
        charge_table_idx = {}
        for key, value in charge_table_dict.items():
           charge_table_idx[AA_TO_INDEX[key]] = value

        self.charge_table = torch.tensor([v for k, v in sorted(charge_table_idx.items())], device=device)

    def __call__(self, data):
        return scatter(self.charge_table[data.h], data.batch, dim=0, reduce='sum')                    