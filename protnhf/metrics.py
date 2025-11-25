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
        _, _, tokens = self.batch_converter([('protein', seq)])

        # Masking one token at a time
        N = tokens.size(1)
        log_probs = []
        for i in range(N):
            masked_tokens = tokens.clone()
            masked_tokens[0, i] = self.alphabet.mask_idx  # mask ith token
            with torch.no_grad():
                logits = self.model(masked_tokens, repr_layers=[33])["logits"]
            # Softmax to get probabilities for the original token
            prob = torch.softmax(logits[0, i], dim=-1)[tokens[0, i]]
            log_probs.append(torch.log(prob))
        # Pseudo-perplexity
        pppl = torch.exp(-torch.stack(log_probs).sum() / (N-2))
        return pppl.item()

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
                    