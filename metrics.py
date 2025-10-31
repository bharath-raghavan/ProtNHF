import torch
import esm

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