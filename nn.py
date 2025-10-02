import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class LookupEmbedd(torch.nn.Module):
    def __init__(self, types, dim):
        super().__init__()
        self.network = torch.nn.Embedding(types, dim)
        
    def forward(self, z):
        return self.network(z)
    
    def reverse(self, x):
        emb_weights = self.network.weight
        output = x.unsqueeze(0)
        
        emb_size = output.size(0), output.size(1), -1, -1
        out_size = -1, -1, emb_weights.size(0), -1
        z = torch.argmin(torch.abs(output.unsqueeze(2).expand(out_size) -
                                        emb_weights.unsqueeze(0).unsqueeze(0).expand(emb_size)).sum(dim=3), dim=2)[0]
                                            
        return z
        
class ZScoreTransform(torch.nn.Module):
    def __init__(self, input_dim, output_dim, mean, std, eps=1e-6):
        super().__init__()
        self.in_layer = nn.Linear(input_dim, output_dim)
        self.out_layer = nn.Linear(output_dim, input_dim)
        self.mean = mean
        self.std = std + eps

    def forward(self, x):
        """Apply z-score normalization."""
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return self.in_layer( (x - mean) / std )

    def reverse(self, x_norm):
        """Invert normalization (back to original scale)."""
        mean = self.mean.to(x_norm.device)
        std = self.std.to(x_norm.device)
        return self.out_layer(x_norm) * std + mean
        
class FCNN(torch.nn.Module):
    def __init__(self, input_nf, hidden_nf, output_nf, output_activated=False, n_layers=1, act_fn=torch.nn.Softplus()):
        super().__init__()
        self.network = torch.nn.Sequential()
        self.network.add_module("layer0", torch.nn.Linear(input_nf, hidden_nf))
        self.network.add_module("act0", act_fn)
        for i in range(0, n_layers):
            self.network.add_module(f"layer{i+1}", torch.nn.Linear(hidden_nf, hidden_nf))
            self.network.add_module(f"act{i+1}", act_fn)
        self.network.add_module(f"layer{n_layers+1}", torch.nn.Linear(hidden_nf, output_nf))
        if output_activated:
            self.network.add_module(f"act{n_layers+1}", act_fn)
        
    def forward(self, x):
         return self.network(x)

def graph_to_transformer_input(h, batch):
    embeddings = [h[batch==i] for i in batch.unique()]
    
    # Pad sequences to max length in this batch
    padded = pad_sequence(embeddings, batch_first=True)  # [B, max_len, emb_dim]

    # Build mask (True = padding)
    lengths = [e.size(0) for e in embeddings]
    max_len = padded.size(1)
    mask = torch.zeros(len(embeddings), max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, l:] = True

    return padded.to(h.device), mask.to(h.device)
             
# Simple Performer-style attention (kernelized linear attention)
class LinearAttention(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def feature_map(self, x):
        return F.elu(x) + 1  # positive kernel feature

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head)

        q, k = self.feature_map(q), self.feature_map(k)

        # Compute linear attention: Q(K^T V)
        kv = torch.einsum("blhd,blhe->bhde", k, v)   # (B, H, D_head, D_head)
        z = 1 / (torch.einsum("blhd,bhd->blh", q, k.sum(dim=1)) + 1e-6)
        out = torch.einsum("blhd,bhde->blhe", q, kv) * z.unsqueeze(-1)
        out = out.reshape(B, L, D)
        return self.out_proj(out)

class TransformerLayer(nn.Module): # transformer should not have dropout, as that is random and breaks NF reversibility
    def __init__(self, d_model=1280, n_heads=4, ff_dim=512):
        super().__init__()
        self.attn = LinearAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.ff(self.norm2(x))
        
        return x
        
class EnergyTransformer(nn.Module):
    def __init__(self, emb_dim=1280, hidden_dim=512, n_heads=4, energy_dim=10, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(emb_dim, n_heads, hidden_dim) for _ in range(n_layers)
        ])
        # Energy prediction head
        self.energy_head = nn.Sequential(
            nn.Linear(emb_dim, energy_dim),
            nn.ReLU(),
            nn.Linear(energy_dim, 1)  # scalar output
        )

    def forward(self, x, batch):
        x, mask = graph_to_transformer_input(x, batch)
        
        for layer in self.layers:
            x = layer(x, mask)
        # Pooling (mean over unmasked positions)
        if mask is not None:
            lengths = (~mask).sum(dim=1, keepdim=True)  # count valid tokens
            pooled = (x * (~mask).unsqueeze(-1)).sum(dim=1) / lengths
        else:
            pooled = x.mean(dim=1)

        # Predict scalar energy
        energy = self.energy_head(pooled).squeeze(-1)  # [batch]
        return energy
