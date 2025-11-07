import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from performer_pytorch import SelfAttention

class ContinousOneHot(torch.nn.Module):
    def __init__(self, n_types, hidden_dim):
        super().__init__()
        self.n_types = n_types
        self.u_generator = GaussianDraw(n_types, hidden_dim)
        
    def forward(self, z):
        x = F.one_hot(z, num_classes=self.n_types).to(torch.float)
        u, log_q = self.u_generator(x)
        
        # make one_hot continous with argmax thresholding (arXiv:2102.05379) and https://github.com/vgsatorras/en_flows/blob/2fd18bf1db59184e6bdb9709fc3c02fa85fafec2/flows/dequantize.py#L120
        T = (x*u).sum(-1, keepdim=True)
        out = (x*u) + (1 - x)*(T - F.softplus(T - u))
        
        log_q = log_q - (1 - x)*F.logsigmoid(T - u)
    
        return out, log_q

    def reverse(self, x): return torch.argmax(x, dim=-1)
            
class GaussianDraw(nn.Module):
    def __init__(self, dim, hidden_nf):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_nf),
                    nn.Softplus(),
                    nn.Linear(hidden_nf, hidden_nf),
                    nn.Softplus(),
                    nn.Linear(hidden_nf, dim*2)) # Gaussian distribution to get mean and sigma
    
    def forward(self, x):        
        params = self.net(x)
        mu, log_sigma = params.chunk(2, dim=1)
        sigma = torch.exp(log_sigma)
        noise_dist = torch.distributions.Normal(mu, sigma)
        u = noise_dist.rsample()  # reparameterized sample for backprop to flow
        log_noise = noise_dist.log_prob(u)

        return u, log_noise

class TransformerLayer(nn.Module): # transformer should not have dropout, as that is random and breaks NF reversibility
    def __init__(self, d_model, n_heads, ff_dim):
        super().__init__()
        self.attn = SelfAttention(dim=d_model, heads=n_heads, causal=False)
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

class TransformerEmbedding(nn.Module):
    def __init__(self, n_types, d_model):
        super().__init__()
        self.d_model = d_model
        self.projection_layer = nn.Linear(n_types, self.d_model)
    
    @staticmethod
    def pad_batch(h, batch):
        embeddings = [h[batch==i] for i in batch.unique()]

        # Pad sequences to max length in this batch
        padded = pad_sequence(embeddings, batch_first=True)  # [B, max_len, emb_dim]

        # Build mask (True = padding)
        lengths = [e.size(0) for e in embeddings]
        max_len = padded.size(1)
        mask = torch.zeros(len(embeddings), max_len, dtype=torch.bool)
        for i, l in enumerate(lengths):
            mask[i, l:] = True
        
        padded = padded.to(h.device)
        mask = (~mask).to(h.device) # Change to False = padding
        return padded, mask
    
    def position_encoder(self, max_seq_length):
        pe = torch.zeros(max_seq_length, self.d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)
            
    def forward(self, x, batch):
        x = self.projection_layer(x)*math.sqrt(self.d_model) # scaling by sqrt(d_model) needed so that embedding doesn't get overwhelmed by pos encoding
        x, mask = TransformerEmbedding.pad_batch(x, batch)
        pe = self.position_encoder(x.size(1)).to(x.device)
        x = x + (pe).requires_grad_(False)

        return x, mask

class EnergyHead(nn.Module):
    def __init__(self, d_model, ff_dim):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, 1)  # scalar output
        )
    
    def forward(self, x, mask):
        # Pooling (mean over unmasked positions)
        if mask is not None:
            lengths = mask.sum(dim=1, keepdim=True)  # count valid tokens
            pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / lengths
        else:
            pooled = x.mean(dim=1)

        return self.ff(pooled).squeeze(-1)  # [batch]
    
                        
class EmbeddingEnergy(nn.Module):
    def __init__(self, n_types, d_model, ff_dim, n_heads, n_layers):
        super().__init__()
                
        self.embedding = TransformerEmbedding(n_types, d_model)
        
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, ff_dim) for _ in range(n_layers)
        ])
        # Energy prediction head
        self.energy_layer = EnergyHead(d_model, ff_dim)
    
    def forward(self, x, batch):
        x, mask = self.embedding(x, batch)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.energy_layer(x, mask) # Predict scalar energy
