import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class ZScoreTransform(torch.nn.Module):
    def __init__(self, mean, std, eps=1e-6):
        super().__init__()
        self.mean = mean
        self.std = std + eps

    def forward(self, x):
        """Apply z-score normalization."""
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std

    def reverse(self, x_norm):
        """Invert normalization (back to original scale)."""
        mean = self.mean.to(x_norm.device)
        std = self.std.to(x_norm.device)
        return x_norm * std + mean
        
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
             
class EmbeddingForce(torch.nn.Module):
    def __init__(self, emb_dim=1280, hidden_dim=256, n_heads=4, n_layers=2):
        super().__init__()

        # Transformer encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
             d_model=emb_dim,
             nhead=n_heads,
             dim_feedforward=hidden_dim,
             batch_first=True  # [batch, seq_len, dim]
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Per-residue output head
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, emb_dim)
        )

    def forward(self, x, batch):
        embeddings, mask = graph_to_transformer_input(x, batch)
        
        # In PyTorch, mask = True means ignore, so we invert it for key_padding_mask
        if mask is not None:
            key_padding_mask = mask
        else:
            key_padding_mask = None

        # Encode sequence
        z = self.transformer(embeddings, src_key_padding_mask=key_padding_mask)
        # Apply MLP per residue
        forces = self.mlp(z)  # [batch, seq_len, emb_dim]

        # Flatten valid residues to [total_residues, emb_dim]
        if mask is not None:
            # mask = True for padding
            valid = ~mask
            forces_flat = forces[valid]  # only keep valid residues
        else:
            forces_flat = forces.view(-1, forces.size(-1))

        return forces_flat
