import torch
from torch_scatter import scatter
from .dataset import AA_TO_INDEX

charge_dict = {
    'D': -1.0,
    'E': -1.0,
    'K': +1.0,
    'R': +1.0,
    'H': +0.1
}

charge_vector = torch.tensor(
    [charge_dict.get(aa, 0.0) for aa in AA_TO_INDEX.keys()],
    dtype=torch.float32
)

def net_charge(x, batch):
    probs = torch.softmax(x, dim=-1)
    qs = (probs * charge_vector).sum(dim=1)
    return scatter(qs, batch, dim=0, reduce='sum')

class CombinedBias:
    def __init__(self, biases):
        self.biases = biases
    
    def __call__(self, x, batch):
        bias = 0
        for i in self.biases:
            bias = bias + i(x, batch)
        
        return bias
        
class CoulombBias:
    def __init__(self, k, xO):
        self.xO = xO
        self.k = k
        self.eps = 1e-6

    def __call__(self, x, batch):
        r = x - self.xO
        dist = torch.sqrt((r**2).sum(dim=1)+self.eps)
        U = self.k / dist
        return scatter(U, batch, dim=0, reduce='sum')

class GaussianBias:
    def __init__(self, k, sigma, xO):
        self.xO = xO
        self.k = k
        self.sigma = sigma

    def __call__(self, x, batch):
        r = x - self.xO
        dist2 = (r**2).sum(dim=1)
        U = self.k * torch.exp(-0.5*dist2/self.sigma**2)
        return scatter(U, batch, dim=0, reduce='sum')

class PositionRestraint:
    def __init__(self, k, i, xO):
        self.xO = xO
        self.k = k
        self.i = i

    def __call__(self, x, batch):
        dx = x - self.xO
        U = 0.5 * self.k * (dx**2).sum(dim=1)
    
        counts = torch.bincount(batch)              # (num_proteins,)

        # residue index within each protein
        res_idx = torch.arange(x.size(0), device=x.device) \
               - torch.repeat_interleave(
                   torch.cumsum(counts, dim=0) - counts,
                   counts
               )
               
        mask = (res_idx == self.i)
        U = U * mask.float()
        return scatter(U, batch, dim=0, reduce='sum')

class NetChargeRestraint:
    def __init__(self, k, qO):
        self.qO = qO
        self.k = k

    def __call__(self, x, batch):
        q = net_charge(x, batch)
        dx = q - self.qO
        U = 0.5 * self.k * (dx**2)
        return U
