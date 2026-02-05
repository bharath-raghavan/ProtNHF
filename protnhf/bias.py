import torch
from torch_scatter import scatter

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
