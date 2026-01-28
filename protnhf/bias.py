from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from .dataset import AA_TO_INDEX

class CombinedBias:
    def __init__(self, biases):
        self.biases = biases
    
    def __call__(self, x, batch):
        bias = 0
        for i in self.biases:
            bias = bias + i(x, batch)
        
        return bias
        
class CoulombBias:
    def __init__(self, k, eps, residue):
        aa = torch.tensor([AA_TO_INDEX[residue]])
        self.xO = F.one_hot(aa, num_classes=20).to(torch.float)
        self.k = k
        self.eps = eps

    def __call__(self, x, batch):
        r = x - self.xO
        dist = torch.sqrt((r**2).sum(dim=1)+self.eps)
        U = self.k / dist
        return scatter(U, batch, dim=0, reduce='sum')

class GaussianBias:
    def __init__(self, k, sigma, residue):
        aa = torch.tensor([AA_TO_INDEX[residue]])
        self.xO = F.one_hot(aa, num_classes=20).to(torch.float)
        self.k = k
        self.sigma = sigma

    def __call__(self, x, batch):
        r = x - self.xO
        dist2 = (r**2).sum(dim=1)
        U = self.k * torch.exp(-0.5*dist2/self.sigma**2)
        return scatter(U, batch, dim=0, reduce='sum')

class TanHBias:
    def __init__(self, k, w, xO):
        self.xO = xO.mean(dim=0)
        self.k = k
        self.v = w / torch.norm(w)  # unit vector

    def __call__(self, x, batch):
        proj = (x-self.xO) @ self.v
        U = self.k * torch.tanh(proj)
        return scatter(U, batch, dim=0, reduce='sum')