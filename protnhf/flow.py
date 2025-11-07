import torch
from torch_scatter import scatter
from .nn import EmbeddingEnergy, GaussianDraw, ContinousOneHot
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F

def batch_2d_mean(tensor_2d, batch):
    dim = tensor_2d.shape[1]
    tensor = tensor_2d.flatten()
    batch = batch.repeat_interleave(dim)
    return scatter(tensor, batch, dim=0, reduce='mean')
              
class Euler(torch.nn.Module):
    def __init__(self, input_nf, dt, d_model, ff_dim, n_heads, n_layers):
        super().__init__()
        self.register_buffer('dt', torch.tensor(dt))
        self.V = EmbeddingEnergy(input_nf, d_model, ff_dim, n_heads, n_layers)
        self.register_parameter(name='a', param=torch.nn.Parameter(1*torch.eye(1)))

    def diff(self, y, x):
        grad_outputs = [torch.ones_like(y)]
        dy = torch.autograd.grad(
               [y],
               [x],
               grad_outputs=grad_outputs,
               create_graph=True,
               retain_graph=True,
           )[0]
        if dy is None:
           raise RuntimeError(
               "Autograd returned None for the force prediction.")
        return dy

    def forward(self, p, q, batch):
        p = p - self.diff(self.V(q, batch), q)*self.dt
        q = q + self.a**2*p*self.dt
        return p, q

    def reverse(self, p, q, batch):
        q = q - self.a**2*p*self.dt
        p = p + self.diff(self.V(q, batch), q)*self.dt
        return p, q

class LeapFrog(Euler):
    def forward(self, p, q, batch):
        p = p - self.diff(self.V(q, batch), q)*self.dt/2
        q = q + self.a**2*p*self.dt
        p = p - self.diff(self.V(q, batch), q)*self.dt/2
        return p, q
    
    def reverse(self, p, q, batch):
        p = p + self.diff(self.V(q, batch), q)*self.dt/2
        q = q - self.a**2*p*self.dt
        p = p + self.diff(self.V(q, batch), q)*self.dt/2
        return p, q
                
class Flow(torch.nn.Module):
    def __init__(self, n_types, hidden_dims, dt, niter, std, integrator, transformer_d_model, transformer_ff_dim, n_transformer_heads, n_transformer_layers):
        super().__init__()

        self.n_types = n_types
        self.embedd = ContinousOneHot(n_types, hidden_dims)
        
        if integrator == 'euler':
            self.integrator = Euler(self.n_types, dt, transformer_d_model, transformer_ff_dim, n_transformer_heads, n_transformer_layers)
        else:
            self.integrator = LeapFrog(self.n_types, dt, transformer_d_model, transformer_ff_dim, n_transformer_heads, n_transformer_layers)
        
        self.register_buffer('niter', torch.tensor(niter))
        
        self.p_generator = GaussianDraw(self.n_types, hidden_dims)
        self.prior = torch.distributions.Normal(0, std)

    def loss(self, p0, q0, log_j, batch):
        l = self.prior.log_prob(q0) + self.prior.log_prob(p0) - log_j
        l_per_graph = batch_2d_mean(l, batch)
        
        return -l_per_graph.mean()
    
    def kl(self, samples):
        target = torch.distributions.Normal(torch.mean(samples), torch.std(samples))
        
        return kl_divergence(target, self.prior)
    
    def forward(self, data, train=True):
        q, log_qT  = self.embedd(data.h)
        qT = q # for reversiblity checking only
        
        p, log_pT = self.p_generator(q)
        log_j = log_qT + log_pT
        q.requires_grad_(True)
        
        with torch.enable_grad():
            for i in range(self.niter):
                p, q = self.integrator(p, q, data.batch)
        
        if train:
            return self.loss(p, q, log_j, data.batch), self.kl(p), self.kl(q)
        else:
            return p, q, qT
    
    def check_reversibility(self, data):
        p0, q0, qT_original = self.forward(data, train=False)
        pT, qT = self.reverse(p0, q0, data.batch)
        return qT_original, p0, q0, pT, qT
    
    def sample(self, num):
        shape = (num, self.n_types)
        p = self.prior.sample(sample_shape=shape)
        q = self.prior.sample(sample_shape=shape)
        
        batch = torch.zeros(num, dtype=torch.int64) # only one sample requested
        
        with torch.no_grad():
            return self.embedd.reverse(self.reverse(p, q, batch)[1])
    
    def reverse(self, p, q, batch):
        q.requires_grad_(True)
        
        with torch.enable_grad():
            for i in range(self.niter):
                p, q = self.integrator.reverse(p, q, batch)
    
        return p, q
