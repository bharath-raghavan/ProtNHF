import torch
from torch_scatter import scatter
from nn import EnergyTransformer, ZScoreTransform, FCNN, LookupEmbedd
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F

def batch_2d_mean(tensor_2d, batch):
    dim = tensor_2d.shape[1]
    tensor = tensor_2d.flatten()
    batch = batch.repeat_interleave(dim)
    return scatter(tensor, batch, dim=0, reduce='mean')
              
class Euler(torch.nn.Module):
    def __init__(self, input_nf, hidden_nf, dt):
        super().__init__()
        self.register_buffer('dt', torch.tensor(dt))
        self.V = EnergyTransformer(input_nf, hidden_nf)
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
    def __init__(self, mean, std, embedding_dim=1280, input_nf=64, hidden_nf=128, dt=0.1, niter=4, temperature=0.7, integrator='leapfrog'):
        super().__init__()

        self.embedd = LookupEmbedd(20, input_nf) #ZScoreTransform(embedding_dim, input_nf, mean, std)
        
        if integrator == 'euler':
            self.integrator = Euler(input_nf, hidden_nf, dt)
        else:
            self.integrator = LeapFrog(input_nf, hidden_nf, dt)
        
        self.input_nf = input_nf
        self.register_buffer('niter', torch.tensor(niter))
        
        self.mu = FCNN(input_nf, hidden_nf, input_nf) # mean of Encoder Gaussian distribution
        self.stdev = FCNN(input_nf, hidden_nf, input_nf, output_activated=True) # stdev of Encoder Gaussian distribution
        self.prior = torch.distributions.Normal(0, temperature)

    def loss(self, p0, pT, q0, mean, sigma, batch):
        f = torch.distributions.Normal(mean, sigma)
        l = self.prior.log_prob(q0) + self.prior.log_prob(p0) - f.log_prob(pT)
        l_per_graph = batch_2d_mean(l, batch)
        
        return -l_per_graph.mean()
    
    def kl(self, samples):
        target = torch.distributions.Normal(torch.mean(samples), torch.std(samples))
        
        return kl_divergence(target, self.prior)
    
    def forward(self, data, loss=True):
        q = self.embedd(data.h)
        
        mean = self.mu(q)
        sigma = torch.clamp(self.stdev(q), min=1e-6) 
        eps = torch.randn_like(mean)
        p = mean + eps*sigma
    
        pT = p
        
        q.requires_grad_(True)
        
        with torch.enable_grad():
            for i in range(self.niter):
                p, q = self.integrator(p, q, data.batch)
        
        ret = [p, q]
        
        if loss:
            loss = self.loss(p, pT, q, mean, sigma, data.batch)
            #with torch.no_grad():
            #    _, qT = self.reverse(p, q, data.batch)
            #loss = loss + F.mse_loss(self.embedd.reverse(qT), data.h)
            return loss, self.kl(p), self.kl(q)
        else:
            return p, q
    
    def check_reversibility(self, data):
        qT_original = self.embedd(data.h)
        p0, q0 = self.forward(data, loss=False)
        pT, qT = self.reverse(p0, q0, data.batch)
        return qT_original, p0, q0, pT, qT
    
    def sample(self, num):
        shape = (num, self.input_nf)
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
